#!/usr/bin/env python3
"""Full-epoch, randomly initialized LitePT + AGILE3D ScanNet40 baseline."""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import run_scannet40_joint as joint
joint._load_train_runtime()
from scratch_clicks import install as install_parallel_clicks
install_parallel_clicks()
from delimit3d.models.litept_wrapper import LitePTBackbone
from delimit3d.evaluation.agile3d_decoder import Agile3DClickDecoder, click_loss_weights

DECODER = dict(feature_dim=72, hidden_dim=128, num_heads=8, dim_feedforward=1024,
               num_decoders=3, num_bg_queries=10, dropout=0., pre_norm=False,
               max_click_events=200, normalize_pos_enc=True, gauss_scale=1., aux=True)
PROTOCOL = dict(precision=dict(amp=True, initial_scale=1., growth_interval=1000000),
                protocol=dict(click_center_method='kdtree'))

def official_dice(logits, target, weights):
    probability = logits.softmax(1)
    one_hot = F.one_hot(target.long(), logits.shape[1]).to(probability.dtype)
    numerator = 2 * (probability * one_hot).mean(1)
    denominator = (probability + one_hot).mean(1)
    score = (numerator + 1e-6) / (denominator + 1e-6)
    return (torch.where(numerator > 1e-6, 1-score, score*0) * weights).mean()

def official_loss(outputs, target, xyz, clicks):
    weights = click_loss_weights(xyz, clicks)
    predictions = [outputs['pred_masks']] + [o['pred_masks'] for o in outputs.get('aux_outputs', [])]
    return sum((F.cross_entropy(p.float(), target, reduction='none')*weights).mean()
               + 2*official_dice(p.float(), target, weights) for p in predictions)

def augment(raw):
    # Upstream AGILE3D XY flips and two Z rotations, applied to normals as well.
    transform = np.eye(3, dtype=np.float32)
    for axis in (0, 1):
        if random.random() < .5:
            transform[axis, axis] = -1
    angle = random.choice((0, .5*np.pi, np.pi, 1.5*np.pi)) + random.uniform(-np.pi, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    transform = transform @ np.asarray([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    coord, _ = joint._center_shift_numpy(raw['coord'] @ transform)
    feat = raw['features'].copy()
    feat[:, 3:6] = feat[:, 3:6] @ transform
    return np.ascontiguousarray(coord), np.ascontiguousarray(feat)

def train_batch(encoder, decoder, raws, optimizer, scaler):
    encoder.train(); decoder.train(); optimizer.zero_grad(set_to_none=True)
    arrays = [augment(raw) for raw in raws]
    offsets = np.cumsum([len(a[0]) for a in arrays])
    coords = torch.from_numpy(np.concatenate([a[0] for a in arrays])).cuda()
    feats = torch.from_numpy(np.concatenate([a[1] for a in arrays])).cuda()
    torch.cuda.synchronize(); encoder_start = time.monotonic()
    with joint._amp_context(PROTOCOL):
        output = encoder(coords, feats, torch.as_tensor(offsets, device='cuda'))
    torch.cuda.synchronize(); encoder_seconds = time.monotonic()-encoder_start
    interaction_seconds = 0.
    token_offsets = output.scene_token_offsets.cpu().tolist()
    token_start = point_start = 0
    losses = []
    rounds = random.randint(0, 19)
    for i, raw in enumerate(raws):
        stop = int(token_offsets[i])
        tokens = output.scene_tokens[token_start:stop]
        xyz = output.scene_xyz[token_start:stop].detach().float()
        reps = output.representative_indices[token_start:stop].detach().cpu().numpy() - point_start
        all_ids = [int(o['instance']) for o in raw['objects']]
        memberships = joint.build_token_targets(reps, raw['masks'], all_ids)
        surviving = [k for k in all_ids if len(memberships[k])]
        if not surviving:
            raise RuntimeError(f"No foreground tokens: {raw['scene']}")
        ids = random.sample(surviving, random.randint(1, min(10, len(surviving))))
        target = np.zeros(len(tokens), dtype=np.int64)
        for label, obj in enumerate(ids, 1):
            target[memberships[obj]] = label
        interaction_start = time.monotonic()
        clicks, times = joint.protocol.initial_clicks(target, xyz.cpu().numpy(), random.randrange(2**31))
        target = torch.from_numpy(target).cuda()
        clicks, times, _ = joint._prefix_interaction(decoder, tokens.detach(), xyz, target, clicks, times, rounds, PROTOCOL)
        interaction_seconds += time.monotonic()-interaction_start
        with joint._amp_context(PROTOCOL):
            prediction = decoder(tokens, xyz, clicks=clicks, click_times=times)
        losses.append(official_loss(prediction, target, xyz, clicks))
        token_start = stop; point_start = int(offsets[i])
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss: {loss.item()}')
    torch.cuda.synchronize(); backward_start = time.monotonic()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_encoder = joint._grad_l2(list(encoder.parameters()))
    grad_decoder = joint._grad_l2(list(decoder.parameters()))
    if not np.isfinite(grad_encoder + grad_decoder) or min(grad_encoder, grad_decoder) <= 0:
        raise FloatingPointError(f'Invalid gradients: {grad_encoder}, {grad_decoder}')
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()), .1)
    scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    return dict(encoder_seconds=encoder_seconds, interaction_seconds=interaction_seconds,
                backward_seconds=time.monotonic()-backward_start, loss=float(loss.detach()), encoder_grad=grad_encoder, decoder_grad=grad_decoder,
                simulation_rounds=rounds, tokens=token_offsets[-1])

def validate(encoder, decoder, manifest, output_dir, epoch, smoke=False):
    from run_scannet40_joint_eval import evaluate_episode
    encoder.eval(); decoder.eval()
    records = {r['scene']: r for r in manifest['validation_scenes']}
    episodes = manifest['panels']['MO'][:1] if smoke else manifest['panels']['MO']
    inputs = joint.SceneInputCache(1)
    metrics = []
    config = dict(seed=42, protocol=dict(click_budget=20, click_center_method='kdtree'))
    for episode in episodes:
        record = records[episode['scene']]
        raw = inputs.get(record)
        with torch.inference_mode():
            encoded = encoder(torch.from_numpy(raw['coord']).cuda(), torch.from_numpy(raw['features']).cuda())
            reps = encoded.representative_indices.cpu().numpy()
            object_ids = [int(o['instance']) for o in raw['objects']]
            memberships = joint.build_token_targets(reps, raw['masks'], object_ids)
            raw_labels = np.zeros(len(raw['coord']), dtype=np.int64)
            for obj in object_ids: raw_labels[raw['masks'][obj]] = obj
            cached = dict(features=encoded.scene_tokens.float(), scene_xyz=encoded.scene_xyz.float(),
                scene_xyz_cpu=encoded.scene_xyz.float().cpu(), inverse_map=encoded.inverse_map.cpu().numpy(),
                object_token_indices={str(k):torch.from_numpy(v) for k,v in memberships.items()},
                objects=raw['objects'], raw_labels=raw_labels)
            trace = evaluate_episode(decoder, cached, record, episode, config, 'MO')
            metrics.append(trace['metrics'])
    report = dict(epoch=epoch, panel='MO', scenes=len(metrics),
        metrics={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]})
    joint.dump(output_dir/f'eval_epoch_{epoch:04d}.json', report)
    print(json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path('/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1/selection_manifest.json'))
    parser.add_argument('--litept-root', default=os.environ.get('LITEPT_ROOT', '/cluster/work/igp_psr/nedela/LitePT'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--grid-size', type=float, default=.02, choices=(.02,.05))
    parser.add_argument('--benchmark-batches', type=int, default=0)
    parser.add_argument('--stop-after-epochs', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    records = manifest['train_scenes']
    if len(records) != 1200:
        raise ValueError('Expected 1200 official ScanNet40 training scenes')
    contract = dict(schema='litept_agile3d_scratch_v1', epochs=1100, batch_scenes=5,
        seed=42, lr=1e-4, weight_decay=1e-4, lr_drop_after_epoch=1000, clip_norm=.1,
        encoder_initialization='random', decoder_initialization='random',
        voxel_size=args.grid_size, benchmark_batches=args.benchmark_batches, input_features='rgbn6', manifest_sha256=joint.file_sha256(args.manifest),
        decoder=DECODER, smoke=args.smoke)
    contract_path = args.output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Run contract changed')
    joint.dump(contract_path, contract)
    joint.set_seed(42)
    encoder = LitePTBackbone(litept_root=args.litept_root, in_channels=6, grid_size=args.grid_size,
        litept_variant='litept_s_star', multi_scale=False, voxel_reduce='representative',
        representative_sampling='first', cache_training_voxelization=False).cuda()
    decoder = Agile3DClickDecoder(**DECODER).cuda()
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=.1)
    scaler = joint._make_scaler(PROTOCOL)
    latest = args.output/'latest.pt'
    epoch0 = 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['contract'] != contract: raise ValueError('Checkpoint contract mismatch')
        encoder.load_state_dict(saved['encoder'], strict=True)
        decoder.load_state_dict(saved['decoder'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler']); joint.restore_rng(saved['rng'])
        epoch0 = saved['epoch']
    else:
        initial = dict(encoder=joint._cpu_state(encoder), decoder=joint._cpu_state(decoder), contract=contract)
        joint.save_torch(args.output/'initial_random.pt', initial)
        joint.dump(args.output/'initialization.json', dict(encoder_sha256=joint.state_sha256(initial['encoder']),
            decoder_sha256=joint.state_sha256(initial['decoder']), public_weights_loaded=False))
    cache = joint.SceneInputCache(8)
    started = time.monotonic()
    for epoch in range(epoch0+1, 1101):
        # Independent epoch RNG makes walltime boundaries reproducible.
        joint.set_seed(42+epoch)
        order = list(range(len(records))); random.shuffle(order)
        if args.smoke: order = order[:10]
        if args.benchmark_batches: order = order[:5*args.benchmark_batches]
        begin = time.monotonic(); results = []
        for batch_index in range(0, len(order), 5):
            tick = time.monotonic()
            result = train_batch(encoder, decoder, [cache.get(records[k]) for k in order[batch_index:batch_index+5]], optimizer, scaler)
            result.update(epoch=epoch, batch=batch_index//5, seconds=time.monotonic()-tick,
                          lr=optimizer.param_groups[0]['lr'], peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            with (args.output/'updates.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
            print(json.dumps(result), flush=True); results.append(result)
        scheduler.step()
        if not args.benchmark_batches and (args.smoke or epoch % 50 == 0 or epoch == 1100):
            validate(encoder, decoder, manifest, args.output, epoch, args.smoke)
        payload = dict(epoch=epoch, encoder=encoder.state_dict(), decoder=decoder.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            rng=joint.rng_state(), contract=contract)
        if not args.benchmark_batches:
            joint.save_torch(latest, payload)
        if not args.benchmark_batches and (epoch % 50 == 0 or epoch == 1100):
            joint.save_torch(args.output/f'epoch_{epoch:04d}.pt', dict(
                epoch=epoch, encoder=payload['encoder'], decoder=payload['decoder'],
                contract=contract, checkpoint_kind='evaluation_weights_only'))
        joint.dump(args.output/'status.json', dict(epoch=epoch, complete=epoch==1100,
            epoch_seconds=time.monotonic()-begin, mean_loss=np.mean([r['loss'] for r in results])))
        if args.smoke or args.benchmark_batches or (args.stop_after_epochs and epoch >= args.stop_after_epochs) or time.monotonic()-started > 108*3600:
            break

if __name__ == '__main__': main()
