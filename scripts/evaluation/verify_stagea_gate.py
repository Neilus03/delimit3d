#!/usr/bin/env python3
"""Independently audit only the final resumed Stage-A MO-5 endpoint on Euler.

Reads frozen Stage-A artifacts; writes only lead_audit/stagea. Never launches
training or evaluation, and never falls back to the interim result.
Exit 0=gate passed, 3=gate failed/mixed, 2=pending, 4=integrity failure.
"""
from __future__ import annotations
import argparse, copy, datetime, hashlib, json, math, pickletools, tempfile, zipfile
from pathlib import Path
import numpy as np

BASE = Path("/cluster/work/igp_psr/nedela/delimit3d_scannetpp_agile3d_mo_v1")
ROOT = Path(str(BASE) + "_retry_20260912")
AUDIT = Path("/cluster/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1/lead_audit/stagea")
EXPERIMENT = "delimit3d_scannetpp_agile3d_mo_v1"
COMMIT = "3df5fdb92df01ba60490859d7a04ddd848b8220a"
SEED, SAMPLES = 20260911, 10000
ARMS = ("public", "delimit3d")
EXPECTED_ENCODERS = {
    "public": "514603ae4f715f3010a2b8cdff452e644159408b3781aae9cd7b5e1e7b82a4fd",
    "delimit3d": "ea2001e33ea9638b9f402ab0a977e13e8f08acc0e795083cb17ccbda330c9585",
}
PINNED = {
    BASE/"selection_manifest.json": "9814d184679f3e25363b3a90be198f4be3d33307aff814443619b54722995cbd",
    BASE/"train_episodes.jsonl": "acdc7bcccc6fbeb83eb92c115a78b9edf92b10976f4b73e93d59d1ad8e7d59d5",
    BASE/"decoder_init.pt": "2fb71382c580a101b5e5ea6d1b2441521deb471bf025f1274c8572d09f2d7766",
    BASE/"geometry_comparison.json": "5df90c0b6de804e96dbabee1f2054bbd38499f256204678a3d85f151be45a07a",
    BASE/"cache_report_public.json": "a2a9a6898c9e669a1975e8c01d19dfbfda350e84cb7ffacd1fd449ba38c37f0d",
    BASE/"cache_report_delimit3d.json": "4ee1c1b0ca289c212fb5a302f04f1cecd82646d38d1e83ef1897df4533aa8b49",
    BASE/"freeze/environment.json": "fe694335e5d765c798720453967156b697deff21f8fec57c8931df24c039154b",
    ROOT/"freeze/resolved_config.yaml": "8ce3777a8a3dfb42edf7d680a08f485a667aaf30003916f714ac6ca27e4ff7cc",
    ROOT/"freeze/retry_source_archive_3df5fdb92df0.tar.gz": "6801068fd954e788d2e3eb4b33989d90ff2aa5ea595174e1946c2cc938408ba1",
    Path("/cluster/work/igp_psr/nedela/public_rgbn6_frozen_pg_20260907_r1/assets/public_checkpoint.pth"): "86408c6371555aeee2a8eda1184b55a411f9748784c275349149b509b661d518",
    Path("/cluster/work/igp_psr/nedela/public_rgbn6_frozen_pg_20260907_r1/results/posttraining_256_r4/checkpoints/audit_e0001_u00000256.pt"): "4717369861cff81fe784efda8cd8a7653de46d35635730f10c893da563050282",
}
ENCODER_FILES = {
    arm: path for arm, path in zip(ARMS, list(PINNED)[-2:])
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def source_path(path):
    return Path(str(path).replace("/tmp/euler_cluster_nedela_rw/", "/cluster/", 1))

def required_paths(root):
    paths = []
    for arm in ARMS:
        paths.extend(root/arm/p for p in (
            "train_report.json", "decoder_u05000.pt", "evaluation_report.json",
            "evaluation/MO-5/scenes.jsonl", "evaluation/MO-5/episodes.jsonl",
        ))
    return paths

def scalar_checkpoint_metadata(path):
    # Inspect primitive metadata without executing pickle globals or loading tensors.
    names = {"experiment_id", "arm", "update", "repo_commit", "encoder_frozen",
             "encoder_tensor_sha256", "decoder_tensor_sha256",
             "decoder_initialization_sha256", "episode_schedule_sha256"}
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith("/data.pkl")]
        require(len(members) == 1, "checkpoint has ambiguous pickle member")
        ops = list(pickletools.genops(z.read(members[0])))
    result = {}
    for i, (op, value, pos) in enumerate(ops[:-1]):
        if not isinstance(value, str) or value not in names or value in result:
            continue
        for nxt, item, _ in ops[i+1:i+5]:
            if nxt.name in {"MEMOIZE", "BINPUT", "LONG_BINPUT"}:
                continue
            if nxt.name in {"NEWTRUE", "NEWFALSE"}:
                result[value] = nxt.name == "NEWTRUE"
            elif isinstance(item, (str, int, float)):
                result[value] = item
            break
    require(set(result) == names, "checkpoint metadata extraction incomplete")
    return result

def validate_arm(arm, report, metadata, checkpoint_hash, evaluation, cache, manifest, root):
    require(report.get("arm") == arm and report.get("updates") == 5000,
            arm + ": final train report must identify update 5000")
    require(report.get("experiment_id") == EXPERIMENT, arm + ": experiment mismatch")
    require(report.get("repo_commit") == COMMIT, arm + ": frozen source commit mismatch")
    require(source_path(report.get("checkpoint", "")) == root/arm/"decoder_u05000.pt",
            arm + ": report does not identify this final retry checkpoint")
    require(report.get("checkpoint_sha256") == checkpoint_hash, arm + ": final checkpoint hash mismatch")
    expected = EXPECTED_ENCODERS[arm]
    require(report.get("encoder_state_unchanged") is True, arm + ": frozen encoder flag failed")
    require(report.get("encoder_tensor_sha256_before") == expected ==
            report.get("encoder_tensor_sha256_after") ==
            cache.get("encoder_tensor_sha256_before") ==
            cache.get("encoder_tensor_sha256_after"), arm + ": encoder hash invariant failed")
    require(cache.get("encoder_state_unchanged") is True, arm + ": cache frozen invariant failed")
    init_hash = manifest["decoder_initialization"]["tensor_state_sha256"]
    schedule = manifest["train_episodes_sha256"]
    require(report.get("decoder_initialization_sha256") == init_hash, arm + ": initialization mismatch")
    require(report.get("episode_schedule_sha256") == schedule, arm + ": schedule mismatch")
    require(report.get("resumed_from_update") == 1000 and report.get("rng_state_restored") is False,
            arm + ": explicit non-bit-exact update1000 resume provenance absent")
    expected_meta = {"experiment_id":EXPERIMENT, "arm":arm, "update":5000,
                     "repo_commit":COMMIT, "encoder_frozen":True,
                     "encoder_tensor_sha256":expected,
                     "decoder_initialization_sha256":init_hash,
                     "episode_schedule_sha256":schedule,
                     "decoder_tensor_sha256":report.get("decoder_tensor_sha256")}
    for key, value in expected_meta.items():
        require(metadata.get(key) == value, arm + ": checkpoint metadata mismatch " + key)
    require(evaluation.get("arm") == arm and evaluation.get("repo_commit") == COMMIT,
            arm + ": evaluation identity/source mismatch")
    e = evaluation.get("reports", {}).get("MO-5", {})
    require(e.get("arm") == arm and e.get("panel") == "MO-5" and e.get("experiment_id") == EXPERIMENT,
            arm + ": MO-5 report absent/invalid")
    require(source_path(e.get("decoder_checkpoint", "")) == root/arm/"decoder_u05000.pt",
            arm + ": evaluation used a different checkpoint")
    require(e.get("decoder_checkpoint_sha256") == checkpoint_hash,
            arm + ": evaluation checkpoint hash mismatch")
    require(e.get("encoder_state_unchanged") is True and e.get("encoder_tensor_sha256") == expected,
            arm + ": evaluation encoder invariant failed")
    require(e.get("encoder_checkpoint_sha256") == PINNED[ENCODER_FILES[arm]],
            arm + ": evaluation encoder checkpoint file hash mismatch")
    require(source_path(e.get("scene_rows", "")) == root/arm/"evaluation/MO-5/scenes.jsonl",
            arm + ": scene row path mismatch")
    require(source_path(e.get("episode_rows", "")) == root/arm/"evaluation/MO-5/episodes.jsonl",
            arm + ": episode row path mismatch")
    return e

def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(bool(rows), "empty rows: " + str(path))
    keys = [str(row["scene"]) for row in rows]
    require(len(keys) == len(set(keys)), "duplicate scenes: " + str(path))
    return {str(row["scene"]):row for row in rows}

def paired_bootstrap(public, adapted):
    require(set(public) == set(adapted) and bool(public), "paired scene identity mismatch")
    scenes = sorted(public)
    a = np.asarray([public[s] for s in scenes], dtype=np.float64)
    b = np.asarray([adapted[s] for s in scenes], dtype=np.float64)
    require(bool(np.isfinite(a).all() and np.isfinite(b).all()), "non-finite metric")
    delta = b-a
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(scenes), size=(SAMPLES, len(scenes)))
    draws = delta[indices].mean(axis=1)
    return {"public":float(a.mean()), "delimit3d":float(b.mean()),
            "delta":float(delta.mean()), "ci95_lower":float(np.quantile(draws, .025)),
            "ci95_upper":float(np.quantile(draws, .975)), "scene_count":len(scenes),
            "scene_differences":dict(zip(scenes, map(float, delta)))}

def decision(metrics):
    b1, b5, noc = (metrics[k] for k in ("iou@1", "iou@5", "noc@0.80"))
    conditions = {
        "iou_at_1_delta_ge_2pp": b1["delta"] >= .02,
        "iou_at_5_delta_ge_1_5pp": b5["delta"] >= .015,
        "paired_ci95_lowers_above_zero": b1["ci95_lower"] > 0 and b5["ci95_lower"] > 0,
        "noc_at_80_not_worse": noc["delta"] <= 0,
    }
    return {"conditions":conditions, "passed":all(conditions.values())}

def run():
    status = {"schema":"delimit3d_independent_stagea_gate/v1",
              "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "stagea_root":str(ROOT), "script_sha256":sha(__file__), "numpy_version":np.__version__,
              "stage_b_launched":False}
    missing = [str(p) for p in required_paths(ROOT) if not p.is_file()]
    if missing:
        status.update(status="pending", missing_final_inputs=missing)
        status["latest_persisted_updates"] = {}
        for arm in ARMS:
            p = ROOT/arm/"resume_train_from_u01000.jsonl"
            try:
                status["latest_persisted_updates"][arm] = json.loads(p.read_text().splitlines()[-1])["update"]
            except (OSError, IndexError, json.JSONDecodeError):
                status["latest_persisted_updates"][arm] = None
        write_json(AUDIT/"status_pending.json", status)
        print(json.dumps(status, indent=2))
        return 2
    hashes = {str(p):sha(p) for p in list(PINNED)+required_paths(ROOT)}
    for path, expected in PINNED.items():
        require(hashes[str(path)] == expected, "pinned provenance changed: " + str(path))
    manifest = read_json(BASE/"selection_manifest.json")
    expected_objects = {row["scene"]:row["objects"] for row in manifest["panels"]["MO-5"]}
    require(len(expected_objects) == 49, "unexpected frozen MO-5 scene count")
    caches = {arm:read_json(BASE/("cache_report_" + arm + ".json")) for arm in ARMS}
    geometries = [{row["scene"]:(row["geometry_sha256"],row["tokens"])
                   for row in caches[arm]["cache_rows"]} for arm in ARMS]
    require(geometries[0] == geometries[1] and len(geometries[0]) == 100,
            "public/adapted cache geometry identity mismatch")
    arms = {}
    for arm in ARMS:
        ckpt = ROOT/arm/"decoder_u05000.pt"
        metadata = scalar_checkpoint_metadata(ckpt)
        train = read_json(ROOT/arm/"train_report.json")
        evaluation = read_json(ROOT/arm/"evaluation_report.json")
        panel = validate_arm(arm,train,metadata,hashes[str(ckpt)],evaluation,caches[arm],manifest,ROOT)
        scenes = load_rows(ROOT/arm/"evaluation/MO-5/scenes.jsonl")
        episodes = load_rows(ROOT/arm/"evaluation/MO-5/episodes.jsonl")
        require(set(scenes) == set(episodes) == set(expected_objects), arm + ": incomplete/changed MO-5 scenes")
        require(panel.get("scenes") == len(scenes) and panel.get("episodes") == len(episodes),
                arm + ": evaluation count mismatch")
        for scene, episode in episodes.items():
            require(episode["panel"] == "MO-5" and episode["object_count"] == 5 and
                    episode["panel_objects"] == expected_objects[scene],
                    arm + ": selected object identity mismatch " + scene)
        arms[arm] = {"metadata":metadata, "train_report":train, "scene_rows":scenes}
    metrics = {}
    for metric in ("iou@1","iou@5","noc@0.80"):
        values = {}
        for arm in ARMS:
            rows = arms[arm]["scene_rows"]
            if metric.startswith("iou"):
                key = metric.split("@")[1]
                values[arm] = {s:row["thresholds"][key]["mean_iou"] for s,row in rows.items()}
            else:
                values[arm] = {s:row["noc"]["0.80"] for s,row in rows.items()}
        metrics[metric] = paired_bootstrap(values["public"],values["delimit3d"])
    gate = decision(metrics)
    status.update(status="passed" if gate["passed"] else "failed_or_mixed", gate=gate, metrics=metrics,
                  paired_scene_identity=sorted(expected_objects), paired_object_identity=expected_objects,
                  input_sha256=hashes, bootstrap={"samples":SAMPLES,"seed":SEED,
                  "method":"NumPy default_rng paired scene percentile bootstrap; separate fresh identical seed per metric"},
                  encoder_invariants=EXPECTED_ENCODERS,
                  decoder_initialization_sha256=manifest["decoder_initialization"]["tensor_state_sha256"],
                  episode_schedule_sha256=manifest["train_episodes_sha256"],
                  checkpoint_metadata={arm:arms[arm]["metadata"] for arm in ARMS},
                  resume={"resumed_from_update":1000,"rng_state_restored":False,
                          "continuation":"Decoder and AdamW restored; random stream is not bit exact"},
                  performance_type="matched frozen-backbone ScanNet++ downstream interactive transfer",
                  noc_definition="Uses the frozen Stage-A scenes.jsonl NoC: average object threshold-reaching click counts normalized by requested objects. Stage-B official scene-mean NoC has a different definition; do not equate them.")
    aggregate_path = ROOT/"aggregate.json"
    if aggregate_path.is_file():
        aggregate = read_json(aggregate_path)
        reference = aggregate["panels"]["MO-5"]["bootstrap"]
        differences = {}
        for metric, result in metrics.items():
            diff = {k:result[ours]-reference[metric][k] for k,ours in (
                ("observed_difference","delta"),("ci95_lower","ci95_lower"),("ci95_upper","ci95_upper"))}
            require(max(abs(v) for v in diff.values()) < 1e-12, "aggregate disagrees with independent recomputation")
            differences[metric] = diff
        status["aggregate_crosscheck"] = differences
        status["input_sha256"][str(aggregate_path)] = sha(aggregate_path)
    # Reject input changes during the audit. This gate never uses mutable log hashes.
    require(all(sha(Path(p)) == h for p,h in hashes.items()), "inputs changed during gate audit")
    write_json(AUDIT/"final_gate.json",status)
    print(json.dumps({k:v for k,v in status.items() if k not in ("paired_object_identity","input_sha256","checkpoint_metadata")},indent=2))
    return 0 if gate["passed"] else 3

def self_test():
    tests = []
    def checked(name, fn):
        fn()
        tests.append(name)
    def rejects(fn):
        try:
            fn()
        except (ValueError,KeyError):
            return
        raise AssertionError("invalid input accepted")
    checked("missing_final_inputs_cannot_be_interim", lambda: require(
        all("u05000" in str(p) for p in required_paths(ROOT) if p.suffix == ".pt") and
        not any("interim" in str(p) for p in required_paths(ROOT)), "wrong final paths"))
    checked("paired_scenes_must_match",lambda: rejects(lambda:paired_bootstrap({"a":.1},{"b":.2})))
    checked("nonfinite_metrics_rejected",lambda: rejects(lambda:paired_bootstrap({"a":math.nan},{"a":.2})))
    b = paired_bootstrap({"a":0.0,"b":0.0},{"a":.03,"b":.05})
    checked("known_paired_bootstrap",lambda: require(abs(b["delta"]-.04)<1e-14 and b["ci95_lower"]==.03 and b["ci95_upper"]==.05, "bad bootstrap"))
    fixture = {"iou@1":{"delta":.02,"ci95_lower":.001},
               "iou@5":{"delta":.015,"ci95_lower":.001}, "noc@0.80":{"delta":0.0}}
    checked("inclusive_thresholds_and_tied_noc_pass",lambda: require(decision(fixture)["passed"],"threshold gate failed"))
    for label,metric,key,value in (
        ("iou1_below_threshold","iou@1","delta",.019999),
        ("iou5_below_threshold","iou@5","delta",.014999),
        ("zero_lower_bound_fails","iou@1","ci95_lower",0.0),
        ("negative_lower_bound_fails","iou@5","ci95_lower",-.001),
        ("worsened_noc_fails","noc@0.80","delta",.000001)):
        bad = copy.deepcopy(fixture);bad[metric][key]=value
        checked(label,lambda bad=bad:require(not decision(bad)["passed"],"bad gate accepted"))
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/"scenes.jsonl"
        p.write_text('{"scene":"a"}\n{"scene":"a"}\n')
        checked("duplicate_scene_rejected",lambda:rejects(lambda:load_rows(p)))
        checked("incomplete_final_set_pending",lambda:require(bool([p for p in required_paths(Path(directory)) if not p.exists()]),"no missing final files"))
    arm, root, h = "public", Path("/fixture/final_retry"), "checkpoint-hash"
    enc = EXPECTED_ENCODERS[arm]
    m = {"decoder_initialization":{"tensor_state_sha256":"same-init"}, "train_episodes_sha256":"same-schedule"}
    cache = {"encoder_state_unchanged":True,"encoder_tensor_sha256_before":enc,"encoder_tensor_sha256_after":enc}
    report = {"arm":arm,"updates":5000,"experiment_id":EXPERIMENT,"repo_commit":COMMIT,
              "checkpoint":str(root/arm/"decoder_u05000.pt"),"checkpoint_sha256":h,
              "encoder_state_unchanged":True,"encoder_tensor_sha256_before":enc,
              "encoder_tensor_sha256_after":enc,"decoder_initialization_sha256":"same-init",
              "episode_schedule_sha256":"same-schedule","resumed_from_update":1000,
              "rng_state_restored":False,"decoder_tensor_sha256":"decoder-hash"}
    meta = {"experiment_id":EXPERIMENT,"arm":arm,"update":5000,"repo_commit":COMMIT,
            "encoder_frozen":True,"encoder_tensor_sha256":enc,
            "decoder_initialization_sha256":"same-init","episode_schedule_sha256":"same-schedule",
            "decoder_tensor_sha256":"decoder-hash"}
    ev = {"arm":arm,"repo_commit":COMMIT,"reports":{"MO-5":{
        "arm":arm,"panel":"MO-5","experiment_id":EXPERIMENT,
        "decoder_checkpoint":str(root/arm/"decoder_u05000.pt"),"decoder_checkpoint_sha256":h,
        "encoder_state_unchanged":True,"encoder_tensor_sha256":enc,
        "encoder_checkpoint_sha256":PINNED[ENCODER_FILES[arm]],
        "scene_rows":str(root/arm/"evaluation/MO-5/scenes.jsonl"),
        "episode_rows":str(root/arm/"evaluation/MO-5/episodes.jsonl")}}}
    checked("valid_final_report_and_evaluation",lambda:validate_arm(arm,report,meta,h,ev,cache,m,root))
    for key,value in (("updates",1000),("checkpoint_sha256","wrong"),
                      ("decoder_initialization_sha256","wrong"),("episode_schedule_sha256","wrong"),
                      ("encoder_tensor_sha256_after","wrong"),("rng_state_restored",True)):
        bad=dict(report,**{key:value})
        checked("invalid_report_"+key,lambda bad=bad:rejects(lambda:validate_arm(arm,bad,meta,h,ev,cache,m,root)))
    bad_meta=dict(meta,update=1000)
    checked("interim_checkpoint_metadata_rejected",lambda:rejects(lambda:validate_arm(arm,report,bad_meta,h,ev,cache,m,root)))
    bad_eval=copy.deepcopy(ev)
    bad_eval["reports"]["MO-5"]["decoder_checkpoint"]="/fixture/interim/decoder_u05000.pt"
    checked("interim_evaluation_alias_rejected",lambda:rejects(lambda:validate_arm(arm,report,meta,h,bad_eval,cache,m,root)))
    print(json.dumps({"status":"passed","tests":tests,"count":len(tests),"numpy_version":np.__version__},indent=2))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        try:
            raise SystemExit(run())
        except Exception as exc:
            write_json(AUDIT/"status_invalid.json",{"status":"integrity_failure","error":str(exc),
                       "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "script_sha256":sha(__file__)})
            print("INTEGRITY FAILURE:",exc)
            raise SystemExit(4)

