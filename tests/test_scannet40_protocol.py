import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
from delimit3d.evaluation import scannet40_protocol as p


def trace(count=3, crossing=7):
    return {"scene":"scene0568_00","object_count":count,"object_ids":[25,24,3][:count],
            "states":[{"total_clicks":0,"mean_iou":0.}]+[{"total_clicks":x,"mean_iou":.8 if x>=crossing else .2} for x in range(count,20*count+1)]}


def test_official_mo_fractional_scene_mean_noc():
    t=trace();m=p.metrics_from_trace(t)
    assert m["NoC@80"]==7/3
    assert m["NoC@90"]==20
    assert m["IoU@1"]==.2
    assert m["IoU@3"]==.8
    rows=list(p.official_lines(0,t,"MO"))
    assert rows[0]=="0 0568_00 3 0.0 0.0\n"
    assert rows[1]=="0 0568_00 3 1.0 0.2\n"
    assert rows[-1].split()[3]=="20.0"


def test_so_original_identity_and_integer_clicks():
    rows=list(p.official_lines(2,trace(1),"SO"))
    assert rows[1].split()[:4]==["2","0568_00","25","1"]


def test_missing_budget_fails_instead_of_interpolation():
    t=trace();t["states"]=[s for s in t["states"] if s["total_clicks"]!=15]
    with pytest.raises(ValueError):p.metrics_from_trace(t)


def test_paired_bootstrap_rejects_unmatched_and_has_expected_sign():
    keys=[f"IoU@{x}" for x in p.CLICKS]+[f"NoC@{x}" for x in p.THRESHOLDS]
    a={"s1":dict.fromkeys(keys,.2),"s2":dict.fromkeys(keys,.3)}
    b={k:{m:v+.1 for m,v in row.items()} for k,row in a.items()}
    r=p.paired_bootstrap(a,b,samples=100,seed=3)
    assert r["IoU@1"]["ci95_lower"]==pytest.approx(.1)
    with pytest.raises(ValueError):p.paired_bootstrap(a,{"s1":b["s1"]})


def test_initial_click_deepest_region_and_paired_order():
    xyz=np.array([[0,0,0],[.1,0,0],[.2,0,0],[1,0,0],[1.1,0,0],[1.2,0,0],[2,0,0]],float)
    target=np.array([1,1,0,2,2,2,0])
    a=p.initial_clicks(target,xyz,44);b=p.initial_clicks(target,xyz,44)
    assert a==b
    assert a[0]["1"]==[0]  # deepest from complement; centroid would select the other tie convention
    assert sorted(v[0] for k,v in a[1].items() if k!="0")==[0,1]


def test_runner_contract_and_lru(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/"scripts/evaluation"
    sys.path.insert(0,str(scripts))
    import run_scannet40_matched as runner
    calls=[]
    monkeypatch.setattr(runner.core,"load_cache",lambda root,arm,scene:calls.append(scene) or {"scene":scene})
    lru=runner.CacheLRU(tmp_path,"public",2)
    for scene in ("a","b","a","c","b"):lru.get(scene)
    assert calls==["a","b","c","b"]
    assert len(lru.values)==2


def test_official_serialization_preserves_threshold_boundary():
    t=trace(1);t["states"][1]["mean_iou"]=.5-1e-13
    written=float(list(p.official_lines(0,t,"SO"))[1].split()[4])
    assert written<.5
    assert written==t["states"][1]["mean_iou"]


def test_official_ply_ids_preserved_when_processed_ids_are_offset(tmp_path):
    from plyfile import PlyData,PlyElement
    scene="scene_fixture";official=tmp_path/"official";pack=tmp_path/"processed/train"/scene
    (official/"scans").mkdir(parents=True);pack.mkdir(parents=True)
    xyz=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype=np.float32)
    rgb=np.full((4,3),100,dtype=np.uint8);normal=np.tile([0.,0.,1.],(4,1)).astype(np.float32)
    vertices=np.zeros(4,dtype=[(k,"f4") for k in ("x","y","z")]+[(k,"u1") for k in ("R","G","B")]+[("label","f8")])
    for j,k in enumerate(("x","y","z")):vertices[k]=xyz[:,j]
    for j,k in enumerate(("R","G","B")):vertices[k]=rgb[:,j]
    vertices["label"]=[1,1,23,-1]
    PlyData([PlyElement.describe(vertices,"vertex")]).write(official/"scans"/f"{scene}.ply")
    for name,array in (("coord",xyz),("color",rgb),("normal",normal),("instance",np.array([31,31,53,-1]))):np.save(pack/f"{name}.npy",array)
    record=p.inspect_scene(scene,"train",official,tmp_path/"processed")
    assert record["processed_instance_plus_one_exact_diagnostic"] is False
    assert [o["instance"] for o in record["objects"]]==[1,23]
    assert [o["instance"] for o in p.objects_from_official_labels(vertices["label"],scene)]==[1,23]
    _,_,_,_,masks=p.load_scene(record)
    np.testing.assert_array_equal(masks[23],[2])

def test_native_cuda_preflight_preserves_observed_boundary_rounding():
    # Four exact coordinates from real RTX4090 job 13911781, scene0568_00.
    points = np.array([
        [0, 0, 1.3399999141693115], [0, 0, 1.33],
        [-2.6000001430511475, 0, 0], [-2.59, 0, 0],
        [0, 3.1999998092651367, 0], [0, 3.19, 0],
        [3.43999981880188, 0, 0], [3.43, 0, 0],
    ], dtype=np.float32)
    from delimit3d.evaluation.scannet40_protocol import native_cuda_representative_indices
    # Native CUDA merges each pair and keeps the first point in each voxel.
    assert native_cuda_representative_indices(points).tolist() == [2, 0, 4, 6]
    divided = np.floor(points / np.float32(.02)).astype(np.int64)
    assert len(np.unique(divided, axis=0)) == 8


def test_native_cuda_preflight_rejects_nonfinite_geometry():
    from delimit3d.evaluation.scannet40_protocol import native_cuda_representative_indices
    import pytest
    with pytest.raises(ValueError, match="finite Nx3"):
        native_cuda_representative_indices([[float("nan"), 0, 0]])
    with pytest.raises(ValueError, match="positive finite"):
        native_cuda_representative_indices([[0, 0, 0]], voxel_size=0)
