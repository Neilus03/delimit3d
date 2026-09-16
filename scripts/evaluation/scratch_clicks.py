"""Parallel exact KD-tree click centers; identical ordering and tie breaking."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from delimit3d.evaluation import agile3d_protocol as reference

_original = reference._cluster_click_candidates
_pool = ThreadPoolExecutor(max_workers=8)

def candidates(prediction, target, xyz, *, method='kdtree'):
    if method != 'kdtree':
        return _original(prediction, target, xyz, method=method)
    prediction=np.asarray(prediction,dtype=np.int64)
    target=np.asarray(target,dtype=np.int64)
    xyz=np.asarray(xyz,dtype=np.float64)
    if prediction.shape != target.shape or xyz.shape != (len(target),3):
        raise ValueError('prediction, target, and xyz are not aligned')
    error=prediction != target
    if not error.any(): return []
    cluster_ids=target*96+prediction*11
    def region_candidate(cluster_id):
        region=error & (cluster_ids==cluster_id)
        indices=np.flatnonzero(region)
        complement=np.flatnonzero(~region)
        if len(complement):
            distances,_=cKDTree(xyz[complement]).query(xyz[indices], k=1, workers=1)
        else:
            distances=np.full(len(indices),np.inf)
        distance=float(np.max(distances))
        farthest=np.flatnonzero(np.isclose(distances,distance,rtol=0.,atol=1e-12))
        center=int(indices[int(farthest[0])])
        return dict(cluster_id=int(cluster_id),target_label=int(target[center]),size=distance,
                    center_index=center,region_points=int(len(indices)))
    result=list(_pool.map(region_candidate,sorted(int(k) for k in np.unique(cluster_ids[error]))))
    result.sort(key=lambda r:(-r['size'],r['cluster_id'],r['center_index']))
    return result

def install():
    reference._cluster_click_candidates=candidates
