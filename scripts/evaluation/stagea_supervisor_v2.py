#!/usr/bin/env python3
"""Repair only the Stage-A monitoring/evaluation pipeline; never launch training."""
import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/tmp/euler_cluster_nedela_rw/work/igp_psr/nedela/delimit3d_scannetpp_agile3d_mo_v1_retry_20260912")
AUDIT = Path("/tmp/euler_cluster_nedela_rw/work/igp_psr/nedela/delimit3d_scannet40_agile3d_matched_v1/lead_audit/supervisor_repair")
REPO = Path("/tmp/delimit3d_retry_source_3df5fdb")
PLOT_REPO = Path("/tmp/delimit3d_retry_source_2feac4e")
CFG = Path("/scratch/nedela/delimit3d_scannetpp_agile3d_mo_retry_20260912.yaml")
PYTHON = "/scratch/nedela/delimit3d/env/bin/python"
ARMS = ("public", "delimit3d")

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def log(message):
    print(now(), message, flush=True)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
def validate_report(report, arm, checkpoint):
    if report.get("arm") != arm or report.get("updates") != 5000:
        raise RuntimeError(f"{arm}: report identity/update mismatch")
    if not report.get("encoder_state_unchanged"):
        raise RuntimeError(f"{arm}: encoder invariant failed")
    if report.get("encoder_tensor_sha256_before") != report.get("encoder_tensor_sha256_after"):
        raise RuntimeError(f"{arm}: encoder hash drift")
    if not checkpoint.is_file() or sha(checkpoint) != report.get("checkpoint_sha256"):
        raise RuntimeError(f"{arm}: final checkpoint missing/hash mismatch")
    if Path(report.get("checkpoint", "")).name != checkpoint.name:
        raise RuntimeError(f"{arm}: final checkpoint name mismatch")
    return True
def valid_final_report(arm):
    path = ROOT / arm / "train_report.json"
    if not path.is_file():
        return False
    # Atomic replacement is not guaranteed in the frozen trainer; retry only JSON reads.
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return validate_report(report, arm, ROOT / arm / "decoder_u05000.pt")
def workers():
    table = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    found = {}
    for line in table.splitlines():
        pieces = line.split()
        if not pieces:
            continue
        args = pieces[1:]
        if "/tmp/delimit3d_resume_train.py" not in args or "--arm" not in args:
            continue
        i = args.index("--arm")
        if i+1 < len(args) and args[i+1] in ARMS:
            found[args[i+1]] = int(pieces[0])
    return found
def waiting_arms(done, live):
    missing = [arm for arm in ARMS if not done[arm] and arm not in live]
    if missing:
        raise RuntimeError(f"unfinished worker missing: {missing}; preserving all artifacts")
    return [arm for arm in ARMS if not done[arm]]
def assert_gpu(index):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], text=True)
    log(raw.strip())
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    names = {int(row[0]): row[1] for row in rows}
    if "RTX 4090" not in names.get(index, ""):
        raise RuntimeError(f"GPU {index} is not an RTX 4090: {names}")
def start_worker(name, command, gpu=None, logfile=None):
    if gpu is not None:
        assert_gpu(gpu)
    env = dict(os.environ, DELIMIT3D_EULER_MOUNT="/tmp/euler_cluster_nedela_rw",
               PYTHONUNBUFFERED="1", PYTHONPATH=str(REPO / "src"),
               CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu))
    output = (logfile or AUDIT / (name + ".log")).open("a")
    p = subprocess.Popen(command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
    (AUDIT / (name + ".pid")).write_text(str(p.pid) + "\n")
    write_json(AUDIT / (name + ".launch.json"), {"pid":p.pid, "gpu":gpu, "command":command, "time":now()})
    log(f"started {name} pid={p.pid} gpu={gpu}")
    return name, p, output
def wait_workers(active):
    while active:
        remaining = []
        for name, process, output in active:
            code = process.poll()
            with (AUDIT / (name + ".heartbeat.log")).open("a") as f:
                f.write(f"{now()} pid={process.pid} status={code if code is not None else 'running'}\n")
            if code is None:
                remaining.append((name, process, output))
            else:
                output.close()
                write_json(AUDIT / (name + ".exit.json"), {"pid":process.pid, "exit_code":code, "time":now()})
                log(f"finished {name} exit={code}")
                if code:
                    raise RuntimeError(f"{name} failed with {code}; any other worker remains untouched")
        active = remaining
        if active:
            time.sleep(30)
def self_test():
    assert waiting_arms({"public":True, "delimit3d":False}, {"delimit3d":7}) == ["delimit3d"]
    assert waiting_arms({"public":False, "delimit3d":True}, {"public":7}) == ["public"]
    assert waiting_arms({"public":True, "delimit3d":True}, {}) == []
    try:
        waiting_arms({"public":False, "delimit3d":True}, {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("dead unfinished worker was accepted")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "decoder_u05000.pt"
        ckpt.write_bytes(b"checkpoint")
        r = {"arm":"public", "updates":5000, "encoder_state_unchanged":True,
             "encoder_tensor_sha256_before":"abc", "encoder_tensor_sha256_after":"abc",
             "checkpoint_sha256":sha(ckpt), "checkpoint":str(ckpt)}
        assert validate_report(r, "public", ckpt)
        for key, value in (("updates",1000), ("encoder_tensor_sha256_after","drift"), ("checkpoint_sha256","bad")):
            wrong = dict(r, **{key:value})
            try:
                validate_report(wrong, "public", ckpt)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"invalid {key} accepted")
    print("PASS: 8 supervisor readiness and integrity checks")
def main():
    import fcntl
    AUDIT.mkdir(parents=True, exist_ok=True)
    lock = (AUDIT / "supervisor.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    (AUDIT / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    log(f"supervisor repaired pid={os.getpid()} script_sha256={sha(__file__)}")
    if (AUDIT / "pipeline_complete.json").exists():
        log("pipeline already complete")
        return
    while True:
        done = {arm:valid_final_report(arm) for arm in ARMS}
        live = workers()
        pending = waiting_arms(done, live)
        log(f"training done={done} live={live} pending={pending}")
        with (AUDIT / "supervisor.heartbeat.log").open("a") as f:
            f.write(f"{now()} pid={os.getpid()} pending={pending}\n")
        if not pending:
            break
        time.sleep(60)
    # Avoid duplicate evaluation after an uncertain interruption.
    for name in ("eval_public", "eval_delimit3d", "aggregate", "visuals"):
        if (AUDIT / (name + ".launch.json")).exists():
            raise RuntimeError(f"prior {name} launch recorded; manual reconciliation required")
    runner = [PYTHON, str(REPO / "scripts/evaluation/run_agile3d_multio.py"), "--config", str(CFG)]
    active = []
    for index, arm in enumerate(ARMS):
        active.append(start_worker("eval_" + arm, runner + ["--mode","evaluate","--arm",arm,"--panel","all"], gpu=index, logfile=ROOT / ("eval_" + arm + ".log")))
    wait_workers(active)
    wait_workers([start_worker("aggregate", runner + ["--mode","aggregate"], logfile=ROOT / "aggregate.log")])
    wait_workers([start_worker("visuals", [PYTHON, str(PLOT_REPO / "scripts/evaluation/plot_agile3d_interactive.py"), "--run-root",str(ROOT),"--output",str(ROOT / "visuals"),"--panel","MO-5","--tag","u05000_mo5"], logfile=ROOT / "visuals.log")])
    write_json(AUDIT / "pipeline_complete.json", {"time":now(), "stagea_root":str(ROOT), "script_sha256":sha(__file__)})
    log("Stage-A evaluation, aggregation and MO-5 visuals complete; no Stage-B launch performed")
if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        try:
            main()
        except Exception as exc:
            log(f"ERROR: {exc}")
            raise
