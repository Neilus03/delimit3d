#!/usr/bin/env python3
"""Explicit, gated tmux launch for two matched frozen ScanNet40 arms.

Run from the frozen source extraction after the lead's final launch review.
This entrypoint never starts from a preparation call and refuses existing jobs.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import run_scannet40_matched as driver

SESSION="delimit3d-scannet40-matched-v1"


def gate(config):
    if config.get("smoke_only"):raise ValueError("production launcher rejects smoke config")
    driver.verify_freeze(config)
    root=driver.root_of(config)
    result=subprocess.run(["ssh","nedela@euler.ethz.ch","env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","/cluster/work/igp_psr/nedela/litept-env/bin/python","/cluster/home/nedela/nedela/projects/delimit3d-scannet40-matched/scripts/evaluation/verify_stagea_gate.py"],capture_output=True,text=True)
    if result.returncode!=0:raise RuntimeError(f"final Stage A gate not passed: {result.stdout[-2000:]} {result.stderr[-500:]}")
    final=json.loads(driver.resolve(config["stagea_gate"]["report"]).read_text())
    if final.get("status")!="passed":raise ValueError("fresh final Stage A gate is not passed")
    stagea=driver.resolve("/cluster/work/igp_psr/nedela/delimit3d_scannetpp_agile3d_mo_v1_retry_20260912")
    if not (root/"lead_audit/supervisor_repair/pipeline_complete.json").exists():raise ValueError("Stage A aggregate/visual pipeline is unfinished")
    smoke_root=driver.resolve(config["paths"]["smoke_root"])
    if not (smoke_root/"smoke_pipeline_complete.json").exists():raise ValueError("complete real RTX4090 smoke required")
    smoke_provenance=json.loads((smoke_root/"freeze/provenance.json").read_text())
    provenance=json.loads((root/"freeze/provenance.json").read_text())
    if smoke_provenance["repo_commit"]!=config["repo_commit"] or smoke_provenance["decoder_initialization"]["tensor_state_sha256"]!=provenance["decoder_initialization"]["tensor_state_sha256"] or smoke_provenance["checkpoints"]!=provenance["checkpoints"]:raise ValueError("smoke belongs to a different source/initialization/encoder pair")
    smoke=json.loads((smoke_root/"smoke_report.json").read_text())
    if "RTX 4090" not in smoke["gpu_name"] or not smoke["geometry_byte_identity"]:raise ValueError("invalid smoke hardware/geometry")
    if not (root/"cpu_test_report.json").exists() or not json.loads((root/"cpu_test_report.json").read_text())["passed"]:raise ValueError("CPU tests must pass")
    cpu=json.loads((root/"cpu_test_report.json").read_text())
    if cpu.get("source_commit")!=config["repo_commit"] or cpu.get("source_entries")!=provenance["source_entries"]:raise ValueError("CPU test evidence does not match frozen source")
    completion=json.loads((smoke_root/"smoke_pipeline_complete.json").read_text())
    if completion.get("source_commit")!=config["repo_commit"] or completion.get("provenance_sha256")!=driver.file_sha256(smoke_root/"freeze/provenance.json") or completion.get("smoke_report_sha256")!=driver.file_sha256(smoke_root/"smoke_report.json"):raise ValueError("smoke completion provenance mismatch")
    manifest,_=driver.load_prepared(config)
    if len(manifest["train_scenes"])!=1200 or len(manifest["validation_scenes"])!=312:raise ValueError("official full scene coverage required")
    return final


def worker(config_path,phase,arm,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",PYTHONUNBUFFERED="1")
    runner=Path(__file__).with_name("run_scannet40_matched.py")
    panels=("MO","SO") if phase=="evaluate" else (None,)
    for panel in panels:
        command=[sys.executable,str(runner),"--config",str(config_path),"--mode",phase,"--arm",arm]
        if panel:command +=["--panel",panel]
        subprocess.run(command,env=env,check=True)


def supervise(config_path,public_gpu,adapted_gpu):
    config=driver.core.load_config(config_path);driver.install_adapter();gate(config)
    root=driver.root_of(config)
    with driver.worker_files(config,"supervisor"):
        for phase in ("cache-features","train","evaluate"):
            processes=[]
            for arm,gpu in (("public",public_gpu),("delimit3d",adapted_gpu)):
                log=(root/"workers"/f"{phase}_{arm}.log").open("a")
                command=[sys.executable,__file__,"--config",str(config_path),"--worker",phase,"--arm",arm,"--gpu",str(gpu)]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);processes.append((process,log,arm))
                driver.dump(root/"workers"/f"{phase}_{arm}.launcher_pid.json",{"pid":process.pid,"gpu":gpu,"arm":arm,"phase":phase,"tmux_session":SESSION})
            codes=[]
            for process,log,arm in processes:
                code=process.wait();log.close();codes.append((arm,code))
            if any(code for _,code in codes):raise RuntimeError(f"{phase} failed: {codes}; preserving all state")
            if phase=="cache-features":driver.verify_caches(config)
            if phase=="train":
                reports=[json.loads((root/a/"train_report.json").read_text()) for a in ("public","delimit3d")]
                if any(r["updates"]!=5000 for r in reports):raise ValueError("both final reports required before evaluation")
        driver.aggregate(config);driver.dump(root/"pipeline_complete.json",{"complete":True,"time":time.time()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--public-gpu",type=int,default=0);parser.add_argument("--adapted-gpu",type=int,default=1)
    parser.add_argument("--supervise",action="store_true");parser.add_argument("--worker",choices=("cache-features","train","evaluate"));parser.add_argument("--arm");parser.add_argument("--gpu",type=int)
    args=parser.parse_args();config=driver.core.load_config(args.config);driver.install_adapter()
    if args.worker:worker(args.config,args.worker,args.arm,args.gpu);return
    if args.supervise:supervise(args.config,args.public_gpu,args.adapted_gpu);return
    gate(config)
    if args.public_gpu==args.adapted_gpu:raise ValueError("distinct GPU per arm required")
    inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total","--format=csv,noheader"],text=True);print(inventory,flush=True)
    names={int(line.split(",")[0]):line.split(",")[1] for line in inventory.splitlines()}
    for gpu in (args.public_gpu,args.adapted_gpu):
        if not any(model in names[gpu] for model in ("RTX 3090","RTX 4090","A100")):raise ValueError("unapproved GPU model")
        pids=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
        if pids:raise ValueError(f"GPU {gpu} is occupied: {pids}")
    if subprocess.run(["tmux","has-session","-t",SESSION],capture_output=True).returncode==0:raise ValueError("existing Stage B tmux session; no duplicate launch")
    root=driver.root_of(config);command=[sys.executable,str(Path(__file__).resolve()),"--config",str(args.config.resolve()),"--supervise","--public-gpu",str(args.public_gpu),"--adapted-gpu",str(args.adapted_gpu)]
    shell=shlex.join(command)+" > "+shlex.quote(str(root/"supervisor.log"))+" 2>&1"
    subprocess.run(["tmux","new-session","-d","-s",SESSION,"-c",str(driver.REPO),shell],check=True)
    print(json.dumps({"tmux":SESSION,"root":str(root),"gpu_assignment":{"public":args.public_gpu,"delimit3d":args.adapted_gpu}}))

if __name__=="__main__":main()
