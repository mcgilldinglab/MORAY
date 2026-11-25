#!/usr/bin/env python3
"""Reads in, species calls out. The deployment driver.

    reads.tsv (read_id, sequence)
      -> stage 1  L1 domain routing        (CPU)   human / bacteria / fungi / virus / no_call / ambiguous
      -> stage 2  split reads by domain    (CPU)
      -> stage 3  K25 evidence per domain  (CPU)   species + genus + phylum masks
      -> stage 4  DL strict chain          (GPU)   phylum -> genus -> species
      -> stage 5  merge                    (CPU)   one row per input read

Every stage is a Slurm job; stages are chained with --dependency=afterok. Nothing heavy runs on the
login node. Stage 3 and 4 fan out over the domains present in the sample.

Design notes that are NOT arbitrary:

* Reads that L1 calls `human`, `no_call` or `ambiguous` never reach a DL model -- there is no human
  model, and a read with zero domain-unique k-mer hits has no evidence to classify. They are carried
  through to the final output with an explicit status, not silently dropped. On a real metagenome this
  is a large fraction of the sample and it is a RESULT, not a failure.

* `parent_evidence` per domain (see config): bacteria and fungi roll their genus/phylum evidence up
  from the species masks; virus queries a per-rank index because its database is direct-rank and a
  rollup provably cannot reproduce what its model was trained on. Do not "simplify" this away.

* The bacteria index is 380 GB and needs ~500 GB of RAM. It is loaded once for the whole sample.

usage:
    run_pipeline.py --reads reads.tsv --out-dir results/ [--domains bacteria,fungi,virus] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <pkg>/pipeline
PKG = HERE.parent                                # <pkg> -- the copyable code package

site_path = PKG / "config" / "site.json"
SITE = json.loads(site_path.read_text()) if site_path.exists() else {}
asset_path = os.environ.get("MBC_ASSETS_ROOT", SITE.get("assets_root", "{PKG}/assets"))
ASSETS = Path(asset_path.replace("{PKG}", str(PKG))).expanduser().resolve().as_posix()
LABELS = Path(os.environ.get("MBC_LABEL_SPACE", str(Path(ASSETS).parent / "config/label_space")))
LABELS = LABELS.expanduser().resolve().as_posix()
quote = shlex.quote

# pipeline.json is a template; {PKG} and {ASSETS} are substituted here so every downstream path is absolute.
CONFIG = json.loads(
    (PKG / "config" / "pipeline.json").read_text()
    .replace("{PKG}", PKG.as_posix()).replace("{ASSETS}", ASSETS)
)

DL_DOMAINS = CONFIG["l1_routing"]["domains_sent_to_dl"]

# Cluster settings live in site.json and are overridable by environment variable, so porting this
# release to another cluster means editing one file, not hunting through generated Slurm scripts.
CL = SITE.get("cluster", {})
def cluster(key: str) -> str:
    return os.environ.get("MBC_" + key.upper(), CL.get(key, ""))

ACCOUNT = cluster("account")
VENV_CPU = cluster("venv_cpu")
VENV_GPU = cluster("venv_gpu")
PY_MODULE = cluster("python_module")
GPU_GRES = cluster("gpu_gres")

# --- deployment mode: slurm (emit sbatch + afterok chains) OR standalone (run stages HERE, sequentially) ---
DEPLOY = SITE.get("deployment", {})
LOCAL = (os.environ.get("MBC_LOCAL", "").lower() in ("1", "true", "yes")
         or DEPLOY.get("type", "standalone") == "standalone")
LOCAL_CPUS = str(DEPLOY.get("local_cpus") or "").strip()   # "" -> $(nproc) at run time
EVAL_BATCH = str(os.environ.get("MBC_EVAL_BATCH_SIZE", DEPLOY.get("eval_batch_size", 0)) or 0)
RES = CONFIG.get("resources", {})


def res(stage: str, key: str, default):
    """Per-stage Slurm resource from config/pipeline.json 'resources' (falls back to `default`)."""
    return RES.get(stage, {}).get(key, default)


def activate(gpu: bool) -> str:
    """Env-setup line(s) for a stage script: module load (Slurm/Alliance only) + venv activate.
    On standalone there is usually no Lmod, so `module load` is skipped when PY_MODULE is empty or LOCAL."""
    venv = VENV_GPU if gpu else VENV_CPU
    mod = f"module load {PY_MODULE}\n" if (PY_MODULE and not LOCAL) else ""
    return mod + (f"source {quote(venv)}" if venv else "")


def sbatch(script: str, deps: list[str] | None, dry: bool) -> str:
    # Standalone: no scheduler -- run the stage script HERE, synchronously. main() dispatches stages in
    # a valid topological order (route->split->[evidence->predict per domain]->merge), so sequential
    # execution satisfies every dependency without afterok. A non-zero exit aborts the whole run.
    if LOCAL:
        name = Path(script).name
        if dry:
            print(f"  DRY(local)  bash {script}")
            return "0"
        logdir = Path(script).resolve().parent.parent / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        log = logdir / (Path(script).stem + ".log")
        print(f"  [local] running {name} ...", flush=True)
        with open(log, "w") as lf:
            r = subprocess.run(["bash", str(script)], stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"[local] stage {name} FAILED (exit {r.returncode}) -- see {log}")
        print(f"  [local] done {name}  (log: {log})")
        return "0"
    cmd = ["sbatch", "--parsable"]
    if deps:
        cmd.append("--dependency=afterok:" + ":".join(deps))
    cmd.append(script)
    if dry:
        print("  DRY  " + " ".join(cmd))
        return "0"
    job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    print(f"  submitted {job}: {Path(script).name}" + (f" (after {','.join(deps)})" if deps else ""))
    return job


def write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(0o750)
    return path


def header(name: str, out: Path, *, cpus: int, mem: str, time: str, gpu: bool = False) -> str:
    # Standalone: no #SBATCH directives (they'd be inert), but the stage bodies reference
    # $SLURM_CPUS_PER_TASK, so define it locally (configured local_cpus, else all cores).
    if LOCAL:
        cpus_expr = LOCAL_CPUS or "$(nproc)"
        return (
            "#!/bin/bash\n"
            "set -Eeuo pipefail\n"
            f"export MBC_LABEL_SPACE={quote(LABELS)}\n"
            f"export SLURM_CPUS_PER_TASK=${{SLURM_CPUS_PER_TASK:-{cpus_expr}}}\n"
            f"# [standalone] {name} (on Slurm this would request cpus={cpus} mem={mem} time={time} gpu={gpu})\n"
        )
    gres = f"#SBATCH --gres={GPU_GRES}\n" if gpu else ""
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={name}\n"
        f"#SBATCH --account={ACCOUNT}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        f"{gres}"
        f"#SBATCH --output={out}/logs/{name}_%j.out\n"
        f"#SBATCH --error={out}/logs/{name}_%j.err\n"
        "set -Eeuo pipefail\n"
        f"export MBC_LABEL_SPACE={quote(LABELS)}\n"
    )


# ---------------------------------------------------------------- stage 1: L1 routing
def stage_route(reads: Path, out: Path) -> str:
    l1 = CONFIG["l1_routing"]
    # Chunked wrapper: step48 slurps the whole reads.tsv into RAM, so on deep samples an unchunked p1
    # OOMs (or needs a scarce big-mem node). run_p1_route_chunked.sh bounds RAM to one chunk + the ~90G
    # compact DB; output is identical to unchunked except row_index (split_by_domain joins on read_id).
    chunk = l1.get("chunk_size", 3000000)
    body = header("p1_route", out, cpus=res("p1_route", "cpus", 16),
                  mem=res("p1_route", "mem", "64G"), time=res("p1_route", "time", "4:00:00")) + f"""
bash {quote(str(HERE / 'run_p1_route_chunked.sh'))} \\
  {quote(str(reads))} \\
  {quote(str(out / '01_routing'))} \\
  $SLURM_CPUS_PER_TASK \\
  {chunk} \\
  {quote(l1['binary'])} \\
  {quote(l1['compact_dir'])} \\
  {l1['compact_shards']}
"""
    return str(write(out / "scripts" / "p1_route.slurm", body))


# ---------------------------------------------------------------- stage 2: split by domain
def stage_split(reads: Path, out: Path) -> str:
    body = header("p2_split", out, cpus=res("p2_split", "cpus", 2),
                  mem=res("p2_split", "mem", "16G"), time=res("p2_split", "time", "1:00:00")) + f"""
{activate(gpu=False)}
python {quote(str(HERE / 'split_by_domain.py'))} \\
  --reads {quote(str(reads))} \\
  --routing {quote(str(out / '01_routing/domain_scores.tsv'))} \\
  --out-dir {quote(str(out / '02_by_domain'))}
"""
    return str(write(out / "scripts" / "p2_split.slurm", body))


# ---------------------------------------------------------------- stage 3: K25 evidence
def stage_evidence(domain: str, out: Path) -> str:
    d = CONFIG["domains"][domain]
    reads = f"{out}/02_by_domain/{domain}_reads.tsv"
    ev = f"{out}/03_evidence/{domain}"
    # bacteria's 380 GB index dominates the memory ask (490G fits the abundant tier, not 560G -> scarce
    # 2TB tier; see resources note in pipeline.json). Mem/cpus/time come from config, right-sized.
    stg = f"p3_{domain}"
    chunk = CONFIG.get("k25_chunk_size", 5000000)
    lines = [
        header(stg, out, cpus=res(stg, "cpus", 16),
               mem=res(stg, "mem", "490G" if domain == "bacteria" else ("64G" if domain == "fungi" else "32G")),
               time=res(stg, "time", "3:00:00")),
        f"mkdir -p {quote(ev)}",
        "",
        "# species evidence -- mphf_query_fast_multi (--chunk-size streams reads so RAM = index + one chunk,",
        "# not index + whole sample). Byte-identical to the model's own evidence (verified by cmp).",
        f"{quote(CONFIG['query_binary_multi'])} \\",
        f"  --index-root {quote(d['species_index'])} --num-shards {d['species_index_shards']} \\",
        f"  --priors {quote(d['priors'])} \\",
        f"  --reads {quote(reads)} --out {quote(ev + '/species_masks.tsv')} --top-k 20 "
        f"--chunk-size {chunk} --threads $SLURM_CPUS_PER_TASK",
        "",
    ]

    if d["parent_evidence"] == "rollup":
        lines += [
            "# genus/phylum by aggregating the species evidence upward through the taxonomy.",
            "# Verified byte-identical to the parent masks this domain's model was trained on.",
            activate(gpu=False),
            "for LVL in phylum genus; do",
            f"  python {quote(CONFIG['rollup_script'])} \\",
            f"    --species-mask-tsv {quote(ev + '/species_masks.tsv')} \\",
            f"    --metadata-tsv {quote(d['rollup_metadata'])} \\",
            "    --level $LVL \\",
            f'    --output-tsv {quote(ev)}/"${{LVL}}_masks.tsv" --summary-json {quote(ev)}/"${{LVL}}_masks.json"',
            "done",
        ]
    else:
        lines += [
            "# genus/phylum scored DIRECTLY at each rank. This domain's DB is direct-rank and a rollup",
            "# cannot reproduce it (the top-20 species reach only their own parents). Do not swap this",
            "# for a rollup without retraining this domain's models.",
            "for LVL in phylum genus; do",
            f"  {quote(CONFIG['query_binary'])} \\",
            f'    --index-root {quote(ASSETS)}/04_virus/db/"k25_${{LVL}}_mphf" --num-shards 1 \\',
            f'    --priors {quote(ASSETS)}/04_virus/db/"k25_${{LVL}}_mphf"/priors.tsv \\',
            "    --mask-shape parent \\",
            f'    --reads {quote(reads)} --out {quote(ev)}/"${{LVL}}_masks.tsv" --top-k 20 --threads $SLURM_CPUS_PER_TASK',
            "done",
        ]

    return str(write(out / "scripts" / f"p3_{domain}.slurm", "\n".join(lines) + "\n"))


# ---------------------------------------------------------------- stage 4: DL strict chain
def stage_predict(domain: str, out: Path) -> str:
    d = CONFIG["domains"][domain]
    ev = f"{out}/03_evidence/{domain}"
    stg = f"p4_{domain}"
    eb = f" --eval-batch-size {EVAL_BATCH}" if EVAL_BATCH not in ("0", "") else ""
    body = header(stg, out, cpus=res(stg, "cpus", 8),
                  mem=res(stg, "mem", "224G" if domain == "bacteria" else "64G"),
                  time=res(stg, "time", "8:00:00"), gpu=True) + f"""
{activate(gpu=True)}
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1

# Strict chained hard-route (step-54 model, epoch 8, fixed gate tau=0.5), resumable (rank+batch
# checkpoints, --ckpt-every-n-batches). Verified read-for-read identical to the step-54 evaluator;
# skips single-candidate reads (forced == k25) + vectorized post-proc.
python {quote(CONFIG['predict_script'])} \\
  --config {quote(d['model_config'])} \\
  --assets-root {quote(ASSETS)} \\
  --task-group {quote(d['task_group'])} \\
  --reads {quote(str(out / '02_by_domain' / (domain + '_reads.tsv')))} \\
  --masks-species {quote(ev + '/species_masks.tsv')} \\
  --masks-genus {quote(ev + '/genus_masks.tsv')} \\
  --masks-phylum {quote(ev + '/phylum_masks.tsv')} \\
  --fixed-threshold 0.5 --epochs 8,8,8{eb} \\
  --out {quote(str(out / '04_predictions' / (domain + '_predictions.tsv')))}
"""
    return str(write(out / "scripts" / f"p4_{domain}.slurm", body))


# ---------------------------------------------------------------- stage 5: merge
def stage_merge(reads: Path, out: Path, domains: list[str], paired_mode: str,
                out_name: str = "final_predictions.tsv") -> str:
    # p5 is a cheap, re-runnable "view" over the durable per-mate artifacts (04_predictions + routing):
    # paired-mode changes ONLY this step, not k25/DL, so a re-merge (see --from-predictions) can switch
    # lca<->independent or fix a merge bug with no GPU. out_name is parameterized so a re-merge in a
    # different mode does not clobber the original final_predictions.tsv (rorqual DEPLOYMENT_CHANGELOG O).
    body = header("p5_merge", out, cpus=res("p5_merge", "cpus", 2),
                  mem=res("p5_merge", "mem", "32G"), time=res("p5_merge", "time", "1:00:00")) + f"""
{activate(gpu=False)}
python {quote(str(HERE / 'merge_results.py'))} \\
  --reads {quote(str(reads))} \\
  --routing {quote(str(out / '01_routing/domain_scores.tsv'))} \\
  --pred-dir {quote(str(out / '04_predictions'))} \\
  --domains {','.join(domains)} \\
  --paired-mode {paired_mode} \\
  --out {quote(str(out / out_name))}
"""
    return str(write(out / "scripts" / "p5_merge.slurm", body))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True, help="TSV with read_id and sequence columns")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--domains", default=",".join(DL_DOMAINS))
    ap.add_argument("--paired-mode", choices=["lca", "strict", "independent"], default="lca",
                    help="paired-end reconciliation at merge (auto-detected from /1 /2 read ids); "
                         "lca=finest agreeing rank, strict=species must agree, independent=per-mate")
    ap.add_argument("--from-predictions", action="store_true",
                    help="Re-run ONLY the merge (p5) from the existing 04_predictions + 01_routing in "
                         "--out-dir -- no route/split/k25/DL, no GPU, minutes on CPU. Use to switch "
                         "--paired-mode (lca<->independent) or re-merge after a merge fix. The per-mate "
                         "04_predictions is the durable artifact; merge is just a view over it.")
    ap.add_argument("--merge-out-name", default="",
                    help="Output filename for the merged result (under --out-dir). Default "
                         "final_predictions.tsv for a full run; for --from-predictions it defaults to "
                         "final_predictions_<paired-mode>.tsv so a re-merge does not clobber the original.")
    ap.add_argument("--local", action="store_true",
                    help="Standalone mode: run all stages sequentially on THIS machine (no Slurm/sbatch). "
                         "Also enabled by deployment.type=standalone in site.json. Needs >=500G RAM + "
                         ">=16G VRAM GPU on this one machine (the k25 index + DL both run here).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global LOCAL
    if args.local:
        LOCAL = True

    reads = Path(args.reads).resolve()
    out = Path(args.out_dir).resolve()
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in domains if d not in CONFIG["domains"]]
    if unknown:
        sys.exit(f"unknown domain(s): {unknown}")
    if not domains or len(set(domains)) != len(domains):
        ap.error("--domains must list one or more distinct domains")
    if not reads.is_file():
        ap.error(f"reads TSV not found: {reads}")
    if args.merge_out_name and Path(args.merge_out_name).name != args.merge_out_name:
        ap.error("--merge-out-name must be a filename within --out-dir")
    if not args.dry_run:
        needed = [Path(LABELS) / d / "metadata.tsv" for d in domains]
        if not args.from_predictions:
            needed += [Path(LABELS) / d / "label_space.tsv" for d in domains]
            needed += [Path(CONFIG["l1_routing"]["compact_dir"]),
                       Path(CONFIG["l1_routing"]["binary"]), Path(CONFIG["query_binary_multi"])]
            if "virus" in domains:
                needed.append(Path(CONFIG["query_binary"]))
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            ap.error("missing classifier assets or binaries:\n  " + "\n  ".join(missing))

    # --- re-merge-only path: reuse existing per-mate predictions, run p5 alone (no k25/DL) ---
    if args.from_predictions:
        need = [out / "01_routing" / "domain_scores.tsv"] + \
               [out / "04_predictions" / f"{d}_predictions.tsv" for d in domains]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            sys.exit("--from-predictions needs a prior full run in --out-dir; missing:\n  "
                     + "\n  ".join(missing))
        (out / "scripts").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        name = args.merge_out_name or f"final_predictions_{args.paired_mode}.tsv"
        print(f"[from-predictions] re-merge only, paired-mode={args.paired_mode} -> {name}")
        print(f"                  reusing 04_predictions (no k25/DL/GPU)\n")
        sbatch(stage_merge(reads, out, domains, args.paired_mode, out_name=name), None, args.dry_run)
        print(f"\nfinal output will be: {out}/{name}")
        return 0

    for sub in ("scripts", "logs", "01_routing", "02_by_domain", "03_evidence", "04_predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    print(f"reads:   {reads}")
    print(f"out:     {out}")
    print(f"domains: {domains}\n")

    j1 = sbatch(stage_route(reads, out), None, args.dry_run)
    j2 = sbatch(stage_split(reads, out), [j1], args.dry_run)

    predict_jobs = []
    for domain in domains:
        j3 = sbatch(stage_evidence(domain, out), [j2], args.dry_run)
        predict_jobs.append(sbatch(stage_predict(domain, out), [j3], args.dry_run))

    full_name = args.merge_out_name or "final_predictions.tsv"
    sbatch(stage_merge(reads, out, domains, args.paired_mode, out_name=full_name), predict_jobs, args.dry_run)
    print(f"\nfinal output will be: {out}/{full_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
