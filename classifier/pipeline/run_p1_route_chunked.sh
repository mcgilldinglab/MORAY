#!/bin/bash
# Chunked wrapper around step48_compact_domain_db's L1 routing query (mode query-reads-stream6).
# No C++ changes: step48's load_reads() slurps the WHOLE reads.tsv into a vector, so its RAM grows
# with sample size (narval saw a 48 GB reads.tsv OOM p1 at 128 G, and 560 G to survive pushed the
# job onto a scarce big-mem tier -- see pitfalls_and_fixes.md 8.1-8.2). This wrapper splits the reads
# into CHUNK_SIZE-row pieces and runs the binary once per chunk, bounding peak RAM to one chunk's
# reads (+ the ~90 GB compact DB, which is re-opened per chunk -- acceptable at 90 GB, unlike the
# 380 GB bacteria k25 index which is instead kept resident by mphf_query_fast_multi).
#
# CORRECTNESS (each chunk MUST be isolated, or data is silently dropped):
#  - own --partial-dir: the binary caches "shard already done" state there and will SKIP a shard it
#    thinks is done -> a shared partial-dir drops that shard's k-mer events for later chunks.
#  - own --output-tsv: the binary skips the whole run if <output-tsv>.done.json exists next to it ->
#    a shared output path means every chunk after the first does nothing.
#  - output's row_index is unique only WITHIN one invocation; chunked row_index != unchunked. Verified
#    harmless: split_by_domain.py joins on read_id only (never row_index). method-label is kept as
#    'deployment_l1' (same as the unchunked stage_route) so the merged domain_scores.tsv differs from
#    an unchunked run ONLY in row_index.
#
# Self-contained inside the package: BIN/COMPACT_DIR/shards are passed by the caller (run_pipeline.py
# resolves {PKG}/{ASSETS} from CONFIG), never hardcoded, so the package stays portable.
#
# Usage: run_p1_route_chunked.sh <reads_tsv> <out_dir> <threads> <chunk_size> <bin> <compact_dir> [compact_shards]
#   writes <out_dir>/domain_scores.tsv (merged, same format as the unchunked run)
set -euo pipefail

READS_TSV="$1"
OUT_DIR="$2"
THREADS="$3"
CHUNK_SIZE="${4:-3000000}"
BIN="$5"
COMPACT_DIR="$6"
COMPACT_SHARDS="${7:-256}"

CHUNK_DIR="${OUT_DIR}/p1_chunks_tmp"
mkdir -p "${CHUNK_DIR}"
FINAL_OUT="${OUT_DIR}/domain_scores.tsv"

HEADER=$(head -1 "${READS_TSV}")

echo "Splitting ${READS_TSV} into ${CHUNK_SIZE}-row chunks: $(date -Is)"
tail -n +2 "${READS_TSV}" | split -l "${CHUNK_SIZE}" -d -a 4 --additional-suffix=.body - "${CHUNK_DIR}/chunk_"

TMP_MERGED="${FINAL_OUT}.merging"
: > "${TMP_MERGED}"
FIRST=1
for body in "${CHUNK_DIR}"/chunk_*.body; do
    [[ -e "${body}" ]] || continue
    idx=$(basename "${body}" .body)
    chunk_tsv="${CHUNK_DIR}/${idx}.tsv"
    { echo "${HEADER}"; cat "${body}"; } > "${chunk_tsv}"
    rm -f "${body}"

    chunk_out="${CHUNK_DIR}/${idx}.output.tsv"
    chunk_partial="${CHUNK_DIR}/${idx}.partials"     # per-chunk (correctness detail above)
    mkdir -p "${chunk_partial}"

    echo "Processing ${idx}: $(date -Is)"
    "${BIN}" --mode query-reads-stream6 \
        --reads-tsv "${chunk_tsv}" \
        --compact-dir "${COMPACT_DIR}" \
        --output-tsv "${chunk_out}" \
        --partial-dir "${chunk_partial}" \
        --compact-shards "${COMPACT_SHARDS}" \
        --threads "${THREADS}" \
        --k 25 \
        --method-label deployment_l1

    if [[ "${FIRST}" -eq 1 ]]; then
        head -1 "${chunk_out}" >> "${TMP_MERGED}"
        FIRST=0
    fi
    tail -n +2 "${chunk_out}" >> "${TMP_MERGED}"

    rm -f "${chunk_tsv}" "${chunk_out}"
    rm -rf "${chunk_partial}"
done

mv "${TMP_MERGED}" "${FINAL_OUT}"
rm -rf "${CHUNK_DIR}"
echo "Done: $(date -Is) -> ${FINAL_OUT}"
