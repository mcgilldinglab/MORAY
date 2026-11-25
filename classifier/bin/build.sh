#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cxx="${CXX:-g++}"
"$cxx" -O3 -fopenmp -std=c++17 -o "$root/bin/mphf_query_fast" "$root/src/mphf_query_fast.cpp"
"$cxx" -O3 -fopenmp -std=c++17 -o "$root/bin/mphf_query_fast_multi" "$root/src/mphf_query_fast_multi.cpp"
"$cxx" -O3 -fopenmp -std=c++17 -o "$root/bin/step48_compact_domain_db" "$root/src/step48_compact_domain_db.cpp" -lz
