// On-disk format contract for the K25 MPHF index. Shared by every builder and every query tool.
//
// WHY THIS FILE EXISTS
// The fingerprint is COMPUTED at build time and stored in meta.bin, then RECOMPUTED at query time
// and compared. So the fingerprint function is part of the on-disk format, not an implementation
// detail. The first version of this index stored a raw bit-slice fingerprint; when that function was
// replaced, every already-built index silently became unreadable -- the new query recomputed a
// different fingerprint, matched nothing, and returned empty results for every read. It did not
// crash. It did not warn. That is the worst possible failure mode.
//
// So: the header below stamps the format version and the fingerprint algorithm id into the file, and
// the loader REFUSES to load an index whose stamp it does not understand. If you change how any
// stored field is computed, bump FP_ALGO_ID (or FORMAT_VER) -- do not let an old index be read by a
// new binary.
//
// meta.bin layout (format v2):
//     MphfHeader (40 bytes)
//     u32 fp[n_keys]           fingerprint, see kmer_fp()
//     u32 payload[n_keys]      bit31=0 -> the species label itself (singleton, ~94.5% of k-mers)
//                              bit31=1 -> low 31 bits are a byte offset into blob
//     u8  blob[blob_bytes]     delta-varint species lists for multi-species k-mers
// mphf.bin: the BBHash MPHF, saved by boomphf::mphf::save().
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;

static const u64 MPHF_MAGIC      = 0x3148504D35324BULL;  // "K25MPH1"
static const u32 MPHF_FORMAT_VER = 2;                    // v1 = headerless (retired), v2 = this

// Fingerprint algorithm ids. NEVER reuse an id for a different function.
static const u32 MPHF_FP_BITSLICE_RETIRED = 0;  // (kmer>>9)&0xffffffff -- DO NOT USE, see below
static const u32 MPHF_FP_SPLITMIX64       = 1;  // (u32)(splitmix64(kmer) >> 32)
static const u32 MPHF_FP_CURRENT          = MPHF_FP_SPLITMIX64;

struct MphfHeader {
    u64 magic;
    u32 format_version;
    u32 fp_algo_id;
    u64 n_keys;
    u64 blob_bytes;
    u64 reserved;
};

// 32-bit fingerprint: splitmix64 finalizer over the FULL 50-bit k-mer.
//
// A raw bit-slice must NOT be used here. The retired version was (kmer>>9)&0xffffffff, i.e. k-mer
// bits 9-40, which is blind to the high bits 41-49 -- and those are the same bits BBHash's slot hash
// barely avalanches into the low output bits it uses to pick a slot. The two blind spots coincide, so
// a k-mer differing from a DB k-mer only in its last few bases both lands on that k-mer's slot AND
// matches its fingerprint. Measured false-positive rate at bacteria density (41.31 B k-mers): 2^-12,
// not the 2^-32 a 32-bit fingerprint should give -- 3,296 phantom matches over 14.5 M absent lookups.
// This function is independent of BBHash's hash, restoring the true 2^-32.
static inline u32 kmer_fp(u64 k) {
    u64 x = k + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (u32)(x >> 32);
}

static inline MphfHeader mphf_make_header(u64 n_keys, u64 blob_bytes) {
    MphfHeader h;
    h.magic = MPHF_MAGIC;
    h.format_version = MPHF_FORMAT_VER;
    h.fp_algo_id = MPHF_FP_CURRENT;
    h.n_keys = n_keys;
    h.blob_bytes = blob_bytes;
    h.reserved = 0;
    return h;
}

// Returns nullptr if the header is usable, else a message explaining why the index must be rebuilt.
static inline const char* mphf_check_header(const MphfHeader& h) {
    if (h.magic != MPHF_MAGIC) return "not a K25 MPHF index (bad magic) -- wrong file or truncated";
    if (h.format_version != MPHF_FORMAT_VER) return "index format version mismatch -- rebuild the index";
    if (h.fp_algo_id != MPHF_FP_CURRENT) return "index was built with a different fingerprint algorithm -- rebuild the index (an old index read by this binary would match NOTHING)";
    return nullptr;
}
