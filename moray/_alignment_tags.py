#!/usr/bin/env python
"""Add NH:i:<n> tags to a name-grouped SAM stream (bowtie2 -k output lacks NH).

bowtie2 --local -k reports up to k alignments per read but emits NO NH tag, so
featureCounts cannot tell multimappers from unique reads (unique==multimap). This
reconstructs NH per read/mate = number of MAPPED alignment records sharing the
QNAME and the same mate bit (read1/read2/single), matching STAR's NH semantics.

Records of the same QNAME must be consecutive (true for bowtie2 output; for an
existing BAM pipe through `samtools collate -O -u` first). SAM in on stdin, SAM
out on stdout. NH is appended only to MAPPED records; unmapped pass unchanged.
"""
import sys

def flush(buf, out):
    if not buf:
        return
    # count mapped records per mate group
    n = {0: 0, 1: 0, 2: 0}   # 0=single, 1=read1, 2=read2
    metas = []
    for line in buf:
        t = line.find('\t')
        t2 = line.find('\t', t + 1)
        flag = int(line[t + 1:t2])
        mapped = not (flag & 0x4)
        mate = 1 if (flag & 0x40) else (2 if (flag & 0x80) else 0)
        metas.append((mate, mapped))
        if mapped:
            n[mate] += 1
    w = out.write
    for line, (mate, mapped) in zip(buf, metas):
        if mapped:
            w(line[:-1] + '\tNH:i:%d\n' % n[mate])
        else:
            w(line)

def main():
    out = sys.stdout
    buf = []
    cur = None
    w = out.write
    for line in sys.stdin:
        if line[0] == '@':
            w(line)
            continue
        q = line[:line.find('\t')]
        if q != cur:
            flush(buf, out)
            buf = [line]
            cur = q
        else:
            buf.append(line)
    flush(buf, out)

if __name__ == '__main__':
    main()
