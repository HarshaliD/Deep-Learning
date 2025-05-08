#!/usr/bin/env python3
import sys
from collections import defaultdict

counts = defaultdict(int)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue  # skip empty lines
    parts = line.split("\t")
    if len(parts) != 2:
        continue  # skip malformed lines
    key, val = parts
    try:
        counts[key] += int(val)
    except ValueError:
        continue  # skip lines with non-integer values

for key in sorted(counts):
    print(f"{key}\t{counts[key]}")

