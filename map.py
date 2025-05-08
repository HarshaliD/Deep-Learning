#!/usr/bin/env python3
import sys

# Skip the header
for idx, line in enumerate(sys.stdin):
    if idx == 0:
        continue  # Skip header
    parts = line.strip().split(",")
    if len(parts) != 6:
        continue  # Skip malformed lines

    year = parts[0]
    try:
        max_temp = float(parts[3])
        min_temp = float(parts[4])
    except ValueError:
        continue  # Skip lines with non-numeric temperature

    # Emit key-value pairs
    print(f"{year}\t{max_temp},{min_temp},1")
