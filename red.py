#!/usr/bin/env python3
import sys

from collections import defaultdict

temp_data = defaultdict(lambda: [0, 0, 0])  # max_sum, min_sum, count

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        continue
    year, values = parts
    try:
        max_temp, min_temp, count = map(float, values.split(","))
        temp_data[year][0] += max_temp
        temp_data[year][1] += min_temp
        temp_data[year][2] += count
    except ValueError:
        continue

# Output: Year -> avg max, avg min
for year in sorted(temp_data):
    max_sum, min_sum, count = temp_data[year]
    avg_max = max_sum / count
    avg_min = min_sum / count
    print(f"{year}\tAvg Max Temp: {avg_max:.2f}, Avg Min Temp: {avg_min:.2f}")
