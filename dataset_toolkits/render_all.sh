#!/bin/bash
# Launch all 8 render shards in parallel, one per GPU.
#
# Usage:  bash render_all.sh
#
# 2024 objects / 8 shards = 253 objects per shard.
# Logs: shard_<i>.log   Output: renders_all/<object_id>/{000..024}.png
# Safe to re-run: objects with an existing transforms.json are skipped.

set -euo pipefail
TOOLKIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TOOLKIT"

RANGE=${1:-253}

for i in $(seq 0 7); do
    nohup bash run_shard.sh "$i" "$RANGE" > "shard_$i.log" 2>&1 &
done

echo "Launched 8 shards (range=$RANGE). Tail a log with: tail -f $TOOLKIT/shard_0.log"
wait
echo "All shards finished. Output in: $TOOLKIT/renders_all"
echo "Rendered objects: $(ls "$TOOLKIT/renders_all" 2>/dev/null | wc -l)"
