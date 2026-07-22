#!/bin/bash
# Render one shard of the PhysX-Mobility dataset on a single GPU.
#
# Usage:  bash run_shard.sh <shard_index> [range]
#   shard_index : which shard to render (0-based)
#   range       : objects per shard (default 253 = ceil(2024/8))
#
# Each shard runs in its own working directory (_shard_<i>) so that the
# shared views.json written per-object does not collide between parallel
# processes. Each shard is pinned to GPU <shard_index>.

set -euo pipefail

SHARD=${1:?usage: bash run_shard.sh <shard_index> [range]}
RANGE=${2:-253}

TOOLKIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$TOOLKIT/_shard_$SHARD"
mkdir -p "$WORKDIR"

# Resolve the dataset to its real path so the shard symlink does not depend on
# the intermediate $TOOLKIT/PhysX_mobility link (which can go missing).
DATA_REAL="$(readlink -f "$TOOLKIT/PhysX_mobility")"
if [ ! -d "$DATA_REAL/partseg" ]; then
    # Fall back to the canonical dataset location under the repo.
    DATA_REAL="$(readlink -f "$TOOLKIT/../dataset/PhysX_mobility")"
fi
if [ ! -d "$DATA_REAL/partseg" ]; then
    echo "[shard $SHARD] ERROR: cannot find PhysX_mobility/partseg (looked at $DATA_REAL)" >&2
    exit 1
fi

# Symlink the resources render_cond_mobility.py expects relative to CWD.
ln -sfn "$TOOLKIT/blender_script"           "$WORKDIR/blender_script"
ln -sfn "$TOOLKIT/utils.py"                  "$WORKDIR/utils.py"
ln -sfn "$TOOLKIT/render_cond_mobility.py"   "$WORKDIR/render_cond_mobility.py"
ln -sfn "$DATA_REAL"                         "$WORKDIR/PhysX_mobility"

cd "$WORKDIR"

# Pin this shard to one GPU. For >8 shards, wrap around the 8 cards.
export CUDA_VISIBLE_DEVICES=$(( SHARD % 8 ))

# System Blender 2.82 cannot build CUDA kernels for RTX 4090; use bundled 3.6.
export BLENDER="$TOOLKIT/blender-3.6/blender"

echo "[shard $SHARD] GPU=$CUDA_VISIBLE_DEVICES range=$RANGE cwd=$WORKDIR"
python render_cond_mobility.py \
    --index "$SHARD" \
    --range "$RANGE" \
    --datapath ./PhysX_mobility/partseg \
    --basepath "$TOOLKIT/renders_all"
echo "[shard $SHARD] done"
