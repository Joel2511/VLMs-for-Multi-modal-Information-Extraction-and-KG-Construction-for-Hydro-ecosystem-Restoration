#!/usr/bin/env bash
# Fast probe of every currently-idle node's PER-GPU VRAM (not node total).

PARTITION="publicgpu"
MIN_VRAM_MB=24000   # 24GB per individual GPU

IDLE_NODES=$(sinfo -p "$PARTITION" -N -h -o "%N %t" | awk '$2=="idle"{print $1}')

if [ -z "$IDLE_NODES" ]; then
    echo "No idle nodes found."
    exit 1
fi

echo "Probing idle nodes: $IDLE_NODES"
echo "================================================"

GOOD_NODES=()

for node in $IDLE_NODES; do
    echo "[$node]"

    # Discover how many GPUs this node actually has (parses Gres like gpu:a100:4)
    GPU_COUNT=$(sinfo -N -h -o "%N %G" -n "$node" | grep -oE 'gpu:[^,]*' | grep -oE '[0-9]+$' | head -1)
    [ -z "$GPU_COUNT" ] && GPU_COUNT=1

    RESULT=$(timeout 30 srun --immediate=15 --partition="$PARTITION" --nodelist="$node" \
        --gres=gpu:"$GPU_COUNT" --time=00:01:00 --mem=4G --cpus-per-task=1 \
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>&1)

    if [ $? -ne 0 ] || [ -z "$RESULT" ]; then
        echo "  UNAVAILABLE ($RESULT)"
        continue
    fi

    MIN_VRAM=999999999
    TOTAL_VRAM=0

    while IFS=',' read -r idx name mem; do
        idx=$(echo "$idx" | xargs); name=$(echo "$name" | xargs); mem=$(echo "$mem" | xargs)
        [ -z "$mem" ] && continue
        echo "  GPU$idx ($name): ${mem} MB"
        TOTAL_VRAM=$((TOTAL_VRAM + mem))
        if [ "$mem" -lt "$MIN_VRAM" ]; then
            MIN_VRAM=$mem
        fi
    done <<< "$RESULT"

    echo "  --> Node total: ${TOTAL_VRAM} MB across GPUs, smallest single GPU: ${MIN_VRAM} MB"

    if [ "$MIN_VRAM" -ge "$MIN_VRAM_MB" ]; then
        GOOD_NODES+=("$node")
    fi
done

echo "================================================"
if [ ${#GOOD_NODES[@]} -gt 0 ]; then
    echo "CANDIDATES where EVERY individual GPU >= ${MIN_VRAM_MB}MB: ${GOOD_NODES[*]}"
else
    echo "No idle node has an individual GPU >= ${MIN_VRAM_MB}MB."
fi