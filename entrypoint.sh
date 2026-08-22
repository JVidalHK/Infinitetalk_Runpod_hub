#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Find out what hardware this worker got before anything tries to use it.
#
# The base images carry SageAttention compiled for one architecture. Landing on
# a foreign one does not degrade to a slower kernel — the first attention call
# raises, ComfyUI finishes with no output node, and the job returns only
# "video not found". Starting ComfyUI on sdpa instead costs about one percent
# and keeps the worker serving, which is a far better trade than a crash.
echo "Probing GPU..."
python /gpu_probe.py || echo "gpu_probe failed; assuming SageAttention is unavailable"

SAGE_FLAG="--use-sage-attention"
if python -c "import json,sys; sys.exit(0 if json.load(open('/gpu_info.json')).get('sage_ok') else 1)" 2>/dev/null; then
    echo "SageAttention has a kernel for this GPU; enabling it."
else
    echo "SageAttention has no kernel for this GPU; starting ComfyUI on sdpa."
    SAGE_FLAG=""
fi

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen $SAGE_FLAG &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120  # 최대 2분 대기
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler in the foreground
# 이 스크립트가 컨테이너의 메인 프로세스가 됩니다.
echo "Starting the handler..."
exec python handler.py
