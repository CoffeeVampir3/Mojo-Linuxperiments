#!/usr/bin/env fish

set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/linuxperiments
set DEFAULT_TARGET test_gemma4_butterquant.mojo

if test (count $argv) -gt 0
    set TARGET $argv[1]
else
    set TARGET $DEFAULT_TARGET
end

if not test -f $TARGET
    echo "Target not found: $TARGET"
    exit 1
end

set BINARY (string replace -r '\.mojo$' '' (basename $TARGET))

rsync -av \
    --exclude='.*' \
    --exclude='pixi.lock' \
    --exclude='__pycache__' \
    --exclude='validation/.venv' \
    --exclude='test_smollm2_bin' \
    --exclude='test_smollm2_tp3_bin' \
    --exclude='test_tp3_bin' \
    --exclude='test_tp_bin' \
    --exclude='test_rings_bin' \
    --exclude='fence_experiment_bin' \
    --exclude='tp_param_bin' \
    --include='checkpoints/SmolLM2/model.safetensors' \
    --exclude='checkpoints/**/*.safetensors' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

echo "✓ Synced to $REMOTE_HOST:$REMOTE_PATH"
echo "→ Building and running $TARGET on $REMOTE_HOST"

ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && env MOJO_ENABLE_RUNTIME=0 pixi run mojo build -I . -D ASSERT=all $TARGET && ./$BINARY"
