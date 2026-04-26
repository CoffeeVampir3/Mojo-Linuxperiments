#!/usr/bin/env fish

set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/linuxperiments
set DEFAULT_API_HOST 0.0.0.0
set DEFAULT_API_PORT 33322

if test (count $argv) -lt 1
    echo "usage: ./remote_python_chat.fish <api-key> [host] [port]"
    exit 1
end

set API_KEY $argv[1]

if test (count $argv) -ge 2
    set API_HOST $argv[2]
else
    set API_HOST $DEFAULT_API_HOST
end

if test (count $argv) -ge 3
    set API_PORT $argv[3]
else
    set API_PORT $DEFAULT_API_PORT
end

set API_KEY_ESCAPED (string escape -- $API_KEY)
set API_HOST_ESCAPED (string escape -- $API_HOST)
set API_PORT_ESCAPED (string escape -- $API_PORT)
set REMOTE_PATH_ESCAPED (string escape -- $REMOTE_PATH)

rsync -av \
    --exclude='.*' \
    --exclude='pixi.lock' \
    --exclude='__pycache__' \
    --exclude='__mojocache__' \
    --exclude='validation/.venv' \
    --exclude='test_smollm2_bin' \
    --exclude='test_smollm2_tp3_bin' \
    --exclude='test_tp3_bin' \
    --exclude='test_rings_bin' \
    --exclude='fence_experiment_bin' \
    --exclude='tp_param_bin' \
    --include='checkpoints/SmolLM2/model.safetensors' \
    --exclude='checkpoints/**/*.safetensors' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

echo "✓ Synced to $REMOTE_HOST:$REMOTE_PATH"
echo "→ Running OpenAI-compatible chat server on $REMOTE_HOST:$API_PORT"

ssh $REMOTE_USER@$REMOTE_HOST "pkill -TERM -f '[p]ython_glue/m27_openai_server.py' 2>/dev/null || true"

set BASH_CMD "cd $REMOTE_PATH_ESCAPED || exit 1; setsid env PYTHONUNBUFFERED=1 M27_API_KEY=$API_KEY_ESCAPED M27_HOST=$API_HOST_ESCAPED M27_PORT=$API_PORT_ESCAPED pixi run python python_glue/m27_openai_server.py & server_pid=\$!; trap 'kill -TERM -- -\$server_pid 2>/dev/null; wait \$server_pid 2>/dev/null' HUP INT TERM EXIT; wait \$server_pid"
set REMOTE_CMD "bash -lc "(string escape -- $BASH_CMD)

ssh $REMOTE_USER@$REMOTE_HOST $REMOTE_CMD
