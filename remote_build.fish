#!/usr/bin/env fish
set LOCAL_PATH .
set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/Thread

rsync -av --exclude='.*' --exclude='pixi.lock' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/
echo "✓ Pushed to server!"
ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && ./run_tests.fish"
