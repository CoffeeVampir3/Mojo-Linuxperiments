#!/usr/bin/env fish

set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/linuxperiments

if test (count $argv) -gt 0
    echo "usage: ./clear_remote_mojo_binding.fish"
    exit 1
end

echo "Clearing remote Mojo Python binding cache on $REMOTE_HOST"
ssh $REMOTE_USER@$REMOTE_HOST "rm -rf $REMOTE_PATH/python_glue/__mojocache__"
echo "Cleared $REMOTE_PATH/python_glue/__mojocache__"
