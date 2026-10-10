#!/bin/bash
# gpu_launch.sh LOGNAME VAR=val ...   (no pkill; caller stops processes with stop_all.sh)
L=$1; shift; cd ~; env "$@" setsid nohup ~/serve_gpu.sh > ~/$L.log 2>&1 < /dev/null &
