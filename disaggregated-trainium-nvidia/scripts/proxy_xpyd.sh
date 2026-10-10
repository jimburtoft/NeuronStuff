#!/bin/bash
# Mixed prefill pool: proxy_xpyd.sh <TRN2_IP> <GPU_IP>
#   prefill on trn2:8100 and GPU:8210, decode on GPU:8200
TRN2_IP=${1:?trn2 private IP}; GPU_IP=${2:?GPU private IP}
. ~/venv/bin/activate; cd ~
setsid nohup python ~/toy_proxy_server.py --port 8002 --prefiller-hosts $TRN2_IP $GPU_IP --prefiller-ports 8100 8210 \
  --decoder-hosts $GPU_IP --decoder-ports 8200 > ~/proxy_xpyd.log 2>&1 < /dev/null &
