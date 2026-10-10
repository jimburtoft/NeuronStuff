#!/bin/bash
# proxy_launch.sh LOG PREFILL_HOST PREFILL_PORT DECODE_HOST DECODE_PORT
. ~/venv/bin/activate; cd ~
setsid nohup python ~/toy_proxy_server.py --port ${PORT:-8000} --prefiller-host $2 --prefiller-port $3 --decoder-host $4 --decoder-port $5 > ~/$1.log 2>&1 < /dev/null &
