#!/bin/bash
for p in $(pgrep -f "vllm serve") $(pgrep -f "toy_proxy_server.py"); do kill $p 2>/dev/null; done
sleep 8
