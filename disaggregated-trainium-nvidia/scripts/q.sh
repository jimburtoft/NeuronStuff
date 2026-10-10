#!/bin/bash
# q.sh ENDPOINT PROMPT [N]
curl -s -m 180 http://$1/v1/completions -H "Content-Type: application/json" -d "$(python3 -c 'import json,sys;print(json.dumps({"model":"llama","prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$2" "${3:-30}")" | python3 -c "import sys,json
try: print(repr(json.load(sys.stdin)['choices'][0]['text']))
except Exception as e: print('ERR',e)"
