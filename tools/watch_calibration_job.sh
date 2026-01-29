#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-}"
API_ROOT="${API_ROOT:-http://127.0.0.1:8000}"
INTERVAL="${INTERVAL:-1}"

if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

while true; do
  curl -s "${API_ROOT}/calibration/jobs/${JOB_ID}" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("{job_id} {phase} {processed}/{total} ({progress:.2f}%)".format(job_id=d.get("job_id"), phase=d.get("phase"), processed=d.get("processed"), total=d.get("total"), progress=(d.get("progress") or 0)*100))'
  sleep "$INTERVAL"
done
