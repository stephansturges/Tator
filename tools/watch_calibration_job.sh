#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi
JOB_ID="$1"
PYHELPER="/tmp/print_job.py"
while true; do
  curl -s "http://127.0.0.1:8000/calibration/jobs/${JOB_ID}" | python3 "$PYHELPER"
  sleep 5
done
