#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
INTERVAL="${INTERVAL:-5}"
JOB_ID=""

usage() {
  cat <<'EOF'
Usage: watch_calibration_job.sh [--base-url URL] [--interval SECONDS] <job_id>

Poll a calibration job and pretty-print the JSON response.
Defaults: BASE_URL or http://127.0.0.1:8000, INTERVAL or 5 seconds.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:?--base-url requires a URL}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:?--interval requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      if [[ $# -gt 1 || -n "$JOB_ID" ]]; then
        echo "Only one job_id may be supplied." >&2
        usage >&2
        exit 2
      fi
      if [[ $# -eq 1 ]]; then
        JOB_ID="$1"
        shift
      fi
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$JOB_ID" ]]; then
        echo "Only one job_id may be supplied." >&2
        usage >&2
        exit 2
      fi
      JOB_ID="$1"
      shift
      ;;
  esac
done

if [[ -z "$JOB_ID" ]]; then
  usage >&2
  exit 1
fi

BASE_URL="${BASE_URL%/}"
while true; do
  curl -fsS "${BASE_URL}/calibration/jobs/${JOB_ID}" | python3 -m json.tool
  sleep "$INTERVAL"
done
