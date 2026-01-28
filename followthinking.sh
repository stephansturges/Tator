tail -f logs/prepass_readable/latest.log | awk 'match($0,/delta=/){printf "%s", substr($0,RSTART+6); fflush()}'
