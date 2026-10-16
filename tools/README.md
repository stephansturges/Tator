# Operator tools

Run these commands from the repository root using the environment for the service
you are maintaining. The [documentation index](../docs/index.rst) separates the
hosted server from the local workbench.

## Hosted server

These commands support the server-owned dataset store. Read the linked guide
before changing a live installation; a dataset ZIP is not an installation backup.

| Task | Command help | Guide |
| --- | --- | --- |
| Create the first administrator | `python -m tools.hosted_admin --help` | [Deployment](../docs/hosted_deployment.rst) |
| Inventory and import a legacy dataset | `python -m tools.hosted_migrate --help` | [Migration](../docs/hosted_migration.rst) |
| Encrypt a backup or restore into a new directory | `python -m tools.hosted_recovery --help` | [Recovery](../docs/hosted_operations.rst) |

Administrator bootstrap prompts for the password; do not put it in shell
arguments. Migration first creates a reviewable inventory, then copies from the
source. Recovery uses `age`, with the private decryption identity held separately
from the server.

## Local workbench

The following tools work with the original local backend or standalone dataset
files. Use the hosted UI and migration/recovery commands for hosted datasets;
editing files beneath hosted storage bypasses its annotation history.

### Start the macOS backend

```bash
tools/run_macos_backend.sh
```

Open `http://127.0.0.1:8000/`. Set `PORT=8080` before the command to choose another
port. The launcher exits if the requested port is already occupied.

### Reorder a local label map

```bash
python tools/reorder_labelmap.py --help
```

### Inspect class mismatches

```bash
python tools/detect_missclassifications.py \
  --images_path <images> \
  --labels_path <labels> \
  --interactive
```

The interactive inspector supports suggested remaps, undo and partial skip-log
saves.

### Watch a calibration job

```bash
tools/watch_calibration_job.sh --base-url http://127.0.0.1:8000 <job_id>
```

Set `INTERVAL` to change the polling interval.
