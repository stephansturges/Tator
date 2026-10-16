Deploying and updating a hosted server
======================================

Hosted Tator is a self-hosted service: users sign in to a server-owned dataset,
annotators submit work, and administrators approve the canonical versions used
by analysis, training and export. Read the `workspace workflow
<hosted_workflow.rst>`_ for those roles and the `storage and recovery guide
<hosted_operations.rst>`_ before importing existing data.

Deployment acceptance remains open for the planned dual-A6000 host: actual GPU
model behavior, WALDO40 migration, live provider execution and final cutover
have not been qualified on that target. The steps below describe deployment;
local implementation and test results do not establish that it has happened.

Prepare the runtime
-------------------

Use Python 3.11–3.13, one control process, supervised model subprocesses and a
private state directory on local disk. Production startup enforces SQLite
**3.51.3 or newer**, including its WAL fix. Check the library linked to the
service's Python interpreter, not just the system ``sqlite3`` executable.
Keep the live database off NFS, SMB and synchronized folders.

The Linux/systemd example uses these paths; adapt all references together if
choosing another layout:

* ``/opt/tator``: reviewed source checkout, with its environment at ``.venv``.
* ``/var/lib/tator-hosted``: new writable state, owned by the service user,
  mode 0700. Never point this at an original dataset directory.
* ``/etc/tator``: operator-owned environment and model configuration.
* ``/srv/tator-models``: trusted, readable model sources; installed copies and
  caches are stored inside the state directory.

From the source checkout, create the environment and install the control
requirements::

    cd /opt/tator
    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r requirements-hosted.txt
    .venv/bin/python -c 'import sqlite3; print(sqlite3.sqlite_version); assert sqlite3.sqlite_version_info >= (3, 51, 3)'
    .venv/bin/python -m pip check

Choose a Python build linked to a sufficiently recent SQLite if this check
fails. Installing Python packages does not upgrade its linked SQLite library.
The `hosted requirements <../requirements-hosted.txt>`_ include Pydantic 2,
HTTPX, Pillow, PyPNG and NumPy/SciPy. PyPNG preserves exact 16-bit PNG
transparency during working-image conversion.

These requirements cover the control process. Before enabling neural tools,
native analyses or the multistage paid-caption pipeline, install the selected
platform's full inference dependencies into that same environment. The
`general runtime requirements <../requirements.txt>`_ and `hosted model/runtime
guide <hosted_runtime.rst>`_ describe the model stacks and configuration; choose
Torch/CUDA builds for the actual host. Starting the control process does not
load models or prove that they can run.

Install ``age`` separately with the operating system's package manager and
check it with the service environment::

    .venv/bin/python -m tools.hosted_recovery check-age

Configure the service
---------------------

Create a dedicated non-login ``tator`` service user and group using the host's
account-management tools. With the example layout, install private state and
configuration files from the checkout::

    sudo install -d -o tator -g tator -m 0700 /var/lib/tator-hosted
    sudo install -d -o root -g tator -m 0750 /etc/tator
    sudo install -o root -g tator -m 0640 deployment/hosted/hosted.env.example /etc/tator/hosted.env
    sudo install -o root -g tator -m 0640 deployment/hosted/models.example.json /etc/tator/models.json

Use these copy commands for a new installation, not over an existing operator
configuration. Edit the installed files before starting the service:

* ``TATOR_HOSTED_STATE_DIR`` names the private state root.
* ``TATOR_HOSTED_ORIGIN`` is the exact browser HTTPS origin, for example
  ``https://tator.example.net``. It must have no path, query or credentials.
* ``TATOR_HOSTED_MODEL_CONFIG`` names the model-policy JSON file. The shipped
  example has no installed models and example ``cuda:0``/``cuda:1`` assignments;
  replace them with the devices and model definitions available on this host.
* Optional ``TATOR_HOSTED_PROVIDER_PRICING`` names a JSON file of verified
  provider prices and token limits; see `Paid providers`_ below.

The `environment example <../deployment/hosted/hosted.env.example>`_,
`model example <../deployment/hosted/models.example.json>`_ and
`systemd unit <../deployment/hosted/tator-hosted.service>`_ must agree on paths.
The unit sets ``HF_HOME`` and ``TORCH_HOME`` below the state root, hides home
directories, and makes the filesystem read-only except for that root. If moving
state, also update those cache variables and ``ReadWritePaths``. The service
must be able to read its source, environment and trusted model files. Original
datasets do not need service write access.

Bootstrap the first administrator as the service user, from the checkout::

    sudo -u tator /opt/tator/.venv/bin/python -m tools.hosted_admin \
      --state-dir /var/lib/tator-hosted administrator

This creates the initial account and prompts twice for a password of at least
12 characters. It is a first-account bootstrap, not a password-reset command.
Keep passwords out of shell arguments and configuration examples. Later,
invite users through **People** and deliver their invitation links yourself.

HTTPS and startup
-----------------

Production requires HTTPS. The supplied unit binds Uvicorn only to
``127.0.0.1:8010`` and trusts forwarded proxy headers only from loopback. Use a
local HTTPS reverse proxy; set ``TATOR_HOSTED_ORIGIN`` to its exact browser
origin. Keep authentication and application authorization enabled behind it.

For a private Tailscale deployment, with Tailscale already installed and joined
to the intended tailnet, configure Serve and use its reported HTTPS address::

    tailscale serve --bg http://127.0.0.1:8010
    tailscale serve status

Restrict tailnet access to intended users; public Funnel is unnecessary for
this private setup. See `Tailscale Serve
<https://tailscale.com/docs/reference/tailscale-cli/serve>`_. A different local
HTTPS proxy can use the same loopback upstream and exact-origin contract.

Install and start the customized unit. In the loopback health check, replace
``tator.example.net`` with the hostname from ``TATOR_HOSTED_ORIGIN``; the
application rejects other Host headers::

    sudo install -o root -g root -m 0644 deployment/hosted/tator-hosted.service /etc/systemd/system/tator-hosted.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now tator-hosted
    sudo systemctl status tator-hosted
    sudo journalctl -u tator-hosted -n 100 --no-pager
    curl --fail -H 'Host: tator.example.net' http://127.0.0.1:8010/health

Do not enable development reload or additional Uvicorn workers. The state-root
process lock rejects another live control process. ``KillMode=control-group``
stops its model subprocesses during shutdown. Keep the unit's explicit
``--forwarded-allow-ips 127.0.0.1``; a wildcard would trust arbitrary callers.

Health and readiness
--------------------

``GET /health`` is unauthenticated and returns
``{"status":"ok","mode":"hosted"}``. Check it through both loopback and the
configured HTTPS address; a loopback request needs the configured Host header
as shown above. It proves the control application responds, not
that models, encryption or a provider are usable. There is no ``/ready`` route.

After signing in through HTTPS:

* ``GET /api/hosted/session`` checks the authenticated session.
* ``GET /api/hosted/tools/catalog?dataset_id=DATASET_ID`` and
  ``GET /api/hosted/tools/state?dataset_id=DATASET_ID`` expose the caller's tools
  and the chosen dataset's model selections. Substitute an accessible dataset
  ID. The **Tools & models** workspace provides these controls and installs
  registered model copies.
* Administrator-only ``GET /api/hosted/tools/runtime`` reports the runtime
  phase and owners. ``available`` with no owners can mean no model has loaded.

Use a disposable dataset to verify upload, editing, submission, review and
export, then invoke each enabled tool with its actual weights. Check the
service logs and resource use. Run ``check-age`` as the service user and
perform a recovery drill before relying on backups.

Migrate and cut over
--------------------

1. Inventory the source dataset, active annotation overlays, labelmap, splits,
   recipe files, fitted corrections, descriptors and model paths. Resolve
   discrepancies before accepting a migration baseline.
2. Follow the `migration guide <hosted_migration.rst>`_ to import into new hosted
   storage. Compare source/destination hashes, image/object counts, exact
   annotations, captions, class order, split/group metadata and registered
   model/recipe artifacts. Keep sources unchanged.
3. Exercise two annotators and an administrator with actual enabled model
   weights: private tool isolation, idle unload, heavy-job interruption and
   restoration, measured latency and memory use.
4. Test submit, return, approve, lost replies, stale tabs, administrator override
   and restart. Incoming work stays outside canonical analysis, training and
   export; explicit ingestion analysis uses separately captured candidates.
5. Download an encrypted backup, restore into a new private directory and
   continue an assignment after signing in again. Old cookies and leases must
   fail. Reconcile paid usage before enabling providers in the restored copy.
6. Resolve data-preservation, authorization and usability findings before
   switching the live entry point. Choose one writable installation; do not
   edit original and migrated copies concurrently.

Updates and rollback
--------------------

Before changing a live release, ask users to save or retain local drafts and
record active jobs. Stop admitting new work, let wanted jobs finish, and record
any unresolved paid requests; a server restart does not cancel remote Batch
work or establish whether an unknown paid request was charged.

Create a full encrypted backup and record its checksum, source revision and
configuration. Keep the previous release/environment available. Back up the
service unit, environment, external model-policy/pricing files and proxy setup
separately: they are not automatically included in the database/artifact
archive. Follow `backup and restore <hosted_operations.rst>`_ for the drill.

Then stop the whole service before replacing source or dependencies::

    sudo systemctl stop tator-hosted
    sudo systemctl is-active tator-hosted

Deploy the reviewed release, install its requirements (including changes to
``requirements-hosted.txt``), and repeat the interpreter/SQLite and ``pip check``
checks. Retain the same state root for a compatible upgrade. Compare shipped
examples with operator configuration instead of overwriting it. Reload systemd
if its unit changed, start the service, and repeat health, login, a small
annotation/export workflow and the enabled tool checks before reopening work.

Do not point older code at a database changed by an incompatible release. For
rollback, stop the service and restore the matching backup into a new root
with the matching release. Update all service state/cache paths, preserve the
current root, and reconcile work performed since the backup. Restored sessions
and leases are invalidated; paid providers remain paused pending reconciliation.
Archive through the workspace and reclaim only eligible unused objects through
its retention checks; never delete source directories to make room for cutover.

Paid providers
--------------

Provider keys are write-only through the administrator API and stored in the
private database. They are not independently encrypted on disk: the service
account and machine administrator can read them. Protect the live volume and
permissions; full recovery archives are age-encrypted. Configuration responses
never return stored credentials. Local Qwen does not use a provider budget.

Set ``TATOR_HOSTED_PROVIDER_PRICING`` to the **path** of an operator-owned JSON
file keyed by permitted model ID. Each entry requires integer
``input_microdollars_per_million``, ``cached_input_microdollars_per_million``,
``output_microdollars_per_million``, ``input_token_ceiling`` and
``max_output_tokens``. Use current provider rates and actual model limits;
unpriced models cannot dispatch. Remote Batch also requires a separate
``batch`` object with those five verified Batch rate/limit fields. Optional
``image_details`` lists allowed image-detail values; ``original`` must be
explicitly included to be available.

Reservations serialize against a UTC monthly budget in millionths of USD.
Unknown outcomes retain their reservation; underestimated usage pauses further
paid calls. These are application reservations, not a provider-enforced cap.
Review and reconcile bills in **Paid services**, including terminal Batch
requests whose output collection failed. Restore pauses providers because a
backup may predate external charges.

The gateway reads protected immutable images, disables automatic transport
retries, redirects and environment proxies, and never resends an exact request
with an unknown outcome. Local caption jobs use the metered multistage pipeline;
Remote Batch uses a separate combined caption/QA recipe. Results require human
application. See `provider admission, billing and recovery
<hosted_runtime.rst>`_ for the detailed contracts.
