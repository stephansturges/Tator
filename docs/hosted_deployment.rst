Hosted deployment and cutover
============================

Deployment status
-----------------

The hosted implementation is being qualified locally. These files prepare a
Linux deployment; they do not establish that a GPU server has been deployed.
The dual-A6000 host details, real model tests, WALDO40 migration and final
cutover remain pending. Existing dataset directories and services must stay
intact until their replacements pass the migration and acceptance checks.

Runtime layout
--------------

Use one control process, a private local-disk state directory, and supervised
model subprocesses. The control process uses Python 3.11–3.13 and SQLite
3.51.3 or later. Check the SQLite library used by that exact Python interpreter;
the system ``sqlite3`` executable may use a different library. Do not place
the live database on NFS, SMB or a synchronized folder.

Install the repository's existing Linux/GPU dependencies in the service's
virtual environment, followed by ``requirements-hosted.txt``. The hosted-only
requirements cover the control process; they do not install the model stacks.
They include Pydantic 2, HTTPX and the NumPy/SciPy routines used to review
retained analysis results. A control-process startup does not load Torch,
Transformers or the legacy API application. Paid caption orchestration and
native model execution still require the repository's full runtime dependencies.
Install ``age`` separately for encrypted recovery downloads. Register existing
weights in the trusted model configuration and install them through Tools &
models. Their registered copies are included in recovery backups.

The provided systemd unit assumes ``/opt/tator`` for source and its virtual
environment, ``/var/lib/tator-hosted`` for writable state, and
``/etc/tator/hosted.env`` for configuration. Create a dedicated ``tator`` user
and a state directory owned by that user with mode 0700. Copy and customize
``deployment/hosted/hosted.env.example`` and ``models.example.json``. The empty
model registry deliberately makes no claim about installed server weights.
See ``hosted_runtime.rst`` for the exact model definitions and class mapping.

The unit's read-only filesystem policy allows model sources under
``/srv/tator-models``. Its cache, generated jobs and installed model copies
must remain under the writable state directory. Home directories are hidden
from the service. Adjust trusted source paths when preparing the server;
never widen write permissions to original datasets to make an analysis work.

Bootstrap the first administrator from the service environment with
``python -m tools.hosted_admin --state-dir /var/lib/tator-hosted administrator``. That command prompts for the password. Do not
put passwords in shell arguments, environment examples or service files.
Subsequent accounts are invited through People. Send invitation links yourself.

Private HTTPS
-------------

Bind Uvicorn only to ``127.0.0.1:8010``. After setting the exact HTTPS origin,
configure Tailscale Serve on the target server::

    tailscale serve --bg http://127.0.0.1:8010
    tailscale serve status

Use the HTTPS address reported by Serve as ``TATOR_HOSTED_ORIGIN``. Restrict
tailnet access to the intended users. Serve provides tailnet access; do not
enable public Funnel for this deployment. The application still requires its
own username/password and checks role, dataset membership, editing ownership,
CSRF and installation identity for every protected operation.

The service trusts forwarded proxy information only from loopback. Do not set
``--forwarded-allow-ips '*'``. Do not start additional Uvicorn workers or enable
development reload. The state-directory process lock rejects a second control
process, and systemd stops the whole worker process group on service shutdown.

The proxy commands and forwarding policy follow `Tailscale Serve
<https://tailscale.com/docs/reference/tailscale-cli/serve>`_ and `Uvicorn deployment
<https://www.uvicorn.org/deployment/>`_. Recheck the installed versions before
cutover. No proxy changes have been applied by the local implementation work.

Acceptance before changing the live entry point
----------------------------------------------

1. Inventory the existing WALDO40 source, active annotation overlays, labelmap,
   splits, recipe files, fitted corrections, SALAD descriptors and model paths.
   Resolve discrepancies before creating a trusted migration baseline.
2. Migrate into new hosted storage. Compare source and destination media hashes,
   image/object counts, exact annotations, captions, stable class mapping,
   split/group metadata and every registered model/recipe artifact. Keep the
   source untouched. See ``hosted_migration.rst``.
3. Run two annotators and an administrator against actual GPU weights. Check all
   first-pane tools, isolation, idle unload, a heavy analysis interruption, and
   automatic restoration. Check real latency and memory use; CPU doubles do
   not qualify GPU model behavior.
4. Submit, return and approve individual images; test a lost connection, stale
   tab, administrator override and restart during active work. Confirm that
   Incoming is absent from every analysis, training input and dataset export.
5. Create an encrypted recovery backup, download it to another machine, restore
   into a fresh private directory and continue an existing assignment. Old
   cookies and editor leases must fail. Reconcile paid usage before enabling
   provider calls on the restored installation.
6. Complete independent data-preservation, authorization, inference lifecycle
   and first-time-user reviews. Resolve findings before switching the live URL.

Recovery and rollback
---------------------

Archive images or datasets through the administrator workspace; this preserves
media and annotation history. Physical reclamation is separate and must obey
the retention and reference checks in ``hosted_operations.rst``. Never delete
source directories to reclaim space during cutover.

A dataset ZIP is an interchange export. An encrypted server backup includes
accounts, work in progress, confirmed versions and registered artifacts. Keep
the private age identity on a separate computer, not on the server. The browser
accepts only the public encryption recipient. Test decryption and continuation
of work before treating a downloaded file as a usable recovery copy.

To roll back an application release, stop the whole service before changing its
source. Restore into a new state directory when a schema is incompatible; do
not overwrite the current database or reuse old browser sessions. Preserve
both states while reconciling any work performed after the backup. Do not
resume editing the original dataset and hosted copy concurrently.

Paid service secrets
-------------------

Provider keys are write-only through the administrator API and stored in the
private server database. They are not independently encrypted on disk: the
service account and machine administrator can read them. Full recovery copies
are age-encrypted. Protect the live volume and its access permissions accordingly.
Browser error responses omit submitted values, and configuration responses never
return a stored credential. Local Qwen does not consume a paid-provider budget.

Reservations use integer millionths of USD and serialize against a UTC monthly
budget. Unknown outcomes retain their reserved cost. An observed underestimate
pauses further paid use rather than hiding the excess. Restoration disables paid
providers because a backup can predate charges on the provider's bill. These
controls include a fixed-endpoint Responses API transport, tested with a mocked
provider only. They do not establish live provider execution acceptance.

Set ``TATOR_HOSTED_PROVIDER_PRICING`` to an operator-owned JSON object keyed by
permitted model ID. Each entry needs integer ``input_microdollars_per_million``,
``cached_input_microdollars_per_million``, ``output_microdollars_per_million``,
``input_token_ceiling`` and ``max_output_tokens``. Populate these from the current
provider rates and actual model context limit before enabling calls; there are
no baked-in price assumptions. An unpriced model cannot dispatch. The gateway
reserves the configured full input ceiling plus maximum output cost, then uses
reported token usage to calculate estimated spend. Reconcile it against provider
billing. These are application reservations, not a provider-enforced hard cap.
For asynchronous Batch jobs, add a ``batch`` object under that same model with
its explicitly verified Batch rates and input/output limits using those five
field names. A model without this separate record is unavailable in Remote
Batch jobs. Optional ``image_details`` lists permitted image-detail values;
``original`` is unavailable unless the operator explicitly includes it.

The gateway takes the image from protected immutable storage, not a client URL
or filesystem path. It disables provider storage, automatic transport retries,
redirects and environment proxies. A lost reply retains its reservation and
requires reconciliation; retrying the same operation never sends it twice.
Late results are billed and retained but cannot modify an image after ownership
or version changes. Billing and result retention commit in one transaction.

Protocol references: `Responses API image inputs
<https://developers.openai.com/api/docs/guides/images-vision>`_ and `text output
<https://developers.openai.com/api/docs/guides/text>`_. Hosted caption jobs now
reuse the native multistage caption/QA pipeline through the metered gateway;
Remote Batch jobs use the separate native combined caption-and-QA recipe.
Both retain results for explicit human approval. Local mocked-transport and
browser checks cover these integrations; actual provider model behavior, billing
and limits must be qualified separately before operational acceptance. See
``hosted_runtime.rst`` for their distinct recovery and admission contracts.
