Hosted storage, backups and recovery
====================================

The administrator's **Export & recovery** workspace has three distinct uses:

* **Dataset export** produces an interchange ZIP of confirmed canonical data.
  It does not contain accounts, assignments or Incoming work.
* **Full recovery backup** captures database state and referenced artifacts in
  an encrypted archive, including work in progress and retained history.
* **Archive/restore** changes image or dataset membership while preserving
  media and annotation history. Physical deletion is a separate operation.

Use the `deployment guide <hosted_deployment.rst>`_ for installation and release
updates, and the `migration guide <hosted_migration.rst>`_ for importing an
existing dataset. Backups cover registered artifacts; they do not automatically
capture every external legacy model or configuration file. The application
and recovery CLI use the schema-aware reference enumerator and restore hook
below by default.

Only server-acknowledged work is in the archive. Unsent edits and recovery
journals stored in a user's browser are not part of a server backup; resolve or
retain them separately before maintenance.

Run the CLI examples from the matching source checkout with the service's
Python environment (for example, ``/opt/tator/.venv/bin/python`` in place of
``python``). They do not load the systemd environment file automatically.

Encrypted full recovery backup
------------------------------

``create_encrypted_backup`` uses SQLite's backup operation, then enumerates
immutable references from that captured database. Both steps and creation of
hard-link pins occur while holding the artifact reference/GC lock. The lock is
released before encryption. Removing a subsequently unreferenced store entry
cannot remove a pinned backup input. Temporary pins are released after success
or failure.

The archive contains a versioned manifest, the consistent SQLite database and
the exact referenced artifacts. File I/O and tar encryption are streamed. The
destination appears only after the encrypted output has completed and been
fsynced. Existing destination files are never overwritten.

Encryption requires an installed ``age`` executable and an explicit public
recipient. There is no plaintext full-backup fallback. Keep the private age
identity on a separate trusted recovery machine; it is not copied into the
archive or submitted to the web application. The capture temporarily contains
plaintext SQLite and hard-link pins inside private
server staging. Normal completion/failure removes that staging; a process crash
can leave private staging for operator inspection and cleanup. Disk encryption
and access controls still apply to the running server and temporary capture.
The command-line flags and streaming convention follow the
`official age documentation <https://github.com/FiloSottile/age#usage>`_.

Before the first backup:

1. Install ``age`` and use ``age-keygen`` on the trusted recovery machine to
   create an identity file. Keep that private file separate from the server
   and backups. In a private directory, create the key once and print its public
   recipient::

       age-keygen -o /secure/recovery/identity.txt
       age-keygen -y /secure/recovery/identity.txt

   Supply only the resulting public ``age1...`` recipient to the browser.
2. Install the service release's `hosted requirements
   <../requirements-hosted.txt>`_ and run the following checks from its checkout
   with the service's Python environment and filesystem permissions::

       python -m tools.hosted_recovery --help
       python -m tools.hosted_recovery check-age

3. In **Export & recovery**, create a full backup using the public recipient,
   wait for completion, and download the encrypted file to separate storage.
   Leave room for the temporary database snapshot and encrypted output.

For operator automation, the equivalent CLI creates a new archive and prints
its result as JSON. Choose a private existing output directory writable by the
operator and a filename that does not already exist::

    python -m tools.hosted_recovery backup \
      --database /var/lib/tator-hosted/hosted.sqlite3 \
      --artifacts /var/lib/tator-hosted/artifacts \
      --output /var/backups/tator/tator-YYYYMMDD.age \
      --recipient AGE_PUBLIC_RECIPIENT

The CLI can capture a consistent running installation using the same reference
lock and SQLite backup operation as the web application. It does not stop the
service or prevent later edits. Do not replace this with a plain copy of a
live ``hosted.sqlite3`` file; committed data can also be in its WAL.

The default ``recovery_artifact_references(connection)`` receives the captured SQLite
connection. It must enumerate all durable media and registered custom-weight,
configuration and other required artifact references, including retained
history. Supported values are ``ArtifactRef``, digest strings, or rows with
``sha256`` and ``size_bytes``. It must not query the live database. A backup that
enumerates media only is not evidence that unregistered external model/config
files are recoverable. Register those files and their references first.
Trusted operators can override the callbacks with ``--references`` or
``--restore-hook`` using an importable ``module:function``. The normal hosted
flow does not require an override.

Record the archive's SHA-256, creation time, source revision and expected
installation. Keep the previous known-good backup until a newer one has passed
a restore drill. Back up the service unit, environment, external model/pricing
JSON and proxy configuration separately; record how to reinstall the matching
code and dependencies. Keep secrets in protected storage. A successful download
alone does not prove recoverability: test decryption and continued work.

Restore into an empty destination
---------------------------------

Stop or isolate the destination application. The restore operation does not
stop services or enter maintenance mode itself. Restore only into an absent or
empty root, on a local filesystem. Use the matching code/dependencies on the
trusted recovery machine where the private identity is available::

    python -m tools.hosted_recovery restore \
      --archive /recovery/tator-YYYYMMDD.age \
      --destination /recovery/tator-restored \
      --identity /secure/recovery/identity.txt

Use the default database filename when restoring for the hosted application.
The optional ``--database-name`` is for package/operator integrations; the web
service expects ``hosted.sqlite3`` under its configured state root.

Decryption, archive member validation, SHA-256 verification and SQLite integrity
checks complete in a private staging root. Traversal paths, symlinks, hard-link
members, duplicate members, unexpected files, missing files and oversized
declared contents are rejected. Limits are configurable with ``RestoreLimits``
for package callers; the defaults cap the manifest at 32 MiB, members at one
million, individual files at 1 TiB and total captured content at 8 TiB.

The required ``restore_hook(connection, manifest)`` runs before publication,
inside a database transaction. The application hook must revoke sessions and
authentication links, clear editor leases, rotate the installation identity and
reconcile interrupted jobs. It preserves assignments, drafts, submissions,
confirmed history and attribution. Old browser operations must fail the new
installation fence before an old receipt can authorize a mutation.
The default hosted hook also validates secondary references for analysis files,
pruning checkpoints, caption evidence, paid-stage inputs, SALAD heads and
registered model packages. A result naming a blob absent from its registered
owner or recovery manifest prevents publication. Paid-provider accounts are
paused after restore until an administrator reconciles spending against the
provider: a recovered billing ledger may predate external charges.

The validated staged root is atomically renamed into the destination. The
default database filename is ``hosted.sqlite3`` and the artifact directory is
``artifacts``. No existing nonempty installation is overwritten. Hook or
decryption failure leaves the destination unpublished. Back up the original
installation and retain original source data until a separately approved cutover.

After a successful restore:

1. Keep the old installation stopped or isolated. Transfer the validated root
   to the chosen host if needed, without copying the private age identity.
   Ensure the restored tree is readable/writable by the service user and the
   state root has mode 0700.
2. Point the service at the new root. For the example systemd unit, update
   ``TATOR_HOSTED_STATE_DIR``, ``ReadWritePaths``, ``HF_HOME`` and ``TORCH_HOME``
   together. Restore external configuration and verify model-source paths.
3. Start one control process, check ``/health``, sign in again and inspect
   assignments, versions, archived membership and registered tools. Old
   sessions and editor leases are intentionally invalid.
4. Continue a small existing assignment, save it, submit/review it and export a
   confirmed image. Verify any enabled model on that host. Inspect interrupted
   jobs and paid calls before resuming them; never create a replacement paid
   request merely because a reply was lost.
5. Reconcile provider spending before enabling paid accounts. Keep the old root
   and backups until any work newer than the recovered snapshot is accounted
   for; do not operate both copies as writable production installations.

Storage reclamation
-------------------

Archive and restore preserve immutable annotation history and its media. The
archive storage view explains those references and the minimum 30-day retention
period; it does not promise that retained images become deletable on day 30.
Archive purge currently refuses while that history remains referenced.

The **Unused storage** section in **Export & recovery** scans stored artifacts
in pages and shows which unreferenced files have passed retention. Preview the
eligible files, inspect their total logical size, then explicitly confirm
deletion. The administrator-only HTTP routes use ``discover_orphans``,
``preview_orphan_purge`` and ``purge_orphans``. The deletion command binds an
exact preview, checks references again and journals per-file outcomes for
interruption recovery. A lost reply leaves **Check cleanup result** available
for the same command. Referenced images and model files remain protected.

Canonical dataset exports
-------------------------

``create_canonical_zip`` is separate from recovery backup. Its caller supplies a
frozen canonical snapshot as explicit ``ExportEntry`` objects: safe archive
paths plus either immutable ``ArtifactRef`` values or version-bound annotation
and labelmap bytes. It never discovers membership by walking a live directory.
The caller enforces administrator authorization and confirmed-only membership.
An ordinary dataset ZIP excludes account state, Incoming work and server
recovery history and cannot replace a full encrypted recovery backup.

Encoded media and immutable working views
-----------------------------------------

All hosted consumers use the original encoded pixel grid. Admission and native
decoding share a 256 MiB input limit and a 64 million decoded-pixel limit, perform
a full decode, and do not resize. EXIF orientation is suppressed before loading
pixels, including TIFF's implicit rotation. Multi-frame inputs use their first
frame as a static image. Source artifacts are never rewritten.

The current working-pixel policy is ``encoded-rgb8-white-v2``. Opaque, single-frame
8-bit RGB or grayscale JPEG, PNG and WebP retain their source bytes when they have
no orientation or color-display transforms. Other inputs receive a metadata-free
RGB PNG: transparency is composited onto white, and unsigned 16-bit samples use
their high byte (divide by 256 and round down), without contrast stretching.
PNG sample depth is checked in the encoded header, because a decoder can expose
16-bit RGB as ordinary RGB. No conversion changes the encoded dimensions.
For a 16-bit PNG with a transparent-color key, the small ``pypng`` dependency
compares the original samples before conversion. Adjacent colors that reduce to
the same 8-bit value are not incorrectly treated as transparent. A missing
decoder returns ``media_decoder_unavailable`` (503); it never substitutes a
different conversion or changes the original file.

``hosted_media_views`` permanently maps each source digest and policy to its
working digest, dimensions and format. Existing ``encoded-rgb-v1`` views remain
immutable alongside v2 views; the coordinate-provenance policy is separate from
this pixel-conversion policy. Both original and working artifacts are
registered for backup, restore validation and orphan-retention protection.
Authenticated original downloads continue to return the original bytes.
Admission verifies that the working representation can be encoded, but registers
its immutable view only when first consumed. Failed label validation or class
version checks leave no permanent media-view references; unadmitted source blobs
remain subject to the ordinary orphan retention policy.

``hosted-snapshot.json`` remains an immutable record of source identities.
Exports and native materializations add ``hosted-media.json`` with the policy,
encoded coordinate space and an ``images`` mapping keyed by hosted image ID.
Each item names the exported relative path, actual working SHA-256 and size,
original source SHA-256 and size, dimensions, format and original orientation.
Native analyses validate the working hash and retain the source hash separately;
cache identity therefore describes the bytes actually consumed. Existing
snapshots and committed export receipts are not rewritten. An export retry
continues to deliver its already retained ZIP.

New analysis and caption work captures its working descriptor before execution.
Materialization verifies and copies those exact bytes; it does not decode them
again under a newer conversion policy. Historical analysis media and thumbnails
use that analysis's retained inputs. A new analysis of an older source snapshot
can capture v2 views without changing the old analysis.

Unfinished local work with an incompatible or unidentified older pixel policy
stops with ``media_policy_upgrade_required``. Retained evidence remains available;
start a new operation to use current views. Compatible, explicitly captured old
bytes can continue unchanged. Paid requests already sent retain their original
inputs and accounting and are never resent merely to upgrade their pixels.

Immutable objects
-----------------

``services.hosted.artifacts.ArtifactStore`` stores objects by their SHA-256
digest under ``blobs/sha256/<first-two-characters>/<digest>``. Media and custom
artifact bytes can share this store. Annotation versions may remain in SQLite.
Public routes must authorize dataset-image membership before reading an object;
knowing its digest does not grant access.

``put_stream`` copies in bounded chunks to a private temporary file, checks any
expected size/hash, flushes and fsyncs it, then publishes using an atomic hard
link that cannot overwrite an existing object. Its directory entry is fsynced
before returning. Existing matching objects are verified rather than replaced.
``put_file`` is an administrative import interface: it reads the source without
following a final symlink and never changes the original file. Remote clients
must not be given arbitrary server-path access through this interface.

Reference creation and removal have one lock order::

    with artifacts.exclusive():
        reference = artifacts.put_stream(upload)
        with store.transaction() as connection:
            # Record this exact reference in SQLite.
            ...

The lock is shared across threads and processes using this store root. Acquire
it before beginning a SQLite write transaction. Garbage collection uses the
same lock and checks references while holding it. The low-level
``delete_unreferenced`` primitive does not implement administrator authorization,
the minimum 30-day recovery period, or purge previews: those policies must be
satisfied by the caller before using it. This is not a user-facing deletion API.

Verification boundaries
-----------------------

The focused tests compose real SQLite snapshots, immutable files, reference
pinning, integrity/path validation, restore hooks and frozen ZIP contents.
Most composition tests explicitly inject a noncryptographic archive transport;
they do not demonstrate encryption. A separate test invokes real ``age`` and
``age-keygen`` when installed and otherwise skips. Production availability checks
fail if age cannot execute. Target-machine backup/download/restore and continued
annotation acceptance remain separate deployment checks.
