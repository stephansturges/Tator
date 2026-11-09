Admin workbench
===============

The admin workbench gives administrators the complete original Tator tools on
the hosted server: Data Quality Explorer, dataset pruning, data ingestion, class
predictor, YOLO, RF-DETR, SAM3 and Qwen training, detection recipes, SAM3 recipe
mining, the vocabulary explorer, dataset management and model settings. Every
tab keeps its full controls, including starting and stopping runs. Annotators
never see it.

The workbench is the original single-user backend running as a second service
beside the hosted service. It keeps its own datasets, caches, runs and trained
models. The hosted dataset stays the only source of truth for confirmed
annotations. Label edits made in the workbench reach a hosted dataset only when
an administrator reviews and applies them.

Opening the workbench
---------------------

When the server has a workbench, the administrator navigation opens it:

* **Data Quality Explorer**, **Dataset pruning** and **Data ingestion** open the
  matching workbench tab.
* **Train models** entries open the matching training tab. SALAD training
  opens Data ingestion, where SALAD heads are trained.
* **More tools** lists detection recipes, SAM3 recipe mining, the SAM3
  vocabulary explorer, workbench annotation, workbench datasets and model
  settings.
* **Server jobs** opens the hosted analysis launcher, which runs analysis on
  frozen hosted snapshots as before.

Each entry carries the dataset selected in the admin workspace. The workbench
opens that dataset's newest workbench copy in the requested tab, switching away
from any other dataset that was open; the open one is saved first. Only a saved
Data Quality Explorer review that is on screen is kept, and the bar then offers
**Open** for the dataset you came from. A reload of the workbench page keeps
what is open. On the first visit for a dataset, the workbench makes the copy
itself, showing progress in the bar, and then opens it. A dataset without
confirmed images has nothing to copy; the bar says so instead of opening an
empty copy, and offers no **Make fresh copy** until images are confirmed.
Arriving never reopens the previously open copy first, and the saved-session
restore offers only that dataset's own copies.

If the newest copy was deleted in the workbench's Dataset Management, the
newest remaining copy opens and the bar says that a newer copy was deleted.
When no copy is left, the bar says so and offers **Make fresh copy**. Deleting
a copy in Dataset Management deletes its unapplied label edits too, so the
delete confirmation warns when the copy still has edits that are not applied.

Tabs with their own dataset list, such as the training tabs, detection
recipes, captions and the ingestion reference, select the open dataset until
you pick another one in that list yourself.

The bar at the top of the workbench returns to the admin workspace. It always
describes the copy that is actually open: open another copy from Dataset
Management and the bar, **Make fresh copy**, **Review edits** and the
**Admin workspace** link follow it to that copy's hosted dataset. When the open
dataset is not a copy of a hosted dataset, the bar says so, disables the hosted
actions and offers **Open** for the dataset you came from. Choosing a dataset
in the **Hosted data** dialog only changes what the dialog lists; **Open here**
next to a listed copy opens that copy in the workbench, saving the open one
first.

**Review edits** counts the edited images of the open copy that can be applied,
and is updated after every label save, not only on arrival. **Make fresh copy**
warns with that count. The address keeps the open tab, so a reload returns to
it. Leaving the workbench releases its annotation lock, so returning later
opens the copy for editing again. Every tab scrolls its own content, and the
Label Images canvas fits the space below the bar. Browser Back returns to the
page you came from. Without a workbench service the navigation shows the hosted
dialogs only.

A copy can still be locked by your own other tab, or by a window that closed
without releasing it (the lock then lapses after about a minute and a half).
Opening it then asks whether to take over editing here. Cancel opens it
read-only, and the bar says read-only and offers **Take over editing**. A copy
that another administrator is editing opens only read-only, or not at all if
you cancel; the bar says which and why, and **Try editing again** reloads the
page to edit once they have closed it.

A copy does not follow later hosted work. When images were edited on the
annotation canvas, newly confirmed or taken out of the confirmed set after the
copy was made, the bar says how many and offers **Update copy**. Updating
makes a fresh copy and opens it. The bar checks for such changes every half
minute and after each apply. Edits that the workbench applied itself do not
count as hosted changes. Apply pending workbench edits before updating;
otherwise they stay in the older copy, which remains reviewable under **Hosted
data**.

Working on a hosted dataset
---------------------------

1. A workbench copy holds every confirmed image and its labels at the moment
   it is made, as one pool: the workbench does not make or keep split
   decisions, and the legacy layout calls that pool train. Training in the
   workbench does not validate on that pool: a YOLO, RF-DETR or SAM3 run on
   a copy keeps the splits set on the hosted dataset (images placed in
   training or validation stay there, test images are left out of the run)
   and holds out about a tenth of the unsplit rest for that run only, keeping
   capture groups together. A Qwen run does not read the hosted splits yet
   and holds out about a tenth of the whole pool. The run log records the
   split; the copy's files are not changed. Copies made before this release
   do not carry the hosted splits; make a fresh copy to use them. Images are hard
   links to the hosted image store, so a copy uses almost no extra disk space.
   Its name is the dataset name followed by "hosted copy" and the time.
   Image and label files are named after the original file name followed by
   ``__`` and the hosted image ID, for example ``beach_0412__image_3f2a….jpg``,
   so names stay unique and readable. Confirmed images without objects have an
   empty label file and count as labeled in the workbench's progress.
   Opening a tool from the admin workspace makes the first copy. **Make fresh copy** in the bar, or
   **Make workbench copy** under **Hosted data**, makes a newer one that
   includes images confirmed since.
2. Run quality analysis, pruning reviews, training or annotation as in the
   local workbench.
3. To bring label edits back, select **Review edits** in the bar, or open
   **Hosted data** and select **Review changes** for a copy. **Hosted data**
   lists every copy with the edits still waiting in it, so edits left in an
   older copy stay visible, and marks copies deleted in the workbench. The
   review compares the workbench labels with the version each image had when
   it was copied. It counts relabeled, moved or resized, added and deleted
   objects per image, and flags images that someone has open on the
   annotation canvas, which apply only once they are closed.
4. Select **Apply**. Only the edits shown in that review are applied. An image
   edited again in the workbench after the review is skipped and reported, so
   review again to include it. Each applied image is saved as a new confirmed
   version by the administrator. Earlier versions stay in the image history. Objects that
   still match keep their hosted identity, so object history and review
   records stay connected.

An image is not applied, and the review says why, when any of these is true:

* it was edited, returned or removed in the hosted dataset after the copy.
  Make a new copy to continue with it;
* its labels use a retired class. Hosted versions keep such objects, so new
  copies contain them too: relabel or delete those objects in the workbench,
  save, and review again;
* another person holds the image open for editing. Apply again once it is
  closed;
* a polygon object was moved or resized, because the workbench edits boxes
  only. Edit polygons on the hosted canvas.

Failures during the apply itself are reported in words. Images left with no objects are
applied only after you tick the confirmation. They are then saved as
background images. Applying the same copy again only applies edits made since
the last apply.

Only box labels come back. Captions, text labels, new images, pruning results
and trained models stay in the workbench. A pruning review can be exported
from the workbench as before. Bring new images into the hosted dataset through
**Incoming**.

Limits
------

* Workbench runs use the workbench service's GPU and disk. They do not appear
  in the hosted job list, and annotators are unaffected.
* Analysis caches, embeddings, runs and trained models accumulate in the
  workbench storage root. Delete unneeded datasets and runs from the workbench
  **Datasets** tab and the training tabs.
* A copy is a frozen view. Later hosted approvals do not appear in it.

Operating the service
---------------------

The workbench is optional. Enable it with two settings in the hosted
environment file:

``TATOR_HOSTED_WORKBENCH_URL``
   The workbench service address: ``unix:`` and the absolute path of its
   socket, for example ``unix:/run/tator-workbench/api.sock``. A loopback
   address such as ``http://127.0.0.1:8020`` is accepted for development only.
``TATOR_HOSTED_WORKBENCH_ROOT``
   The workbench storage root. It must exist, contain no symlinks, and be on the
   same filesystem as the hosted state directory, because copies hard-link
   images from the hosted image store. The hosted service needs write access
   to it.

Run the workbench with the
`example unit <../deployment/hosted/tator-workbench.service>`_. It starts
``uvicorn app:app`` from the same release and Python environment as the hosted
service, with the storage root as working directory. ``DATASET_ROOT`` and
``DATASET_LINK_ROOTS`` must point inside that root. The legacy backend has no
sign-in of its own, so serve it on a Unix socket in a directory only the
service user can open. Any local process could use a loopback TCP port, and
a browser on the server could reach it too. The hosted service admits enabled
administrators, requires same-origin requests for changes, and strips hosted
cookies and CSRF headers before forwarding. It forwards paths exactly, keeps
legacy redirects under ``/workbench/api`` and never stores legacy cookies.
Uploads to the workbench may be up to 64 GiB.

Both services run as the same operating-system user, so the workbench is
trusted at that level. Images in a copy are read-only links, and the legacy
writers never modify them in place, but a process running as that user could.
The workbench pages also run on the hosted origin with administrator
authority. Annotator-supplied text, such as class names and captions, is
escaped by the legacy pages, but a missed escape there would act with
administrator rights. Serving the workbench from a separate origin is the
stronger isolation if annotators are not fully trusted.

To reuse model weights that hosted analysis already has, set the legacy model
variables in the workbench unit, for example ``DINOV3_TORCH_MODEL_ROOT``,
``CRADIO_TORCH_MODEL_ROOT``, ``SAM3_CHECKPOINT_PATH`` and ``SAM3_BPE_PATH``.
Pin the workbench to one GPU with ``CUDA_VISIBLE_DEVICES``.

Restart the workbench service after each release update, together with the
hosted service. If it is stopped, workbench pages report that the service is
unavailable and the hosted workspace keeps working. Making a copy is then
refused at once, before anything is frozen or linked, and the bar offers
**Reload** to use once the service runs again.

Related guides are listed in the `documentation index <index.rst>`_.
