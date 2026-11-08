"""Explicit capture grouping shared by split changes and training admission."""

from utils.capture_metadata import CAPTURE_GROUP_FIELDS
from utils.run_split import annotation_capture_groups

SPLITS = ('train', 'val', 'test', 'unsplit')
GROUP_FIELDS = CAPTURE_GROUP_FIELDS


def capture_groups(annotations):
    # One reading of capture identities for hosted splits and every training run.
    return annotation_capture_groups(annotations)
