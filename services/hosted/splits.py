"""Explicit capture grouping shared by split changes and training admission."""

from utils.capture_metadata import CAPTURE_GROUP_FIELDS, portable_capture_metadata

SPLITS = ('train', 'val', 'test', 'unsplit')
GROUP_FIELDS = CAPTURE_GROUP_FIELDS


def capture_groups(annotations):
    migration = annotations.get('migration', {})
    migration = migration.get('metadata', {}) if isinstance(migration, dict) else {}
    captured = annotations.get('capture_metadata', {})
    metadata = portable_capture_metadata({**(migration if isinstance(migration, dict) else {}),
                                          **(captured if isinstance(captured, dict) else {})})
    return tuple((key, str(metadata[key])) for key in GROUP_FIELDS
                 if key in metadata)
