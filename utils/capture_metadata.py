"""Portable explicit capture provenance shared by hosted and native consumers."""
import math


# Preserve the native capture-hint priority. The final aliases also occur in
# migration mappings, source-independence records and pruning export manifests.
CAPTURE_GROUP_FIELDS = (
    "capture_group_id", "sequence_id", "video_id", "flight_id",
    "video_group", "source_group", "duplicate_group",
)
CAPTURE_METADATA_FIELDS = CAPTURE_GROUP_FIELDS + (
    "camera_id", "frame_index", "capture_timestamp",
    "capture_perceptual_hash", "capture_perceptual_image_sha256",
)


def portable_capture_metadata(row):
    result = {}
    for field in CAPTURE_METADATA_FIELDS:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        result[field] = value
    return result
