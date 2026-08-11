import pytest

from localinferenceapi import _raise_on_labelmap_mismatch, HTTPException


def test_labelmap_mismatch_raises():
    with pytest.raises(HTTPException):
        _raise_on_labelmap_mismatch(expected=["car", "person"], actual=["car", "truck"], context="yolo")
