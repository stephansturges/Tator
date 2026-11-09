import json
from pathlib import Path

from services.datasets import _yolo_resolve_split_paths_impl
from services.detectors import _yolo_write_data_yaml_impl
from utils.run_split import choose_run_split, split_unsplit_pool


def _pool():
    items = [{"id": f"single{i:03d}", "groups": [], "classes": {"car"}} for i in range(60)]
    items += [{"id": f"frame{i:03d}", "groups": [("video_id", "flight")], "classes": {"car"}} for i in range(20)]
    return items + [{"id": "odd", "groups": [], "classes": {"rare"}}]


def test_run_split_holds_out_whole_capture_groups_and_keeps_every_class_in_training():
    items = _pool()
    held, summary = split_unsplit_pool(items, "ds")
    frames = {held[index] for index, item in enumerate(items) if item["id"].startswith("frame")}
    assert len(frames) == 1 and not held[-1] and 0 < sum(held) < len(items)
    assert summary["validation_images"] == sum(held) and summary["validation_share"] == 0.1
    assert split_unsplit_pool(items, "ds") == (held, summary)
    # Declared splits are kept; unsplit members follow their group's split.
    splits, _ = choose_run_split([{**items[60], "split": "val"}] + [{**item, "split": "unsplit"} for item in items[61:80]], "ds")
    assert set(splits) == {"val"}


def test_yolo_training_validates_on_held_out_images_when_the_dataset_has_none(tmp_path):
    dataset = tmp_path / "hosted_copy"
    for sub in ("images", "labels", "annotations"):
        (dataset / "train" / sub).mkdir(parents=True)
    for item in _pool():
        stem = f"{item['id']}__img_{item['id']}"
        (dataset / "train" / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (dataset / "train" / "labels" / f"{stem}.txt").write_text(f"{int('rare' in item['classes'])} 0.5 0.5 0.1 0.1\n")
        body = {"capture_metadata": {"video_id": "flight"}} if item["groups"] else {}
        (dataset / "train" / "annotations" / f"img_{item['id']}.json").write_text(json.dumps({"annotations": body}))
    before = sorted((path, path.read_bytes() if path.is_file() else b"") for path in dataset.rglob("*"))
    run = tmp_path / "run"
    run.mkdir()
    recorded = []
    data = json.loads(_yolo_write_data_yaml_impl(
        run, dataset, "split", None, resolve_split_paths_fn=_yolo_resolve_split_paths_impl,
        yolo_load_labelmap_fn=lambda _path: [], yaml_dump_fn=json.dumps, copy_file_fn=lambda _a, _b: None,
        run_split_key="ds", metadata_roots=[dataset], on_run_split=recorded.append).read_text())
    train = {Path(line).stem for line in Path(data["train"]).read_text().split()}
    val = {Path(line).stem for line in Path(data["val"]).read_text().split()}
    assert val and not train & val and len(train | val) == 81
    frames = {stem for stem in train | val if stem.startswith("frame")}
    assert frames <= train or frames <= val
    assert recorded[0]["validation_images"] == len(val) and json.loads((run / "run_split.json").read_text()) == recorded[0]
    assert sorted((path, path.read_bytes() if path.is_file() else b"") for path in dataset.rglob("*")) == before


def test_workbench_training_keeps_splits_set_on_the_hosted_dataset(tmp_path):
    # A workbench copy pools every image; sidecars carry the hosted split.
    dataset = tmp_path / "hosted_copy"
    for sub in ("images", "labels", "annotations"):
        (dataset / "train" / sub).mkdir(parents=True)
    declared = {"single000": "val", "single001": "val", "single002": "test", "single003": "train"}
    for item in _pool():
        stem = f"{item['id']}__img_{item['id']}"
        (dataset / "train" / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (dataset / "train" / "labels" / f"{stem}.txt").write_text(f"{int('rare' in item['classes'])} 0.5 0.5 0.1 0.1\n")
        sidecar = {"annotations": {}, **({"hosted_split": declared[item["id"]]} if item["id"] in declared else {})}
        (dataset / "train" / "annotations" / f"img_{item['id']}.json").write_text(json.dumps(sidecar))
    run = tmp_path / "run"
    run.mkdir()
    recorded = []
    data = json.loads(_yolo_write_data_yaml_impl(
        run, dataset, "split", None, resolve_split_paths_fn=_yolo_resolve_split_paths_impl,
        yolo_load_labelmap_fn=lambda _path: [], yaml_dump_fn=json.dumps, copy_file_fn=lambda _a, _b: None,
        run_split_key="ds", metadata_roots=[dataset], on_run_split=recorded.append).read_text())
    train = {Path(line).stem.split("__")[0] for line in Path(data["train"]).read_text().split()}
    val = {Path(line).stem.split("__")[0] for line in Path(data["val"]).read_text().split()}
    assert {"single000", "single001"} <= val and "single003" in train and "single002" not in train | val
    assert not train & val and len(train | val) == 80 and len(val) > 2  # the unsplit rest is split too
    assert recorded[0]["declared_images"] == 4 and recorded[0]["test_images"] == 1
