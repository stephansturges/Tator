import subprocess
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_annotation_diversity_metric_scores_rare_current_classes_higher():
    script = textwrap.dedent(
        """
        const assert = require("assert");
        const api = require("./ybat-master/annotation_diversity.js");

        const classNames = ["common", "rare", "empty"];
        const common = api.computeImageDiversityMetric({
            currentCounts: { common: 1 },
            datasetCounts: { common: 101, rare: 1 },
            classNames,
        });
        const rare = api.computeImageDiversityMetric({
            currentCounts: { rare: 1 },
            datasetCounts: { common: 100, rare: 1 },
            classNames,
        });
        const multi = api.computeImageDiversityMetric({
            currentCounts: { common: 1, rare: 1 },
            datasetCounts: { common: 101, rare: 1 },
            classNames,
        });

        assert(rare.score > common.score, `${rare.score} should beat ${common.score}`);
        assert(multi.score > common.score, `${multi.score} should beat ${common.score}`);
        assert(rare.newClasses.includes("rare"));
        assert(api.formatImageDiversityMetric(rare).startsWith("Image value "));
        """
    )
    _run_node(script)


def test_annotation_diversity_metric_counts_buckets_and_yolo_rows():
    script = textwrap.dedent(
        """
        const assert = require("assert");
        const api = require("./ybat-master/annotation_diversity.js");

        const bucketCounts = api.countBoxesByClassFromBuckets({
            car: [
                { class: "car", x: 1, y: 2, width: 10, height: 5 },
                { class: "car", points: [{ x: 0, y: 0 }, { x: 5, y: 0 }, { x: 3, y: 4 }] },
                { class: "car", points: [{ x: 0, y: 0 }, { x: "bad", y: 0 }, { x: 3, y: null }] },
                { class: "car", x: 0, y: 0, width: 0, height: 1 },
            ],
            truck: [{ class: "truck", x: 0, y: 0, width: 2, height: 2 }],
        });
        assert.deepStrictEqual(bucketCounts, { car: 2, truck: 1 });

        const rowCounts = api.countBoxesByClassFromYoloLines([
            "0 0.5 0.5 0.25 0.25",
            "1 0.1 0.1 0.2 0.1 0.2 0.2",
            "1 0.5 0.5 0 0.2",
            "bad row",
        ], ["car", "truck"]);
        assert.deepStrictEqual(rowCounts, { car: 1, truck: 1 });

        const empty = api.computeImageDiversityMetric({
            currentCounts: {},
            datasetCounts: { car: 10 },
            classNames: ["car"],
        });
        assert.strictEqual(empty.score, 0);
        assert.strictEqual(empty.status, "empty");
        assert(api.formatImageDiversityMetric(empty).includes("no bboxes"));
        """
    )
    _run_node(script)
