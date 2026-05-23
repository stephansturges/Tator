from types import SimpleNamespace

from services.qwen_jobs import _qwen_job_append_metric_impl


def test_qwen_job_metrics_sanitize_nonfinite_values():
    job = SimpleNamespace(metrics=[], updated_at=0.0)

    _qwen_job_append_metric_impl(
        job,
        {
            "loss": float("nan"),
            "nested": {"value": float("inf")},
            "items": [float("-inf"), 1.25],
        },
        max_points=None,
    )

    assert job.metrics == [
        {
            "loss": None,
            "nested": {"value": None},
            "items": [None, 1.25],
        }
    ]
