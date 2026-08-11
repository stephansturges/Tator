from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "ybat-master" / "tator.html").read_text(encoding="utf-8")
JS = (ROOT / "ybat-master" / "ybat.js").read_text(encoding="utf-8")


def test_fast_sam1_and_preload_are_interactive_defaults():
    assert '<option value="sam1">SAM 1 - Fastest interactive</option>' in HTML
    assert '<option value="sam3" selected>SAM 3 - Default / advanced</option>' in HTML
    assert 'id="samPreload" name="samPreload" checked' in HTML
    assert 'let samVariant = "sam3";' in JS
    assert "let samPreloadEnabled = true;" in JS
    assert "SAM 3 uses accelerated BF16 MLX automatically on Apple Silicon" in HTML


def test_sam3_runtime_notice_updates_the_model_tooltip():
    assert "result?.runtime_notice" in JS
    assert "samVariantSelect.dataset.defaultTitle" in JS
    assert "SAM3_BACKEND=torch" in HTML or "actual backend" in HTML


def test_sam3_runtime_selector_is_benchmark_guided_and_checkpoint_scoped():
    assert 'id="sam3Runtime"' in HTML
    assert '<option value="mlx-bf16">MLX BF16 - Recommended</option>' in HTML
    assert '<option value="mlx-mxfp4">MLX MXFP4 - Smallest good</option>' in HTML
    assert '<option value="torch">Torch - Full-precision reference</option>' in HTML
    assert 'value="mlx-6bit"' not in HTML
    assert 'value="mlx-nvfp4"' not in HTML
    assert "function selectedSamPredictorVariant" in JS
    assert "sam3@${sam3Runtime" in JS
    assert "variantForRequest = selectedSamPredictorVariant()" in JS
    assert "sam3_runtime_options" in JS

    tweak_start = JS.index("async function ensureSamReadyForMagicTweak")
    tweak_end = JS.index("async function runMagicTweakForBbox", tweak_start)
    tweak_body = JS[tweak_start:tweak_end]
    assert "const variantForRequest = selectedSamPredictorVariant();" in tweak_body


def test_same_image_preload_is_single_flight_across_variants():
    prepare_start = JS.index("async function prepareSamForCurrentImage")
    prepare_end = JS.index("async function ensureImageRecordReady", prepare_start)
    prepare_body = JS[prepare_start:prepare_end]
    assert "isSamPreloadPendingForImage(targetName)" in prepare_body
    assert "getSamPreloadPendingVariantForImage(targetName)" in prepare_body
    assert "await waitForSamPreloadIfActive(targetName, pendingVariant)" in prepare_body
    assert "force: false" in prepare_body
    assert "samPreloadCurrentVariant = samVariant" not in JS


def test_raw_image_token_is_reused_across_sam_variants():
    assert JS.count("getAnySamTokenForImage(imageName, variantSnapshot)") >= 1
    assert "getAnySamTokenForImage(imageSnapshot.name, variantSnapshot)" in JS
    assert "forgetSamTokensForImage(imageSnapshot.name)" in JS
