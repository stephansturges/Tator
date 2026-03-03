import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return text.strip("_") or "artifact"


def _artifact_dir() -> Path:
    path = Path(os.environ.get("UI_E2E_ARTIFACT_DIR", "tmp/ui_e2e/usability"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_soft_artifact(page, name: str, payload: dict | None = None) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{_slugify(name)}"
    root = _artifact_dir()
    screenshot_path = root / f"{stem}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    if payload is not None:
        report_path = root / f"{stem}.json"
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def assert_visible_in_viewport(page, selector: str, padding: float = 0.0) -> None:
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no bounding box"
    assert box["width"] > 0 and box["height"] > 0, f"{selector} has zero size"
    viewport = page.viewport_size or page.evaluate("({ width: window.innerWidth, height: window.innerHeight })")
    assert box["x"] + box["width"] >= -padding, f"{selector} is off-screen left"
    assert box["y"] + box["height"] >= -padding, f"{selector} is off-screen top"
    assert box["x"] <= viewport["width"] + padding, f"{selector} is off-screen right"
    assert box["y"] <= viewport["height"] + padding, f"{selector} is off-screen bottom"


def assert_text_not_clipped(page, selector: str, tolerance_px: int = 1) -> dict:
    metrics = page.eval_on_selector(
        selector,
        """(el) => {
            const text = (el.textContent || "").trim();
            const style = getComputedStyle(el);
            return {
                text,
                display: style.display,
                visibility: style.visibility,
                opacity: Number(style.opacity || "1"),
                clientWidth: el.clientWidth,
                scrollWidth: el.scrollWidth,
                clientHeight: el.clientHeight,
                scrollHeight: el.scrollHeight,
            };
        }""",
    )
    assert metrics["display"] != "none", f"{selector} is display:none"
    assert metrics["visibility"] != "hidden", f"{selector} is visibility:hidden"
    assert metrics["opacity"] > 0, f"{selector} has zero opacity"
    assert metrics["text"], f"{selector} has empty text"
    assert metrics["scrollWidth"] <= metrics["clientWidth"] + tolerance_px, (
        f"{selector} text is horizontally clipped "
        f"(scrollWidth={metrics['scrollWidth']} clientWidth={metrics['clientWidth']})"
    )
    assert metrics["scrollHeight"] <= metrics["clientHeight"] + tolerance_px, (
        f"{selector} text is vertically clipped "
        f"(scrollHeight={metrics['scrollHeight']} clientHeight={metrics['clientHeight']})"
    )
    return metrics


def assert_no_horizontal_overflow(page, selector: str, tolerance_px: int = 2) -> dict:
    metrics = page.eval_on_selector(
        selector,
        "(el) => ({ clientWidth: el.clientWidth, scrollWidth: el.scrollWidth })",
    )
    assert metrics["scrollWidth"] <= metrics["clientWidth"] + tolerance_px, (
        f"{selector} overflows horizontally "
        f"(scrollWidth={metrics['scrollWidth']} clientWidth={metrics['clientWidth']})"
    )
    return metrics


def assert_min_font_size(page, selector: str, min_px: float = 12.0) -> float:
    font_size = float(
        page.eval_on_selector(
            selector,
            "(el) => Number.parseFloat(getComputedStyle(el).fontSize || '0') || 0",
        )
    )
    assert font_size >= min_px, f"{selector} font-size too small ({font_size}px < {min_px}px)"
    return font_size


def assert_min_contrast(page, selector: str, threshold: float = 4.5) -> dict:
    report = page.eval_on_selector(
        selector,
        """(el) => {
            const parseColor = (raw) => {
                const value = String(raw || "").trim();
                if (!value) return null;
                if (value.startsWith("#")) {
                    const hex = value.slice(1);
                    if (hex.length === 3) {
                        const r = parseInt(hex[0] + hex[0], 16);
                        const g = parseInt(hex[1] + hex[1], 16);
                        const b = parseInt(hex[2] + hex[2], 16);
                        return [r, g, b, 1];
                    }
                    if (hex.length === 6) {
                        const r = parseInt(hex.slice(0, 2), 16);
                        const g = parseInt(hex.slice(2, 4), 16);
                        const b = parseInt(hex.slice(4, 6), 16);
                        return [r, g, b, 1];
                    }
                    return null;
                }
                const m = value.match(/^rgba?\\(([^)]+)\\)$/i);
                if (!m) return null;
                const parts = m[1].split(",").map((p) => Number.parseFloat(p.trim()));
                if (parts.length < 3) return null;
                return [parts[0], parts[1], parts[2], Number.isFinite(parts[3]) ? parts[3] : 1];
            };
            const srgbToLin = (x) => {
                const v = x / 255;
                return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
            };
            const luminance = (rgb) => 0.2126 * srgbToLin(rgb[0]) + 0.7152 * srgbToLin(rgb[1]) + 0.0722 * srgbToLin(rgb[2]);
            const contrast = (a, b) => {
                const l1 = luminance(a);
                const l2 = luminance(b);
                const light = Math.max(l1, l2);
                const dark = Math.min(l1, l2);
                return (light + 0.05) / (dark + 0.05);
            };
            const style = getComputedStyle(el);
            const fg = parseColor(style.color) || [0, 0, 0, 1];
            let node = el;
            let bg = null;
            while (node) {
                const nStyle = getComputedStyle(node);
                const parsed = parseColor(nStyle.backgroundColor);
                if (parsed && parsed[3] > 0.01) {
                    bg = parsed;
                    break;
                }
                node = node.parentElement;
            }
            if (!bg) bg = [255, 255, 255, 1];
            const ratio = contrast(fg, bg);
            return {
                ratio,
                fg: `rgba(${fg[0]}, ${fg[1]}, ${fg[2]}, ${fg[3]})`,
                bg: `rgba(${bg[0]}, ${bg[1]}, ${bg[2]}, ${bg[3]})`,
            };
        }""",
    )
    assert report["ratio"] >= threshold, (
        f"{selector} contrast too low ({report['ratio']:.2f} < {threshold:.2f}); "
        f"fg={report['fg']} bg={report['bg']}"
    )
    return report


def open_tooltip_and_assert_readable(page, selector: str, min_chars: int = 8) -> dict:
    locator = page.locator(selector)
    locator.hover()
    report = page.eval_on_selector(
        selector,
        """(el) => {
            const after = getComputedStyle(el, "::after");
            const before = getComputedStyle(el, "::before");
            const normalize = (content) => {
                const raw = String(content || "").trim();
                if (!raw || raw === "none") return "";
                return raw.replace(/^['\\"]|['\\"]$/g, "");
            };
            return {
                content: normalize(after.content),
                opacity: Number(after.opacity || "0"),
                pointerEvents: after.pointerEvents,
                fontSize: Number.parseFloat(after.fontSize || "0") || 0,
                lineHeight: Number.parseFloat(after.lineHeight || "0") || 0,
                maxWidth: Number.parseFloat(after.maxWidth || "0") || 0,
                beforeOpacity: Number(before.opacity || "0"),
            };
        }""",
    )
    assert report["content"], f"{selector} tooltip content is empty"
    assert len(report["content"]) >= min_chars, (
        f"{selector} tooltip content too short ({len(report['content'])} chars)"
    )
    assert report["opacity"] >= 0.9, f"{selector} tooltip did not become visible"
    assert report["beforeOpacity"] >= 0.9, f"{selector} tooltip arrow did not become visible"
    assert report["fontSize"] >= 11, f"{selector} tooltip font-size too small ({report['fontSize']}px)"
    assert report["maxWidth"] >= 140, f"{selector} tooltip max-width unexpectedly small ({report['maxWidth']}px)"
    return report
