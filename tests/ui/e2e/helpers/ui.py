import re


def go_to_tab(page, tab_button_selector: str, panel_selector: str) -> None:
    page.click(tab_button_selector)
    # Tabs are always present in the DOM; wait for active class to avoid false positives.
    page.wait_for_function(
        "(selector) => document.querySelector(selector)?.classList.contains('active')",
        arg=tab_button_selector,
        timeout=15000,
    )
    page.wait_for_function(
        "(selector) => document.querySelector(selector)?.classList.contains('active')",
        arg=panel_selector,
        timeout=15000,
    )
    page.wait_for_function(
        "(selector) => getComputedStyle(document.querySelector(selector)).display !== 'none'",
        arg=panel_selector,
        timeout=15000,
    )


def open_datasets_tab(page) -> None:
    go_to_tab(page, "#tabDatasetsButton", "#tabDatasets")
    page.wait_for_selector("#datasetPathInput", timeout=15000)


def extract_transient_session_id(page) -> str:
    summary = page.text_content("#datasetPathSummary") or ""
    match = re.search(r"Transient session:\s*([^\s\u2022]+)", summary)
    if not match:
        raise AssertionError(f"Unable to parse transient session id from: {summary}")
    return match.group(1).strip()


def open_transient_session(page, dataset_path: str) -> str:
    open_datasets_tab(page)
    page.fill("#datasetPathInput", dataset_path)
    page.click("#datasetPathOpenBtn")
    page.wait_for_selector("#datasetPathAnnotateBtn:not([disabled])", timeout=15000)
    return extract_transient_session_id(page)


def ensure_local_mode(page) -> None:
    go_to_tab(page, "#tabLabelingButton", "#tabLabeling")
    page.wait_for_selector("#annotationSourceMode", timeout=15000)
    mode_text = (page.text_content("#annotationSourceMode") or "").lower()
    if "local" in mode_text:
        return
    close_btn = page.locator("#annotationCloseBtn")
    close_btn.wait_for(state="visible", timeout=15000)
    close_btn.click(timeout=5000)
    page.wait_for_function(
        "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('local')",
        timeout=20000,
    )


def open_transient_in_annotation(page, dataset_path: str) -> None:
    open_transient_session(page, dataset_path)
    page.click("#datasetPathAnnotateBtn")
    page.wait_for_function(
        "document.querySelector('#annotationSourceMode')?.textContent?.toLowerCase().includes('transient')",
        timeout=20000,
    )
