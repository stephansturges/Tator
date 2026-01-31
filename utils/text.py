from __future__ import annotations

import re


def _agent_clean_plan_text(text: str, max_len: int = 4000) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    for prefix in ("PLAN:", "Plan:", "FINAL:", "Final:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    match = re.search(r"\\d+[\\).:-]", cleaned)
    if match:
        cleaned = cleaned[match.start():].lstrip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    numbered = []
    for line in lines:
        if re.match(r"^\\d+[\\).:-]", line):
            numbered.append(line)
    if not numbered:
        step_parts = re.split(r"(?=\\d+[\\).:-]\\s)", cleaned)
        step_parts = [part.strip() for part in step_parts if part.strip()]
        if step_parts and re.match(r"^\\d+[\\).:-]", step_parts[0]):
            numbered = step_parts
    if numbered:
        cleaned = "\n".join(numbered)
    else:
        cleaned = "\n".join(lines)
        if cleaned:
            cleaned = re.sub(r"^(got it|sure|okay|ok|alright)[,\\s]+", "", cleaned, flags=re.IGNORECASE)
            cleaned = f"1. {cleaned}"
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned
