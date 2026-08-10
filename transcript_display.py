"""Reconstruct a faithful display transcript from Retell's transcript_object.

Retell's ASR segments audio by timing, so when a caller speaks over the agent
mid-sentence the sentence is split into fragments interleaved as fake "turns".
The dashboard then shows a jumbled call that sounded fine in the audio.

This module rebuilds turns using word timestamps. It may only merge, annotate,
and mark — never add, drop, reword, or reattribute a word. That contract is
machine-checked on every call by verify_fidelity(); build_display() returns
None (dashboard falls back to the raw transcript) if the check ever fails.

Rules:
1. Same-speaker utterances separated by <= GAP_SECONDS are one continuous
   speech stream the ASR split — merge them into one turn.
2. A short other-speaker fragment (<= OVERLAP_MAX_WORDS words, or inaudible)
   that temporally overlaps a continuous stream is simultaneous speech, not a
   turn — attach it to the merged turn as an overlap note, in order.
3. Longer or clearly sequential other-speaker utterances are real turns and
   are never folded in.
"""
import re

GAP_SECONDS = 1.0        # max gap between same-speaker utterances to merge
OVERLAP_MAX_WORDS = 4    # other-speaker fragments this short can be overlaps
OVERLAP_SLACK = 0.4      # seconds of tolerance when testing temporal overlap


def _span(utterance):
    words = utterance.get("words") or []
    if not words:
        return (None, None)
    return (words[0].get("start"), words[-1].get("end"))


def _smooth(utterances):
    turns = []
    i = 0
    while i < len(utterances):
        u = utterances[i]
        start, end = _span(u)
        cur = {
            "role": u.get("role"),
            "parts": [(u.get("content") or "").strip()],
            "overlaps": [],
            "end": end,
        }
        j = i + 1
        while j < len(utterances):
            nxt = utterances[j]
            n_start, n_end = _span(nxt)
            if nxt.get("role") == cur["role"]:
                if (
                    n_start is not None
                    and cur["end"] is not None
                    and n_start - cur["end"] <= GAP_SECONDS
                ):
                    cur["parts"].append((nxt.get("content") or "").strip())
                    cur["end"] = n_end or cur["end"]
                    j += 1
                    continue
                break
            content = nxt.get("content") or ""
            inaudible = "(inaudible" in content
            short = len(content.split()) <= OVERLAP_MAX_WORDS or inaudible
            has_close_continuation = False
            if j + 1 < len(utterances) and utterances[j + 1].get("role") == cur["role"]:
                c_start, _ = _span(utterances[j + 1])
                has_close_continuation = (
                    c_start is not None
                    and cur["end"] is not None
                    and c_start - cur["end"] <= GAP_SECONDS + OVERLAP_SLACK
                )
            # fragments without timestamps (some inaudible blips): trust the
            # continuation test alone rather than dropping the fold
            overlaps_in_time = (
                n_start is None
                or cur["end"] is None
                or n_start <= cur["end"] + OVERLAP_SLACK
            )
            if short and has_close_continuation and overlaps_in_time:
                cur["overlaps"].append(
                    {"role": nxt.get("role"), "text": content.strip(), "inaudible": inaudible}
                )
                j += 1
                continue
            break
        turns.append(cur)
        i = j
    return turns


def _norm_words(text):
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.split(" ") if text else []


def verify_fidelity(utterances, turns):
    """Every word each speaker said must appear in the smoothed structure,
    attributed to the same speaker, in the same order — nothing added."""
    source = {}
    for u in utterances:
        source.setdefault(u.get("role"), []).extend(_norm_words(u.get("content")))
    output = {}
    for t in turns:
        for part in t["parts"]:
            output.setdefault(t["role"], []).extend(_norm_words(part))
        for o in t["overlaps"]:
            output.setdefault(o["role"], []).extend(_norm_words(o["text"]))
    return source == output


def build_display(call):
    """Return the display-transcript structure for a Retell call dict, or None
    when there is nothing to build or the fidelity check fails (callers should
    then fall back to the raw transcript)."""
    utterances = call.get("transcript_object") or []
    if not utterances:
        return None
    try:
        turns = _smooth(utterances)
        if not verify_fidelity(utterances, turns):
            return None
        display = []
        for t in turns:
            who = "agent" if t["role"] == "agent" else "caller"
            notes = [
                {
                    "who": "agent" if o["role"] == "agent" else "caller",
                    "text": o["text"],
                    "inaudible": o["inaudible"],
                }
                for o in t["overlaps"]
            ]
            display.append(
                {"who": who, "text": " ".join(p for p in t["parts"] if p), "notes": notes}
            )
        return display
    except Exception:
        return None
