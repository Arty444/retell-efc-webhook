from transcript_display import build_display, verify_fidelity, _smooth


def utt(role, content, start, end):
    words = []
    if content and start is not None:
        # spread word timings evenly across the span
        toks = content.split()
        step = (end - start) / max(len(toks), 1)
        words = [
            {"word": w, "start": start + i * step, "end": start + (i + 1) * step}
            for i, w in enumerate(toks)
        ]
    return {"role": role, "content": content, "words": words}


def test_clean_alternating_call_is_unchanged():
    call = {
        "transcript_object": [
            utt("agent", "Hi, how can I help?", 0.0, 2.0),
            utt("user", "I want a trial class.", 3.0, 4.5),
            utt("agent", "Great, for you or a child?", 5.5, 7.0),
        ]
    }
    display = build_display(call)
    assert [t["who"] for t in display] == ["agent", "caller", "agent"]
    assert all(t["notes"] == [] for t in display)
    assert display[1]["text"] == "I want a trial class."


def test_split_sentence_with_overlap_is_stitched_and_annotated():
    # agent sentence split by the ASR because the caller spoke over it
    call = {
        "transcript_object": [
            utt("agent", "Just checking in—did", 75.8, 76.9),
            utt("user", "What do you", 76.9, 77.5),
            utt("agent", "you have", 76.9, 77.6),
        ]
    }
    display = build_display(call)
    assert len(display) == 1
    assert display[0]["text"] == "Just checking in—did you have"
    assert display[0]["notes"] == [
        {"who": "caller", "text": "What do you", "inaudible": False}
    ]


def test_long_interruption_stays_a_real_turn():
    call = {
        "transcript_object": [
            utt("agent", "Classes run Monday through", 0.0, 2.0),
            utt("user", "Stop, I just need to cancel my membership please", 2.1, 4.5),
            utt("agent", "Thursday at five", 4.6, 5.5),
        ]
    }
    display = build_display(call)
    assert [t["who"] for t in display] == ["agent", "caller", "agent"]


def test_fidelity_failure_returns_none(monkeypatch):
    import transcript_display

    monkeypatch.setattr(transcript_display, "verify_fidelity", lambda *_: False)
    call = {"transcript_object": [utt("agent", "Hello there", 0.0, 1.0)]}
    assert transcript_display.build_display(call) is None


def test_no_transcript_object_returns_none():
    assert build_display({}) is None
    assert build_display({"transcript_object": []}) is None


def test_verify_fidelity_catches_dropped_word():
    utts = [utt("agent", "one two three", 0.0, 1.0)]
    turns = _smooth(utts)
    turns[0]["parts"] = ["one two"]
    assert not verify_fidelity(utts, turns)


def test_verify_fidelity_catches_reattributed_word():
    utts = [
        utt("agent", "hello", 0.0, 0.5),
        utt("user", "goodbye", 3.0, 3.5),
    ]
    turns = _smooth(utts)
    turns[0]["role"], turns[1]["role"] = turns[1]["role"], turns[0]["role"]
    assert not verify_fidelity(utts, turns)
