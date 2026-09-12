"""Recording and replaying drafts.

`LiveDrafter` is not exercised here and deliberately so: it is the one class in
the project that calls an answer model, and a test that mocked it would assert
what the mock was told to say. What is tested is everything around it — the
recording format, the replay, and the two ways a replay can be wrong.
"""

import json

import pytest

from rag_contract.answering import Passage
from rag_contract.drafter import (
    DrafterError,
    FixtureDrafter,
    Recording,
    load_recordings,
    record,
    write_recording,
)
from rag_contract.grounding import Claim

QUESTION = "Which HTTP request methods are defined as safe?"
CLAIMS = [Claim("The safe methods are GET, HEAD, OPTIONS, and TRACE.", "rfc9110#9.2.1")]


def passage(section_id: str, rank: int = 1) -> Passage:
    return Passage(
        section_id=section_id,
        citation=f"citation for {section_id}",
        rank=rank,
        score=0.7,
        text=f"text of {section_id}",
    )


def recording(question: str = QUESTION, claims=None, passages=("rfc9110#9.2.1",)):
    return Recording(
        question_id="q04",
        question=question,
        model="claude-opus-5",
        index_version="0fc1763d6701",
        recorded_at="2026-09-12T00:00:00Z",
        passages=tuple(passages),
        claims=tuple(CLAIMS if claims is None else claims),
    )


class TestRecordingFormat:
    def test_a_recording_round_trips_through_json(self):
        before = recording()
        after = Recording.from_dict(json.loads(json.dumps(before.to_dict())))
        assert after == before

    def test_it_records_what_the_model_was_shown(self):
        made = record(
            question_id="q04",
            question=QUESTION,
            passages=[passage("rfc9110#9.2.1"), passage("rfc9110#9.2.2", 2)],
            claims=CLAIMS,
            index_version="0fc1763d6701",
            model="claude-opus-5",
        )
        assert made.passages == ("rfc9110#9.2.1", "rfc9110#9.2.2")

    def test_a_malformed_fixture_fails_loudly(self):
        with pytest.raises(DrafterError, match="malformed draft fixture"):
            Recording.from_dict({"question_id": "q04"})

    def test_recordings_are_written_and_read_back(self, tmp_path):
        write_recording(recording(), tmp_path)
        assert load_recordings(tmp_path) == [recording()]

    def test_a_missing_fixture_directory_names_the_command_that_fills_it(
        self, tmp_path
    ):
        with pytest.raises(DrafterError, match="record-drafts"):
            load_recordings(tmp_path / "absent")


class TestReplay:
    def test_a_recorded_draft_replays(self):
        drafter = FixtureDrafter([recording()])
        assert drafter.draft(QUESTION, [passage("rfc9110#9.2.1")]) == CLAIMS

    def test_an_empty_recorded_draft_replays_as_a_decline(self):
        drafter = FixtureDrafter([recording(claims=[])])
        assert drafter.draft(QUESTION, [passage("rfc9110#9.2.1")]) == []

    def test_replaying_a_question_that_was_never_recorded_fails(self):
        # The failure mode this guards: a question edited after the drafts
        # were recorded would otherwise be scored against a model response to
        # a prompt nobody sent.
        drafter = FixtureDrafter([recording()])
        with pytest.raises(DrafterError, match="no committed draft"):
            drafter.draft("Which methods are idempotent?", [])

    def test_two_fixtures_for_one_question_fail(self):
        with pytest.raises(DrafterError, match="same question"):
            FixtureDrafter([recording(), recording()])

    def test_replay_does_not_read_the_passages_it_is_handed(self):
        # The recording holds what the model actually saw. Passing different
        # passages at replay must not silently change the draft.
        drafter = FixtureDrafter([recording()])
        assert drafter.draft(QUESTION, []) == CLAIMS


class TestDrift:
    def test_no_drift_when_retrieval_is_unchanged(self):
        drafter = FixtureDrafter([recording(passages=("rfc9110#9.2.1",))])
        assert drafter.drift(QUESTION, [passage("rfc9110#9.2.1")]) == ()

    def test_a_section_the_model_never_saw_is_reported(self):
        drafter = FixtureDrafter([recording(passages=("rfc9110#9.2.1",))])
        drifted = drafter.drift(
            QUESTION, [passage("rfc9110#9.2.1"), passage("rfc9110#9.2.2", 2)]
        )
        assert drifted == ("rfc9110#9.2.2",)

    def test_drift_is_reported_rather_than_raised(self):
        # Retrieval legitimately moves with the chunk parameters. A replay
        # across such a change is still worth scoring; it just has to say so.
        drafter = FixtureDrafter([recording(passages=("rfc9110#9.2.1",))])
        assert drafter.draft(QUESTION, [passage("rfc9111#5.2")]) == CLAIMS
