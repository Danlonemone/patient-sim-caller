"""The hang-up marker must never be spoken, even when the stream splits it across chunks."""

import pytest

from caller.speech import HANGUP_MARKER, SpeechFilter, marker_prefix_len


def run(chunks):
    """Feed chunks through a fresh filter; return (spoken text, hung_up)."""
    f = SpeechFilter()
    spoken = "".join(f.feed(c) for c in chunks) + f.flush()
    return spoken, f.hung_up


def test_plain_text_passes_through():
    assert run(["Hi, I'd like to ", "book an appointment."]) == (
        "Hi, I'd like to book an appointment.",
        False,
    )


def test_marker_in_one_chunk_is_stripped():
    assert run(["Great, thanks so much. Bye!<hangup>"]) == ("Great, thanks so much. Bye!", True)


@pytest.mark.parametrize("split", range(1, len(HANGUP_MARKER)))
def test_marker_split_across_chunks_is_stripped(split):
    chunks = ["Okay, bye!" + HANGUP_MARKER[:split], HANGUP_MARKER[split:]]
    assert run(chunks) == ("Okay, bye!", True)


def test_marker_split_one_character_per_chunk():
    assert run(["Bye!"] + list(HANGUP_MARKER)) == ("Bye!", True)


def test_text_after_marker_is_dropped():
    assert run(["Bye!<hangup>", " Oh, one more thing"]) == ("Bye!", True)


def test_partial_marker_that_never_completes_is_spoken():
    # "<hang" is held back in case it becomes the marker; at end of stream it is released.
    assert run(["Press <hang"]) == ("Press <hang", False)


def test_held_text_is_released_when_marker_does_not_follow():
    f = SpeechFilter()
    assert f.feed("Is it <h") == "Is it "
    assert f.feed("ere?") == "<here?"
    assert not f.hung_up


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Um, I need a refill.", "I need a refill."),
        ("Uh I think so.", "I think so."),
        ("Hmm, Tuesday works.", "Tuesday works."),
        ("Hm. Okay.", "Okay."),
    ],
)
def test_fillers_are_removed(text, expected):
    assert run([text])[0] == expected


def test_words_containing_filler_letters_are_kept():
    assert run(["My umbrella and the hummingbird."])[0] == "My umbrella and the hummingbird."


def test_marker_prefix_len():
    assert marker_prefix_len("Bye!") == 0
    assert marker_prefix_len("Bye!<") == 1
    assert marker_prefix_len("Bye!<hangu") == len("<hangu")
    # a complete marker is not a *proper* prefix; the caller checks for it before this
    assert marker_prefix_len("<hangup>") == 0
