"""Cleans the patient LLM's streamed reply before it reaches text-to-speech.

Two jobs, kept free of LiveKit imports so they can be unit-tested on their own:

- Hang-up marker. The patient ends a call by appending HANGUP_MARKER to its goodbye. We use a
  text marker rather than a function tool because sending even one tool definition added
  0.2-0.4 s to Claude's time-to-first-token on every turn (measured). The marker can arrive split
  across stream chunks ("<hang" + "up>"), so a possible partial marker at the end of the buffer
  is held back until the next chunk shows whether it completes.
- Filler words. Deepgram Aura-2 renders filler sounds badly ("um" came out as "Bob", "hmm" as
  "Him"), and the prompt alone didn't stop the LLM writing them, so they are removed here.
"""

from __future__ import annotations

import re

HANGUP_MARKER = "<hangup>"
FILLERS = re.compile(r"\b(?:[Uu]m+|[Uu]h+|[Hh]mm+|[Hh]m)\b[,.]?\s*")


def marker_prefix_len(text: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of HANGUP_MARKER."""
    for k in range(min(len(text), len(HANGUP_MARKER) - 1), 0, -1):
        if text.endswith(HANGUP_MARKER[:k]):
            return k
    return 0


class SpeechFilter:
    """Feed it streamed text chunks; it returns what is safe to speak so far.

    After the marker is seen, `hung_up` is True and any further text is dropped, since nothing
    the model writes after its goodbye should be said aloud.
    """

    def __init__(self) -> None:
        self.hung_up = False
        self._held = ""  # trailing text that could still turn into the marker

    def feed(self, text: str) -> str:
        if self.hung_up:
            return ""
        self._held += text
        if HANGUP_MARKER in self._held:
            self.hung_up = True
            before = self._held.split(HANGUP_MARKER)[0]
            self._held = ""
            return FILLERS.sub("", before)
        keep = marker_prefix_len(self._held)
        ready, self._held = self._held[: len(self._held) - keep], self._held[len(self._held) - keep :]
        return FILLERS.sub("", ready)

    def flush(self) -> str:
        """Release whatever is still held once the stream ends (a partial marker is just text)."""
        out, self._held = FILLERS.sub("", self._held), ""
        return out
