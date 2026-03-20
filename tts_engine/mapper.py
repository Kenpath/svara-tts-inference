
from __future__ import annotations
import re
from typing import List, Optional, Iterable

_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")

def extract_custom_token_numbers(text: str):
    """Extract custom token numbers from a text string.
    
    Each custom token is represented by a <custom_token_N> tag in the text, where N is the token number.
    This function extracts all the token numbers from the text and yields them one by one.
    
    Args:
        text: The text string to extract custom token numbers from.
        
    Yields:
        int: The custom token number.
    """
    for m in _TOKEN_RE.findall(text or ""):
        try:
            n = int(m)
            if n != 0:
                yield n
        except Exception:
            continue

def raw_to_code_id(raw_num: int, good_idx: int) -> int:
    """Convert a raw number to a code id.
    
    The code id is calculated using the band offset rule: raw_num - 10 - ((good_idx % 7) * 4096).
    
    Args:
        raw_num: The raw number to convert.
        good_idx: The index of the good token.
    """
    return raw_num - 10 - ((good_idx % 7) * 4096)

class SvaraMapper:
    """
    Aggregates code ids, keeping track of good token count.
    Emits a sliding window of `window_size` codes every time a full frame
    boundary is reached and enough codes have accumulated.

    Default window_size=28 (4 frames). Can be set to 56 (8 frames) for
    fewer, larger decode calls — trades TTFB for throughput.

    Args:
        window_size: Number of codes per emitted window. Must be a multiple
                     of 7. Default 28 (4 frames).
    """
    def __init__(self, window_size: int = 28):
        if window_size < 7 or window_size % 7 != 0:
            raise ValueError(f"window_size must be a multiple of 7, got {window_size}")
        self.window_size = window_size
        self.codes: List[int] = []
        self.good = 0

    def feed_raw(self, raw: int) -> Optional[List[int]]:
        """Feed a raw token number. Returns a window when ready, else None."""
        code = raw_to_code_id(raw, self.good)
        if code <= 0:
            return None
        self.codes.append(code)
        self.good += 1
        if self.good % 7 == 0 and self.good >= self.window_size:
            return self.codes[-self.window_size:]
        return None

    def feed_text(self, token_text: str) -> List[List[int]]:
        """Return zero or more ready windows from a token_text."""
        out: List[List[int]] = []
        for n in extract_custom_token_numbers(token_text):
            win = self.feed_raw(n)
            if win is not None:
                out.append(win)
        return out
