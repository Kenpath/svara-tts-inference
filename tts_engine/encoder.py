import torch
from typing import List, Optional, Union

from .constants import (
    BOS_TOKEN,
    END_OF_TURN,
    START_OF_HUMAN,
    END_OF_HUMAN,
    START_OF_AI,
    END_OF_AI,
    START_OF_SPEECH,
    END_OF_SPEECH,
    AUDIO_TOKEN,
)

# Optional: pre-create scalar token tensors (1,1) to avoid re-alloc each call
BOS_ID             = torch.tensor([[BOS_TOKEN]],        dtype=torch.int64)
START_OF_HUMAN_ID  = torch.tensor([[START_OF_HUMAN]],   dtype=torch.int64)
END_OF_HUMAN_ID    = torch.tensor([[END_OF_HUMAN]],     dtype=torch.int64)
START_OF_AI_ID     = torch.tensor([[START_OF_AI]],      dtype=torch.int64)
END_OF_AI_ID       = torch.tensor([[END_OF_AI]],        dtype=torch.int64)
START_OF_SPEECH_ID = torch.tensor([[START_OF_SPEECH]],  dtype=torch.int64)
END_OF_SPEECH_ID   = torch.tensor([[END_OF_SPEECH]],    dtype=torch.int64)
END_OF_TURN_ID     = torch.tensor([[END_OF_TURN]],      dtype=torch.int64)
AUDIO_TOKEN_ID     = torch.tensor([[AUDIO_TOKEN]],      dtype=torch.int64)


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is shape (1, seq_len)."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t


def _human_turn(text_ids: torch.Tensor) -> torch.Tensor:
    """
    Build a human text block:
    START_OF_HUMAN, AUDIO_TOKEN, text_ids, END_OF_HUMAN, END_OF_TURN
    """
    text_ids = _ensure_2d(text_ids)
    return torch.cat(
        [
            START_OF_HUMAN_ID,
            AUDIO_TOKEN_ID,
            text_ids,
            END_OF_HUMAN_ID,
            END_OF_TURN_ID,
        ],
        dim=1,
    )


def _audio_turn(audio_ids: torch.Tensor) -> torch.Tensor:
    """
    Build an AI audio reference block:
    START_OF_AI, START_OF_SPEECH, audio_ids, END_OF_SPEECH, END_OF_AI, END_OF_TURN
    """
    audio_ids = _ensure_2d(audio_ids)
    return torch.cat(
        [
            START_OF_AI_ID,
            START_OF_SPEECH_ID,
            audio_ids,
            END_OF_SPEECH_ID,
            END_OF_AI_ID,
            END_OF_TURN_ID,
        ],
        dim=1,
    )


def _final_generation_prefix() -> torch.Tensor:
    """
    Final tail to start generation of speech tokens:
    START_OF_AI, START_OF_SPEECH
    (no END_OF_SPEECH / END_OF_AI here; model generates them)
    """
    return torch.cat(
        [
            START_OF_AI_ID,
            START_OF_SPEECH_ID,
        ],
        dim=1,
    )



def svara_text_to_tokens(
    text: str,
    speaker_id: Optional[str],
    audio_tokens: Optional[List[int]] = None,
    transcript: Optional[str] = None,
    tokenizer = None,
    return_decoded: bool = False,
) -> Union[List[int], str]:
    """
    Build Svara-TTS prompt for:
      - Standard TTS (speaker_id + text)
      - Zero-shot TTS (audio reference only)
      - Zero-shot TTS (audio reference + transcript)

    Args:
        text: Target text to synthesize.
        speaker_id: Speaker identifier (e.g., "Hindi (Male)"). Required for standard TTS.
        audio_tokens: SNAC token sequence from reference audio (WITH offsets).
        transcript: Optional transcript of the reference audio.
        tokenizer: Tokenizer used to convert text to IDs.
        return_decoded: 
            - False (default): return List[int] of token IDs (for `prompt_token_ids` style usage).
            - True: return decoded string prompt (for OpenAI-compatible HTTP APIs that expect text).

    Returns:
        List[int] if return_decoded=False
        str      if return_decoded=True
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for svara_text_to_tokens")
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    blocks: List[torch.Tensor] = [BOS_ID]

    # ---------------------------------------------------------------------
    # ZERO-SHOT PATH (audio_tokens is provided)
    # ---------------------------------------------------------------------
    if audio_tokens is not None:
        audio_tokens_tensor = torch.tensor([audio_tokens], dtype=torch.int64)

        # (a) Optional reference transcript as a human turn
        if transcript and isinstance(transcript, str) and transcript.strip():
            transcript_ids = tokenizer(
                transcript,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            blocks.append(_human_turn(transcript_ids))

        # (b) Reference audio as an AI turn
        blocks.append(_audio_turn(audio_tokens_tensor))

        # (c) Target text as human turn + final AI speech start
        target_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        blocks.append(_human_turn(target_ids))
        blocks.append(_final_generation_prefix())

    # ---------------------------------------------------------------------
    # STANDARD TTS PATH (no audio_tokens)
    # ---------------------------------------------------------------------
    else:
        if speaker_id is None:
            raise ValueError("speaker_id is required for standard TTS")

        prompt = f"{speaker_id}: {text}"
        text_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        # Single human turn, then start AI speech
        blocks.append(_human_turn(text_ids))
        blocks.append(_final_generation_prefix())

    # ---------------------------------------------------------------------
    # Concatenate all blocks
    # ---------------------------------------------------------------------
    full_input_ids = torch.cat(blocks, dim=1).view(-1)

    if return_decoded:
        # Keep special tokens, since your prompt format depends on them.
        return tokenizer.decode(full_input_ids.tolist(), skip_special_tokens=False)

    return full_input_ids.tolist()