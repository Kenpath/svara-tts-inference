"""
Token constants for Svara TTS model.

This module centralizes all token IDs and special tokens used throughout the codebase.
"""

# ============================================================================
# Base Tokenizer Configuration
# ============================================================================

TOKENISER_LENGTH = 128256  # Base vocabulary size of the tokenizer


# ============================================================================
# Special Tokens (from base tokenizer)
# ============================================================================

START_OF_TEXT = 128000
END_OF_TEXT = 128001
EOT_ID = 128009  # End of turn token


# ============================================================================
# Custom Speech Tokens
# ============================================================================

START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2    # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3   # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4     # 128260
START_OF_AI = TOKENISER_LENGTH + 5      # 128261
END_OF_AI = TOKENISER_LENGTH + 6        # 128262
PAD_TOKEN = TOKENISER_LENGTH + 7        # 128263


# ============================================================================
# Audio Token Configuration
# ============================================================================

AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # 128266
AUDIO_VOCAB_SIZE   = 4096  # Each hierarchical level has 4096 possible codes

# Audio token offset positions (7 tokens per frame)
# These are added to raw SNAC codes [0, 4096] to map them into model vocabulary
AUDIO_TOKEN_OFFSETS = [
    AUDIO_TOKENS_START + (0 * AUDIO_VOCAB_SIZE),  # 128266 - codes[0]
    AUDIO_TOKENS_START + (1 * AUDIO_VOCAB_SIZE),  # 132362 - codes[1][2*i]
    AUDIO_TOKENS_START + (2 * AUDIO_VOCAB_SIZE),  # 136458 - codes[2][4*i]
    AUDIO_TOKENS_START + (3 * AUDIO_VOCAB_SIZE),  # 140554 - codes[2][4*i+1]
    AUDIO_TOKENS_START + (4 * AUDIO_VOCAB_SIZE),  # 144650 - codes[1][2*i+1]
    AUDIO_TOKENS_START + (5 * AUDIO_VOCAB_SIZE),  # 148746 - codes[2][4*i+2]
    AUDIO_TOKENS_START + (6 * AUDIO_VOCAB_SIZE),  # 152842 - codes[2][4*i+3]
]


# ============================================================================
# Special Token Strings (for prompt formatting)
# ============================================================================

BEGIN_OF_TEXT_STR = "<|begin_of_text|>"
END_OF_TEXT_STR = "<|end_of_text|>"
EOT_ID_STR = "<|eot_id|>"
AUDIO_STR = "<|audio|>"  # Token ID: 156939

# Custom token strings
START_OF_SPEECH_STR = "<custom_token_1>"
END_OF_SPEECH_STR = "<custom_token_2>"
START_OF_HUMAN_STR = "<custom_token_3>"
END_OF_HUMAN_STR = "<custom_token_4>"
START_OF_AI_STR = "<custom_token_5>"
END_OF_AI_STR = "<custom_token_6>"
PAD_TOKEN_STR = "<custom_token_7>"