
from __future__ import annotations
from typing import List, Literal
from langcodes import Language

def svara_prompt(text: str, lang_code: str, gender: Literal["male", "female"]) -> str:
    """Format the prompt for the Svara-TTS model.
    
    Args:
        text: The text to synthesize.
        lang_code: The language code.
        gender: The gender of the voice.
    """
    language = Language.get(lang_code).display_name()    
    gender   = gender.capitalize()
    voice   = f"{language} ({gender})"        
    base = f"<|audio|> {voice}: {text}<|eot_id|>"
    return "<custom_token_3>" + base + "<custom_token_4><custom_token_5>"

_DEFAULT_SEPARATORS = [
    "\n\n",   # paragraphs
    "\n",     # lines
    "। ",      # Hindi danda (sentence end)
    ". ", "? ", "! ", "… ",  # sentence enders
    ",",      # comma only if no space available
    " ",      # space (preferred over comma)
    "",       # hard fallback (character-level)
]


def _split_text_recursive(
    text: str,
    max_len: int,
    overlap: int,
    separators: List[str],
) -> List[str]:
    """
    Recursively split text using a hierarchy of separators.
    
    Args:
        text: Text to split
        max_len: Maximum chunk size
        overlap: Number of characters to overlap between chunks
        separators: List of separators to try in order of preference
    """
    if len(text) <= max_len:
        return [text]
    
    chunks: List[str] = []
    
    # Try each separator in order
    for separator in separators:
        if separator == "":
            # Fallback: character-level split
            break
            
        # Split by the current separator
        if separator in text:
            parts = text.split(separator)
            current_chunk = ""
            
            for i, part in enumerate(parts):
                # Re-add separator (except for last part)
                if i < len(parts) - 1:
                    part_with_sep = part + separator
                else:
                    part_with_sep = part
                
                # If this part alone is too long, recursively split it
                if len(part_with_sep) > max_len:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # Recursively split with remaining separators
                    remaining_seps = separators[separators.index(separator) + 1:]
                    if remaining_seps:
                        sub_chunks = _split_text_recursive(
                            part_with_sep, max_len, overlap, remaining_seps
                        )
                        chunks.extend(sub_chunks)
                    else:
                        # Hard split if no more separators
                        chunks.extend(_hard_split(part_with_sep, max_len, overlap))
                    continue
                
                # Check if adding this part would exceed max_len
                if len(current_chunk) + len(part_with_sep) > max_len:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Start new chunk with overlap
                        if overlap > 0 and len(current_chunk) >= overlap:
                            current_chunk = current_chunk[-overlap:] + part_with_sep
                        else:
                            current_chunk = part_with_sep
                    else:
                        current_chunk = part_with_sep
                else:
                    current_chunk += part_with_sep
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return [c for c in chunks if c]
    
    # If no separator worked, do hard split
    return _hard_split(text, max_len, overlap)


def _hard_split(text: str, max_len: int, overlap: int) -> List[str]:
    """Split text at character boundaries when no separator is found."""
    chunks: List[str] = []
    start = 0
    
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end
        
        # Prevent infinite loop
        if start >= len(text) or (overlap > 0 and start == end - overlap and end == len(text)):
            break
    
    return chunks


def chunk_text(
    text: str,
    max_len: int = 280,
    overlap: int = 24,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Split text into chunks using a hierarchy of separators.

    Args:
        text: input text
        max_len: desired max chunk size (characters)
        overlap: desired overlap between chunks (characters)
        separators: override the default separator preference order
    """
    seps = separators or _DEFAULT_SEPARATORS
    
    if not text:
        return []
    
    if len(text) <= max_len:
        return [text]
    
    chunks = _split_text_recursive(text, max_len, overlap, seps)
    return [c.strip() for c in chunks if c.strip()]