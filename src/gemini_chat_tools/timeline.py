"""Timeline analysis functions for Gemini chat conversations.

This module provides functions to analyze the sequential structure and flow
of Gemini AI chat conversations. Since Gemini AI Studio exports do NOT include
timestamps, all analysis is based on sequential order (chunk indices) rather
than calendar time.

IMPORTANT: Thinking chunks are EXCLUDED by default from timeline analysis.
Set include_thinking=True to include them if needed for specialized analysis.
"""

import pandas as pd
from typing import List, Dict
from gemini_chat_tools import ChatAnalysis, _merge_file_upload_chunks


def get_conversation_timeline(
    analysis: ChatAnalysis,
    include_thinking: bool = False
) -> pd.DataFrame:
    """Extract sequential structure of the conversation.
    
    **IMPORTANT**: By default, thinking chunks are EXCLUDED from analysis.
    Set include_thinking=True to include them.
    
    Thinking chunks represent internal model reasoning and are not part
    of the user-facing conversation flow. For presentation purposes and
    conversation analysis, we focus on the interactive dialogue.
    
    **Note**: Gemini AI Studio chat exports do NOT include timestamps.
    This function uses chunk indices as a sequential ordering proxy.
    
    **Note**: This function is now deprecated in favor of analysis.timeline().
    Use the timeline() method on ChatAnalysis objects instead:
    
        analysis = analyze_gemini_chat("file.json")
        timeline = analysis.timeline()
    
    Args:
        analysis: ChatAnalysis object from analyze_gemini_chat()
        include_thinking: If True, include thinking chunks (default: False)
    
    Returns:
        DataFrame with columns:
            - chunk_index (int): Position in conversation (0 to N)
            - sequence_position (float): Normalized position (0.0 to 1.0)
            - role (str): 'user' or 'model'
            - text (str): Message text content
            - tokens (int): Token count for this chunk
            - cumulative_tokens (int): Running total of tokens
            - is_thinking (bool): Whether this is a thinking chunk
            - has_file_upload (bool): Whether chunk includes file upload
    
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.timeline import get_conversation_timeline
        >>> 
        >>> analysis = analyze_gemini_chat("my_chat.json")
        >>> timeline = get_conversation_timeline(analysis)
        >>> 
        >>> # Show conversation structure
        >>> print(f"Total chunks analyzed: {len(timeline)}")
        >>> print(f"User messages: {len(timeline[timeline['role'] == 'user'])}")
        >>> print(f"Model responses: {len(timeline[timeline['role'] == 'model'])}")
        >>> print(f"Total tokens: {timeline['cumulative_tokens'].iloc[-1]:,}")
    """
    # Delegate to the ChatAnalysis.timeline() method
    return analysis.timeline(include_thinking=include_thinking)


def get_conversation_timeline_from_file(
    file_path: str,
    include_thinking: bool = False
) -> pd.DataFrame:
    """Extract conversation timeline directly from a chat file.
    
    **IMPORTANT**: By default, thinking chunks are EXCLUDED from analysis.
    Set include_thinking=True to include them.
    
    This is a convenience function that loads the file and extracts the timeline
    in one step.
    
    Args:
        file_path: Path to the Gemini chat JSON file
        include_thinking: If True, include thinking chunks (default: False)
    
    Returns:
        DataFrame with conversation timeline (see get_conversation_timeline_from_chunks)
    
    Example:
        >>> from gemini_chat_tools.timeline import get_conversation_timeline_from_file
        >>> 
        >>> timeline = get_conversation_timeline_from_file("my_chat.json")
        >>> print(timeline.head())
    """
    import json
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists() and path.suffix == "":
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
    elif not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunkedPrompt', {}).get('chunks', [])
    return get_conversation_timeline_from_chunks(chunks, include_thinking=include_thinking)


def get_conversation_timeline_from_chunks(
    chunks: List[Dict],
    include_thinking: bool = False
) -> pd.DataFrame:
    """Extract conversation timeline from raw chunks.
    
    **IMPORTANT**: By default, thinking chunks are EXCLUDED from analysis.
    Set include_thinking=True to include them.
    
    Thinking chunks represent internal model reasoning and are not part
    of the user-facing conversation flow. For presentation purposes and
    conversation analysis, we focus on the interactive dialogue.
    
    **File Upload Merging**: Google Drive uploads (driveDocument and driveImage)
    appear as separate chunks with no text. This function automatically merges
    consecutive file upload chunks with the following user message chunk (if any)
    to represent them as a single conversational turn.
    
    **Error Chunks**: Model chunks with errorMessage fields (e.g., rate limit errors)
    are excluded from the timeline as they don't represent actual conversational turns.
    
    **Note**: Gemini AI Studio chat exports do NOT include timestamps.
    This function uses chunk indices as a sequential ordering proxy.
    
    Args:
        chunks: List of chunk dictionaries from Gemini chat JSON
        include_thinking: If True, include thinking chunks (default: False)
    
    Returns:
        DataFrame with columns:
            - chunk_index (int): Position of first chunk in conversation (0 to N)
            - sequence_position (float): Normalized position (0.0 to 1.0)
            - role (str): 'user' or 'model'
            - text (str): Message text content
            - tokens (int): Token count (sum of all merged chunks)
            - cumulative_tokens (int): Running total of tokens
            - is_thinking (bool): Whether this is a thinking chunk
            - has_file_upload (bool): Whether this turn includes file uploads
            - file_upload_count (int): Number of files uploaded in this turn
    
    Example:
        >>> import json
        >>> with open("my_chat.json") as f:
        ...     data = json.load(f)
        >>> chunks = data['chunkedPrompt']['chunks']
        >>> 
        >>> from gemini_chat_tools.timeline import get_conversation_timeline_from_chunks
        >>> timeline = get_conversation_timeline_from_chunks(chunks)
        >>> 
        >>> # Analyze token distribution
        >>> user_tokens = timeline[timeline['role'] == 'user']['tokens'].sum()
        >>> model_tokens = timeline[timeline['role'] == 'model']['tokens'].sum()
        >>> print(f"User: {user_tokens:,} tokens, Model: {model_tokens:,} tokens")
    """
    if not chunks:
        return pd.DataFrame(columns=[
            'chunk_index', 'sequence_position', 'role', 'text', 'tokens',
            'cumulative_tokens', 'is_thinking', 'has_file_upload', 'file_upload_count'
        ])
    
    # Use shared merging logic from __init__.py
    merged_chunks = _merge_file_upload_chunks(chunks, include_thinking=include_thinking)
    
    # Create DataFrame
    df = pd.DataFrame(merged_chunks)
    
    if df.empty:
        return df
    
    # Calculate cumulative tokens
    df['cumulative_tokens'] = df['tokens'].cumsum()
    
    # Calculate normalized sequence position (0.0 to 1.0)
    if len(df) > 1:
        df['sequence_position'] = df.index / (len(df) - 1)
    else:
        df['sequence_position'] = 0.0
    
    # Reorder columns
    df = df[[
        'chunk_index',
        'sequence_position', 
        'role',
        'text',
        'tokens',
        'cumulative_tokens',
        'is_thinking',
        'has_file_upload',
        'file_upload_count'
    ]]
    
    return df


__all__ = [
    'get_conversation_timeline',
    'get_conversation_timeline_from_file',
    'get_conversation_timeline_from_chunks'
]
