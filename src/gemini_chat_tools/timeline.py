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
from gemini_chat_tools import ChatAnalysis


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
    # Load the raw JSON data to access chunks
    # We need to re-parse since ChatAnalysis doesn't expose raw chunks
    import json
    from pathlib import Path
    
    # Note: This is a limitation - we don't have direct access to the file path
    # For now, we'll need to work with what we have in the analysis object
    # TODO: Consider refactoring ChatAnalysis to include chunks or file_path
    
    # Since we can't access the original file from ChatAnalysis alone,
    # we need to accept chunks directly or modify the approach
    raise NotImplementedError(
        "get_conversation_timeline() requires access to raw chunks. "
        "Please use get_conversation_timeline_from_file() instead, "
        "or pass chunks directly to get_conversation_timeline_from_chunks()."
    )


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
    
    **File Upload Merging**: Google Drive file uploads appear as separate chunks
    with driveDocument fields but no text. This function automatically merges
    consecutive file upload chunks with the following user message chunk (if any)
    to represent them as a single conversational turn.
    
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
    
    # First pass: identify file upload sequences and merge them
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        chunk = chunks[i]
        is_thinking = chunk.get('isThought', False)
        
        # Skip thinking chunks if not requested
        if is_thinking and not include_thinking:
            i += 1
            continue
        
        role = chunk.get('role', 'unknown')
        has_drive_doc = 'driveDocument' in chunk
        
        # Check if this is a file upload chunk (user role, has driveDocument, no text)
        if role == 'user' and has_drive_doc and not chunk.get('text', '').strip():
            # Start accumulating file uploads
            file_uploads = [chunk]
            j = i + 1
            
            # Look ahead to collect consecutive file upload chunks
            while j < len(chunks):
                next_chunk = chunks[j]
                next_is_thinking = next_chunk.get('isThought', False)
                
                # Skip thinking chunks in the lookahead
                if next_is_thinking and not include_thinking:
                    j += 1
                    continue
                
                next_role = next_chunk.get('role', 'unknown')
                next_has_drive = 'driveDocument' in next_chunk
                next_text = next_chunk.get('text', '').strip()
                
                # If it's another file upload chunk, add it
                if next_role == 'user' and next_has_drive and not next_text:
                    file_uploads.append(next_chunk)
                    j += 1
                # If it's a user message with text, merge it with the uploads
                elif next_role == 'user' and next_text:
                    file_uploads.append(next_chunk)
                    j += 1
                    break
                # Otherwise, stop looking
                else:
                    break
            
            # Create merged chunk
            merged_chunk = {
                'chunk_index': i,
                'role': 'user',
                'text': file_uploads[-1].get('text', '') if len(file_uploads) > 1 else '',
                'tokens': sum(c.get('tokenCount', 0) for c in file_uploads),
                'is_thinking': False,
                'has_file_upload': True,
                'file_upload_count': len([c for c in file_uploads if 'driveDocument' in c])
            }
            merged_chunks.append(merged_chunk)
            i = j
        else:
            # Regular chunk (not a file upload sequence)
            merged_chunk = {
                'chunk_index': i,
                'role': role,
                'text': chunk.get('text', ''),
                'tokens': chunk.get('tokenCount', 0),
                'is_thinking': is_thinking,
                'has_file_upload': 'driveDocument' in chunk,
                'file_upload_count': 1 if 'driveDocument' in chunk else 0
            }
            merged_chunks.append(merged_chunk)
            i += 1
    
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
