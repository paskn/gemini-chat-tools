"""Gemini Chat Tools - Utilities for analyzing Gemini chat exports."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FileReference:
    """A reference to a file mentioned in the chat."""
    
    chunk_index: int
    role: str  # 'user' or 'model'
    reference_type: str  # 'drive_document', 'attached', 'mentioned', 'extension'
    context: str  # The text snippet mentioning the file
    detected_filenames: List[str] = field(default_factory=list)
    drive_id: str = ''  # Google Drive document ID if applicable
    token_count: int = 0  # Token count for the file content
    

@dataclass
class ChatAnalysis:
    """Analysis results for a Gemini chat export file."""
    
    file_size_mb: float
    total_chunks: int
    user_messages: int  # Raw chunk count (includes separate file upload chunks)
    model_messages: int  # Raw chunk count (includes thinking chunks)
    thinking_chunks: int
    total_tokens: int
    has_grounding: bool
    web_searches: List[str]
    grounding_sources_count: int
    file_references: List[FileReference]
    structure_summary: Dict[str, Any]
    user_turns: int = 0  # Conversational turns (file uploads merged with messages)
    model_turns: int = 0  # Conversational turns (excluding thinking chunks)
    _chunks: List[Dict[str, Any]] = field(default_factory=list, repr=False)  # Raw chunks for timeline generation
    _timeline_cache: Any = field(default=None, repr=False)  # Cache for timeline DataFrame
    _timeline_cache_params: bool = field(default=False, repr=False)  # Track include_thinking param
    _files_used_cache: Dict[int, List[str]] = field(default=None, repr=False)  # Cache for files_used mapping
    
    def timeline(self, include_thinking: bool = False):
        """Get conversation timeline as a pandas DataFrame.
        
        This method provides access to the sequential structure of the conversation,
        with file uploads automatically merged with their accompanying messages.
        
        **IMPORTANT**: By default, thinking chunks are EXCLUDED from the timeline.
        Set include_thinking=True to include them.
        
        Args:
            include_thinking: If True, include thinking chunks (default: False)
        
        Returns:
            pandas DataFrame with columns:
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
            >>> from gemini_chat_tools import analyze_gemini_chat
            >>> 
            >>> analysis = analyze_gemini_chat("my_chat.json")
            >>> timeline = analysis.timeline()
            >>> 
            >>> # Analyze conversation structure
            >>> print(f"Total turns: {len(timeline)}")
            >>> print(f"User turns: {len(timeline[timeline['role'] == 'user'])}")
            >>> print(f"Model turns: {len(timeline[timeline['role'] == 'model'])}")
        """
        # Import here to avoid circular dependency
        from gemini_chat_tools.timeline import get_conversation_timeline_from_chunks
        
        # Check if we need to regenerate cache
        if self._timeline_cache is None or self._timeline_cache_params != include_thinking:
            self._timeline_cache = get_conversation_timeline_from_chunks(
                self._chunks,
                include_thinking=include_thinking
            )
            self._timeline_cache_params = include_thinking
        
        return self._timeline_cache
    
    def _build_files_used_mapping(self) -> Dict[int, List[str]]:
        """Build mapping from chunk_index (messageID) to Drive IDs.
        
        This method replicates the merge logic from _merge_file_upload_chunks
        to correctly assign Drive IDs to the corresponding timeline messages.
        
        Returns:
            Dict mapping chunk_index to list of Drive IDs (documents and images)
        """
        files_used = {}
        chunks = self._chunks
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            is_thinking = chunk.get('isThought', False)
            role = chunk.get('role', 'unknown')
            
            # Skip thinking chunks (consistent with timeline)
            if is_thinking:
                i += 1
                continue
            
            # Skip error chunks (consistent with timeline)
            if 'errorMessage' in chunk:
                i += 1
                continue
            
            has_drive_doc = 'driveDocument' in chunk
            has_drive_image = 'driveImage' in chunk
            has_file_upload = has_drive_doc or has_drive_image
            
            # Check if this is a file upload chunk (user role, has drive upload, no text)
            if role == 'user' and has_file_upload and not chunk.get('text', '').strip():
                # Start accumulating file uploads
                start_chunk_index = i  # This is the messageID
                drive_ids = []
                j = i
                
                # Collect Drive IDs from consecutive file upload chunks
                while j < len(chunks):
                    next_chunk = chunks[j]
                    next_is_thinking = next_chunk.get('isThought', False)
                    next_role = next_chunk.get('role', 'unknown')
                    
                    # Skip thinking chunks in lookahead
                    if next_is_thinking:
                        j += 1
                        continue
                    
                    # Skip error chunks in lookahead
                    if 'errorMessage' in next_chunk:
                        j += 1
                        continue
                    
                    next_has_drive_doc = 'driveDocument' in next_chunk
                    next_has_drive_image = 'driveImage' in next_chunk
                    next_has_file = next_has_drive_doc or next_has_drive_image
                    next_text = next_chunk.get('text', '').strip()
                    
                    # If it's another file upload chunk, collect its Drive ID
                    if next_role == 'user' and next_has_file and not next_text:
                        if next_has_drive_doc:
                            drive_id = next_chunk['driveDocument'].get('id', '')
                            if drive_id:
                                drive_ids.append(drive_id)
                        if next_has_drive_image:
                            drive_id = next_chunk['driveImage'].get('id', '')
                            if drive_id:
                                drive_ids.append(drive_id)
                        j += 1
                    # If it's a user message with text, stop (end of upload sequence)
                    elif next_role == 'user' and next_text:
                        j += 1
                        break
                    # Otherwise, stop looking
                    else:
                        break
                
                # Store the mapping
                if drive_ids:
                    files_used[start_chunk_index] = drive_ids
                
                i = j
            else:
                # Check if this single chunk has a file upload
                if has_file_upload:
                    drive_ids = []
                    if has_drive_doc:
                        drive_id = chunk['driveDocument'].get('id', '')
                        if drive_id:
                            drive_ids.append(drive_id)
                    if has_drive_image:
                        drive_id = chunk['driveImage'].get('id', '')
                        if drive_id:
                            drive_ids.append(drive_id)
                    
                    if drive_ids:
                        files_used[i] = drive_ids
                
                i += 1
        
        return files_used
    
    def preprocess_for_topics(
        self,
        remove_urls: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        keep_only_alpha: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        spacy_model: str = "en_core_web_sm"
    ) -> Any:
        """Preprocess conversation timeline for topic modeling.
        
        This is a convenience method that applies standard NLP preprocessing
        to prepare the conversation for topic modeling with BERTopic.
        
        Args:
            remove_urls: Remove HTTP/HTTPS URLs from text
            remove_stopwords: Remove common English stop words
            lemmatize: Apply lemmatization using spaCy
            keep_only_alpha: Keep only alphabetic characters (removes punctuation, numbers)
            custom_stopwords: Additional stop words to remove (e.g., ['gemini', 'ai'])
            spacy_model: spaCy model to use for lemmatization
            
        Returns:
            pandas DataFrame with preprocessed 'text' column
            
        Example:
            >>> from gemini_chat_tools import analyze_gemini_chat
            >>> 
            >>> analysis = analyze_gemini_chat("chat.json")
            >>> preprocessed_timeline = analysis.preprocess_for_topics()
            >>> 
            >>> # Use with topic modeling
            >>> from gemini_chat_tools.topic_model import TopicModelAnalysis
            >>> topic_analysis = TopicModelAnalysis.from_timeline(
            ...     preprocessed_timeline,
            ...     preprocess=False  # Already preprocessed
            ... )
        """
        from gemini_chat_tools.topic_model import preprocess_timeline
        
        timeline = self.timeline()
        return preprocess_timeline(
            timeline,
            remove_urls=remove_urls,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            keep_only_alpha=keep_only_alpha,
            custom_stopwords=custom_stopwords,
            spacy_model=spacy_model
        )
    
    @property
    def files_used(self) -> Dict[int, List[str]]:
        """Get mapping from message chunk_index to Drive IDs.
        
        Returns a dictionary where:
        - Keys are chunk_index values (matching timeline's chunk_index column)
        - Values are lists of Google Drive IDs (documents and images)
        
        The keys correspond to messages in the timeline that have file uploads.
        You can use this to look up which files were attached to a specific message:
        
            timeline_row = timeline[timeline['chunk_index'] == 21].iloc[0]
            drive_ids = analysis.files_used[21]
        
        Returns:
            Dict[int, List[str]]: Mapping from chunk_index to list of Drive IDs
            
        Example:
            >>> from gemini_chat_tools import analyze_gemini_chat
            >>> 
            >>> analysis = analyze_gemini_chat("my_chat.json")
            >>> timeline = analysis.timeline()
            >>> 
            >>> # Get all messages with file uploads
            >>> file_messages = timeline[timeline['has_file_upload']]
            >>> 
            >>> # For each message, get the Drive IDs
            >>> for idx, row in file_messages.iterrows():
            >>>     chunk_idx = row['chunk_index']
            >>>     drive_ids = analysis.files_used[chunk_idx]
            >>>     print(f"Message {chunk_idx}: {len(drive_ids)} files")
            >>> 
            >>> # Look up specific message
            >>> if 21 in analysis.files_used:
            >>>     print(f"Chunk 21 files: {analysis.files_used[21]}")
        """
        if self._files_used_cache is None:
            self._files_used_cache = self._build_files_used_mapping()
        return self._files_used_cache
    
    def __str__(self) -> str:
        """Return a formatted string representation of the analysis."""
        lines = [
            "=" * 60,
            "GEMINI CHAT EXPORT ANALYSIS",
            "=" * 60,
            "",
            "FILE INFORMATION:",
            f"  Size: {self.file_size_mb:.2f} MB",
            f"  Total conversation chunks: {self.total_chunks}",
            "",
            "MESSAGE BREAKDOWN:",
            f"  User messages: {self.user_messages} chunks",
            f"  Model messages: {self.model_messages} chunks",
            f"    - Thinking chunks: {self.thinking_chunks}",
            f"    - Response chunks: {self.model_messages - self.thinking_chunks}",
            "",
            "CONVERSATIONAL TURNS:",
            f"  User turns: {self.user_turns}",
            f"  Model turns: {self.model_turns}",
            "",
            "  Note: Turns differ from chunks because:",
            "    - File uploads are merged with their accompanying messages",
            "    - Thinking chunks are excluded from turn counts",
            "",
            "TOKEN USAGE:",
            f"  Total tokens counted: {self.total_tokens:,}",
            "",
            "GROUNDING & RESEARCH:",
            f"  Has grounding data: {'Yes' if self.has_grounding else 'No'}",
        ]
        
        if self.has_grounding:
            lines.extend([
                f"  Web search queries: {len(self.web_searches)}",
                f"  Grounding sources: {self.grounding_sources_count}",
            ])
            
            if self.web_searches:
                lines.append("")
                lines.append("  Search queries:")
                for i, query in enumerate(self.web_searches[:5], 1):
                    lines.append(f"    {i}. {query}")
                if len(self.web_searches) > 5:
                    lines.append(f"    ... and {len(self.web_searches) - 5} more")
        
        # File references section
        lines.extend([
            "",
            "FILE REFERENCES:",
            f"  Total file references found: {len(self.file_references)}",
        ])
        
        if self.file_references:
            # Count by type
            drive_docs = sum(1 for ref in self.file_references if ref.reference_type == 'drive_document')
            attached = sum(1 for ref in self.file_references if ref.reference_type == 'attached')
            mentioned = sum(1 for ref in self.file_references if ref.reference_type == 'mentioned')
            extension = sum(1 for ref in self.file_references if ref.reference_type == 'extension')
            
            lines.extend([
                f"    Google Drive documents: {drive_docs}",
                f"    Attached files: {attached}",
                f"    Mentioned files: {mentioned}",
                f"    File extensions referenced: {extension}",
            ])
            
            # Show Drive document details
            drive_refs = [ref for ref in self.file_references if ref.reference_type == 'drive_document']
            if drive_refs:
                total_drive_tokens = sum(ref.token_count for ref in drive_refs)
                lines.append("")
                lines.append(f"  Google Drive uploads ({len(drive_refs)} documents, {total_drive_tokens:,} tokens):")
                for i, ref in enumerate(drive_refs[:10], 1):
                    lines.append(f"    {i}. Chunk {ref.chunk_index}: {ref.token_count:,} tokens (ID: {ref.drive_id[:20]}...)")
                if len(drive_refs) > 10:
                    lines.append(f"    ... and {len(drive_refs) - 10} more")
            
            # Extract unique filenames
            all_filenames = []
            for ref in self.file_references:
                all_filenames.extend(ref.detected_filenames)
            unique_filenames = sorted(set(all_filenames))
            
            if unique_filenames:
                lines.append("")
                lines.append("  Detected filenames:")
                for i, filename in enumerate(unique_filenames[:10], 1):
                    lines.append(f"    {i}. {filename}")
                if len(unique_filenames) > 10:
                    lines.append(f"    ... and {len(unique_filenames) - 10} more")
            
            # Show attachment contexts
            attachments = [ref for ref in self.file_references if ref.reference_type == 'attached']
            if attachments:
                lines.append("")
                lines.append("  Attachment references:")
                for i, ref in enumerate(attachments[:5], 1):
                    context_preview = ref.context[:80] + "..." if len(ref.context) > 80 else ref.context
                    lines.append(f"    {i}. Chunk {ref.chunk_index} ({ref.role}): {context_preview}")
                if len(attachments) > 5:
                    lines.append(f"    ... and {len(attachments) - 5} more")
        
        lines.extend([
            "",
            "STRUCTURE DETAILS:",
            f"  Run settings: {self.structure_summary.get('has_run_settings', False)}",
            f"  System instruction: {self.structure_summary.get('has_system_instruction', False)}",
            f"  Chunked prompt: {self.structure_summary.get('has_chunked_prompt', False)}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def _detect_file_references(chunks: List[Dict[str, Any]]) -> List[FileReference]:
    """
    Detect file references in chat chunks.
    
    Args:
        chunks: List of conversation chunks
        
    Returns:
        List of FileReference objects
    """
    file_references = []
    
    # Patterns for detecting file references
    attachment_patterns = [
        r'attach(?:ed|ing|ment)?\s+(?:to\s+)?(?:this\s+)?(?:message\s+)?(?:is\s+)?(?:a\s+)?(?:the\s+)?(?:file|document|data|draft)',
        r'(?:i|i\'ve|i\s+have)\s+attached',
        r'see\s+(?:the\s+)?attached',
        r'uploaded?\s+(?:the\s+)?(?:file|document)',
        r'here\s+is\s+(?:the\s+)?(?:file|document|data)',
    ]
    
    mention_patterns = [
        r'in\s+(?:the\s+)?file\s+["\']?([^\s"\',.]+)',
        r'(?:file|document)\s+(?:named|called)\s+["\']?([^\s"\',.]+)',
        r'from\s+(?:the\s+)?file\s+["\']?([^\s"\',.]+)',
    ]
    
    # File extension patterns
    extension_patterns = [
        r'\b([a-zA-Z0-9_\-]+\.(?:csv|xlsx?|json|txt|pdf|docx?|py|r|dta|sav|rds|parquet|feather|hdf5?))\b',
    ]
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        role = chunk.get('role', 'unknown')
        
        # Check for Google Drive documents first
        if 'driveDocument' in chunk:
            drive_doc = chunk.get('driveDocument', {})
            drive_id = drive_doc.get('id', '')
            token_count = chunk.get('tokenCount', 0)
            
            # Try to find a description in nearby chunks
            context = f"Google Drive document (ID: {drive_id})"
            if i + 1 < len(chunks) and chunks[i + 1].get('role') == 'user':
                next_text = chunks[i + 1].get('text', '')
                if next_text:
                    context = next_text[:150]
            
            file_references.append(FileReference(
                chunk_index=i,
                role=role,
                reference_type='drive_document',
                context=context,
                detected_filenames=[],
                drive_id=drive_id,
                token_count=token_count
            ))
            continue  # Skip other checks for this chunk
        
        # Check for attachment references
        for pattern in attachment_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Extract any filenames mentioned in the same context
                filenames = []
                for ext_pattern in extension_patterns:
                    matches = re.findall(ext_pattern, text, re.IGNORECASE)
                    filenames.extend(matches)
                
                # Get a reasonable context snippet
                context = text[:200].strip()
                
                file_references.append(FileReference(
                    chunk_index=i,
                    role=role,
                    reference_type='attached',
                    context=context,
                    detected_filenames=filenames
                ))
                break  # Only count each chunk once as an attachment
        
        # Check for file mentions with specific names
        for pattern in mention_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                filename = match.group(1) if match.groups() else ''
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)].strip()
                
                file_references.append(FileReference(
                    chunk_index=i,
                    role=role,
                    reference_type='mentioned',
                    context=context,
                    detected_filenames=[filename] if filename else []
                ))
        
        # Check for file extensions
        for pattern in extension_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Only add this reference if we haven't already added an attachment reference for this chunk
                already_referenced = any(
                    ref.chunk_index == i and ref.reference_type in ['attached', 'mentioned']
                    for ref in file_references
                )
                
                if not already_referenced:
                    # Find context around first match
                    first_match = re.search(pattern, text, re.IGNORECASE)
                    if first_match:
                        context = text[max(0, first_match.start() - 50):min(len(text), first_match.end() + 50)].strip()
                    else:
                        context = text[:200].strip()
                    
                    file_references.append(FileReference(
                        chunk_index=i,
                        role=role,
                        reference_type='extension',
                        context=context,
                        detected_filenames=list(set(matches))  # Unique filenames
                    ))
    
    return file_references


def _merge_file_upload_chunks(
    chunks: List[Dict[str, Any]], 
    include_thinking: bool = False
) -> List[Dict[str, Any]]:
    """
    Merge file upload chunks with their accompanying messages.
    
    This function encapsulates the core merging logic used by both the timeline
    module and turn counting. It handles:
    - File upload chunks (driveDocument and driveImage) merged with following messages
    - Thinking chunks (excluded by default)
    - Error chunks (always excluded)
    
    Args:
        chunks: List of chunk dictionaries from Gemini chat JSON
        include_thinking: If True, include thinking chunks (default: False)
        
    Returns:
        List of merged chunk dictionaries, where each dict represents a 
        conversational turn and contains:
            - chunk_index: Original index of first chunk
            - role: 'user' or 'model'
            - text: Combined text content
            - tokens: Sum of token counts
            - is_thinking: Whether this is a thinking chunk
            - has_file_upload: Whether this turn includes file uploads
            - file_upload_count: Number of files uploaded
    """
    merged_chunks = []
    i = 0
    
    while i < len(chunks):
        chunk = chunks[i]
        is_thinking = chunk.get('isThought', False)
        role = chunk.get('role', 'unknown')
        
        # Skip thinking chunks if not requested
        if is_thinking and not include_thinking:
            i += 1
            continue
        
        # Skip error chunks (model errors like rate limits)
        if 'errorMessage' in chunk:
            i += 1
            continue
        
        has_drive_doc = 'driveDocument' in chunk
        has_drive_image = 'driveImage' in chunk
        has_file_upload = has_drive_doc or has_drive_image
        
        # Check if this is a file upload chunk (user role, has drive upload, no text)
        if role == 'user' and has_file_upload and not chunk.get('text', '').strip():
            # Start accumulating file uploads
            file_uploads = [chunk]
            j = i + 1
            
            # Look ahead to collect consecutive file upload chunks
            while j < len(chunks):
                next_chunk = chunks[j]
                next_is_thinking = next_chunk.get('isThought', False)
                next_role = next_chunk.get('role', 'unknown')
                
                # Skip thinking chunks in the lookahead
                if next_is_thinking and not include_thinking:
                    j += 1
                    continue
                
                # Skip error chunks in the lookahead
                if 'errorMessage' in next_chunk:
                    j += 1
                    continue
                
                next_has_drive_doc = 'driveDocument' in next_chunk
                next_has_drive_image = 'driveImage' in next_chunk
                next_has_file = next_has_drive_doc or next_has_drive_image
                next_text = next_chunk.get('text', '').strip()
                
                # If it's another file upload chunk, add it
                if next_role == 'user' and next_has_file and not next_text:
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
            # Count both driveDocument and driveImage uploads
            upload_count = len([c for c in file_uploads 
                               if 'driveDocument' in c or 'driveImage' in c])
            
            merged_chunk = {
                'chunk_index': i,
                'role': 'user',
                'text': file_uploads[-1].get('text', '') if len(file_uploads) > 1 else '',
                'tokens': sum(c.get('tokenCount', 0) for c in file_uploads),
                'is_thinking': False,
                'has_file_upload': True,
                'file_upload_count': upload_count
            }
            merged_chunks.append(merged_chunk)
            i = j
        else:
            # Regular chunk (not a file upload sequence)
            has_upload = 'driveDocument' in chunk or 'driveImage' in chunk
            merged_chunk = {
                'chunk_index': i,
                'role': role,
                'text': chunk.get('text', ''),
                'tokens': chunk.get('tokenCount', 0),
                'is_thinking': is_thinking,
                'has_file_upload': has_upload,
                'file_upload_count': 1 if has_upload else 0
            }
            merged_chunks.append(merged_chunk)
            i += 1
    
    return merged_chunks


def _count_conversational_turns(chunks: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Count conversational turns (with file upload merging and thinking exclusion).
    
    This uses the same logic as the timeline module to provide semantically
    meaningful turn counts:
    - File upload chunks (driveDocument and driveImage) are merged with their accompanying user messages
    - Thinking chunks are excluded from model turn counts
    - Error chunks (with errorMessage) are excluded
    
    Args:
        chunks: List of chunk dictionaries from Gemini chat JSON
        
    Returns:
        Tuple of (user_turns, model_turns)
    """
    # Use shared merging logic
    merged_chunks = _merge_file_upload_chunks(chunks, include_thinking=False)
    
    # Count turns by role
    user_turn_count = sum(1 for chunk in merged_chunks if chunk['role'] == 'user')
    model_turn_count = sum(1 for chunk in merged_chunks if chunk['role'] == 'model')
    
    return user_turn_count, model_turn_count


def analyze_gemini_chat(file_path: str | Path) -> ChatAnalysis:
    """
    Analyze a Gemini chat export JSON file.
    
    Args:
        file_path: Path to the JSON file (with or without extension)
        
    Returns:
        ChatAnalysis object with detailed information about the chat
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(file_path)
    
    # If path doesn't exist and has no extension, it might be a file without extension
    if not path.exists() and path.suffix == "":
        # Try to find it as-is (file without extension)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
    elif not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    
    # Load and parse JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Analyze structure
    has_run_settings = 'runSettings' in data
    has_system_instruction = 'systemInstruction' in data
    has_chunked_prompt = 'chunkedPrompt' in data
    
    # Count chunks and messages
    chunks = data.get('chunkedPrompt', {}).get('chunks', [])
    total_chunks = len(chunks)
    
    user_messages = sum(1 for chunk in chunks if chunk.get('role') == 'user')
    model_messages = sum(1 for chunk in chunks if chunk.get('role') == 'model')
    thinking_chunks = sum(1 for chunk in chunks if chunk.get('isThought', False))
    
    # Count conversational turns (with file upload merging)
    user_turns, model_turns = _count_conversational_turns(chunks)
    
    # Count tokens
    total_tokens = sum(chunk.get('tokenCount', 0) for chunk in chunks)
    
    # Analyze grounding data (from the last model response with grounding)
    has_grounding = False
    web_searches = []
    grounding_sources_count = 0
    
    for chunk in reversed(chunks):
        if chunk.get('role') == 'model' and 'grounding' in chunk:
            has_grounding = True
            grounding = chunk.get('grounding', {})
            web_searches = grounding.get('webSearchQueries', [])
            grounding_sources_count = len(grounding.get('groundingSources', []))
            break
    
    # Detect file references
    file_references = _detect_file_references(chunks)
    
    structure_summary = {
        'has_run_settings': has_run_settings,
        'has_system_instruction': has_system_instruction,
        'has_chunked_prompt': has_chunked_prompt,
    }
    
    return ChatAnalysis(
        file_size_mb=file_size_mb,
        total_chunks=total_chunks,
        user_messages=user_messages,
        model_messages=model_messages,
        thinking_chunks=thinking_chunks,
        total_tokens=total_tokens,
        has_grounding=has_grounding,
        web_searches=web_searches,
        grounding_sources_count=grounding_sources_count,
        file_references=file_references,
        structure_summary=structure_summary,
        user_turns=user_turns,
        model_turns=model_turns,
        _chunks=chunks,  # Store chunks for timeline generation
    )


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: gemini-chat-tools <path-to-chat-file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        analysis = analyze_gemini_chat(file_path)
        print(analysis)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


__all__ = ['analyze_gemini_chat', 'ChatAnalysis', 'FileReference', 'main', '_merge_file_upload_chunks']
