"""Gemini Chat Tools - Utilities for analyzing Gemini chat exports."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
    user_messages: int
    model_messages: int
    thinking_chunks: int
    total_tokens: int
    has_grounding: bool
    web_searches: List[str]
    grounding_sources_count: int
    file_references: List[FileReference]
    structure_summary: Dict[str, Any]
    
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
            f"  User messages: {self.user_messages}",
            f"  Model responses: {self.model_messages}",
            f"  Thinking/reasoning chunks: {self.thinking_chunks}",
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


__all__ = ['analyze_gemini_chat', 'ChatAnalysis', 'main']
