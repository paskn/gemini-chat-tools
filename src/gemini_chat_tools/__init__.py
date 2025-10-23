"""Gemini Chat Tools - Utilities for analyzing Gemini chat exports."""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass


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
