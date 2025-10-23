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

