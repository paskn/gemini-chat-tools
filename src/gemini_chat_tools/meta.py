import re
from typing import List, Dict, Any
import pandas as pd


def _calculate_specificity_score(
    text: str, 
    word_count: int, 
    indicators: Dict[str, str]
) -> float:
    """Calculate a specificity score for a prompt (0-1).
    
    Higher scores indicate more concrete, specific prompts with:
    - Quoted text
    - Line/section references
    - Named sections or reviewers
    - File names
    - Specific terms
    - Longer length
    
    Args:
        text: The prompt text
        word_count: Number of words in prompt
        indicators: Dictionary of indicator names to regex patterns
        
    Returns:
        Float between 0 and 1
    """
    score = 0.0
    max_score = 10.0  # Maximum possible score
    
    # 1. Check for specificity indicators (each worth 1 point)
    for pattern in indicators.values():
        if re.search(pattern, text, re.IGNORECASE):
            score += 1.0
    
    # 2. Length bonus (longer prompts tend to be more specific)
    # 0-20 words: 0 points
    # 20-50 words: 0-1 points (linear)
    # 50+ words: 1 point
    if word_count >= 50:
        score += 1.0
    elif word_count > 20:
        score += (word_count - 20) / 30  # Linear scale from 20 to 50
    
    # 3. Quoted text bonus (explicit examples are very specific)
    quoted_count = len(re.findall(r'["\'].*?["\']', text))
    if quoted_count > 0:
        score += min(quoted_count * 0.5, 2.0)  # Up to 2 points for quotes
    
    # Normalize to 0-1
    return min(score / max_score, 1.0)
