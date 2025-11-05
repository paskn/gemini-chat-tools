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


def _categorize_prompt_type(
    text: str, 
    has_question: bool, 
    is_command: bool, 
    is_followup: bool
) -> str:
    """Categorize the type of prompt.
    
    Args:
        text: The prompt text
        has_question: Whether prompt contains '?'
        is_command: Whether prompt is a command/instruction
        is_followup: Whether prompt is a follow-up/iteration
        
    Returns:
        String category: 'copyedit_request', 'clarity_improvement',
        'explanation_request', 'argument_strengthening', 'methodology_detail',
        'alternatives_request', 'iteration_feedback', 'question', 'command',
        'acknowledgment', 'general'
    """
    text_lower = text.lower()
    
    # Priority order matters - check most specific first
    
    # Acknowledgments (very short, non-substantive)
    if len(text.split()) <= 5:
        ack_patterns = ['ok', 'thanks', 'thank you', 'good', 'great', 'yes', 'no', 'sure', 'fine']
        if any(text_lower.strip() == pattern or text_lower.strip().startswith(pattern + ' ') or text_lower.strip().startswith(pattern + ',') 
               for pattern in ack_patterns):
            return 'acknowledgment'
    
    # Iteration feedback (follow-ups with rejection or alternative request)
    if is_followup:
        rejection_patterns = [
            r'\bthat\s+(?:doesn\'t|does\s+not)\s+work',
            r'\btry\s+again',
            r'\bno[,\s]',
            r'\btoo\s+(?:formal|informal|generic|vague)',
        ]
        if any(re.search(pattern, text_lower) for pattern in rejection_patterns):
            return 'iteration_feedback'
    
    # Alternatives request
    alternatives_patterns = [
        r'\b(?:give|show|provide|suggest)\s+(?:me\s+)?(?:\d+|some|several|multiple|other)\s+(?:options?|versions?|alternatives?|ways?)',
        r'\bwhat\s+(?:else|other)',
        r'\bdifferent\s+(?:way|approach|version)',
        r'\btry\s+(?:a\s+)?different',
    ]
    if any(re.search(pattern, text_lower) for pattern in alternatives_patterns):
        return 'alternatives_request'
    
    # Copyedit request (grammar, tense, consistency)
    copyedit_patterns = [
        r'\b(?:grammar|tense|spelling|punctuation|consistency|typo)',
        r'\bfix\s+(?:the\s+)?(?:grammar|tense|errors?)',
        r'\bcheck\s+(?:for\s+)?(?:grammar|tense|errors?|consistency)',
        r'\bcopyedit',
        r'\bproofreading',
    ]
    if any(re.search(pattern, text_lower) for pattern in copyedit_patterns):
        return 'copyedit_request'
    
    # Argument strengthening (check before explanation_request)
    argument_patterns = [
        r'\b(?:reviewer|critique|criticism|objection)',
        r'\baddress\s+(?:the\s+)?(?:concern|critique|criticism|objection)',
        r'\brespond\s+to.*reviewer',
        r'\bstrengthen\s+(?:the\s+)?argument',
        r'\bmore\s+(?:convincing|persuasive|compelling)',
        r'\bsupport\s+(?:the|this)\s+claim',
        r'\bspeculation',
    ]
    if any(re.search(pattern, text_lower) for pattern in argument_patterns):
        return 'argument_strengthening'
    
    # Methodology detail (check before explanation_request to catch specific cases)
    methodology_patterns = [
        r'\b(?:methodology|procedure|process|calculation)',
        r'\bdescribe\s+(?:the\s+)?(?:methodology|method)',
        r'\bexplain\s+(?:how|the).*(?:calculated|measured|determined)',
        r'\bhow\s+(?:was|were).*(?:calculated|measured|determined)',
        r'\bmore\s+detail.*(?:method|process)',
    ]
    if any(re.search(pattern, text_lower) for pattern in methodology_patterns):
        return 'methodology_detail'
    
    # Clarity improvement
    clarity_patterns = [
        r'\b(?:clarify|clarity|clearer|simplify|accessible)',
        r'\bmake\s+(?:this|it)\s+(?:clearer|simpler|more\s+accessible)',
        r'\bhard\s+to\s+(?:understand|follow)',
        r'\bconfusing',
        r'\bexplain\s+(?:this\s+)?(?:better|more\s+clearly)',
        r'\brephrase',
        r'\bless\s+(?:technical|jargon)',
    ]
    if any(re.search(pattern, text_lower) for pattern in clarity_patterns):
        return 'clarity_improvement'
    
    # Explanation request (more general, check after specific categories)
    explanation_patterns = [
        r'\bexplain\s+(?:what|why)',
        r'\bwhat\s+(?:does|is)\s+.*\s+mean',
        r'\bhow\s+(?:does|do|can|should)',
        r'\bwhy\s+(?:does|is|did)',
        r'\bwhat\'s\s+the\s+(?:difference|meaning)',
        r'\bhelp\s+me\s+understand',
    ]
    if any(re.search(pattern, text_lower) for pattern in explanation_patterns):
        return 'explanation_request'
    
    # Generic categories based on structure
    if has_question:
        return 'question'
    elif is_command:
        return 'command'
    else:
        return 'general'
