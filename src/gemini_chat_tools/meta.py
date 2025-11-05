"""Meta-analysis functions for analyzing Gemini chat conversations.

This module provides tools to analyze conversation patterns, prompt quality,
and interaction dynamics in Gemini chat exports.
"""

import re
from typing import List, Dict, Any
import pandas as pd


def analyze_prompt_patterns(chunks: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze patterns in user prompts throughout the conversation.
    
    This function extracts various features from user prompts to understand
    how prompt style and quality evolved over the conversation.
    
    Args:
        chunks: List of chunk dictionaries from Gemini chat JSON
        
    Returns:
        pandas DataFrame with columns:
            - chunk_index (int): Position in conversation
            - user_text (str): The prompt text
            - prompt_length (int): Character count
            - word_count (int): Number of words
            - has_question (bool): Contains '?'
            - is_command (bool): Starts with imperative verb
            - is_followup (bool): References previous conversation
            - specificity_score (float): 0-1, based on concrete details
            - prompt_type (str): Categorized type of prompt
            
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.meta import analyze_prompt_patterns
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> prompt_df = analyze_prompt_patterns(analysis._chunks)
        >>> 
        >>> # Analyze prompt evolution
        >>> print(f"Average prompt length: {prompt_df['prompt_length'].mean():.1f}")
        >>> print(f"Question prompts: {prompt_df['has_question'].sum()}")
    """
    
    # Imperative verbs commonly used in commands
    IMPERATIVE_VERBS = [
        'fix', 'change', 'rewrite', 'revise', 'edit', 'update', 'modify',
        'improve', 'clarify', 'explain', 'describe', 'add', 'remove',
        'delete', 'insert', 'replace', 'make', 'create', 'generate',
        'show', 'tell', 'give', 'provide', 'suggest', 'list', 'check',
        'review', 'analyze', 'identify', 'find', 'address', 'respond'
    ]
    
    # Follow-up patterns
    FOLLOWUP_PATTERNS = [
        r'\bthat\s+(?:doesn\'t|does\s+not|won\'t|will\s+not)\s+work\b',
        r'\btry\s+again\b',
        r'\bwhat\s+about\b',
        r'\bhow\s+about\b',
        r'\binstead\b',
        r'\bno[,\s]',
        r'\bactually\b',
        r'\bwait\b',
        r'\bnever\s*mind\b',
        r'\blet\'s\s+try\b',
        r'\b(?:the\s+)?(?:previous|last|earlier)\b',
        r'\b(?:that|this)\s+(?:response|suggestion|version|one)\b',
        r'\bbut\s+',
        r'\bhowever\b',
    ]
    
    # Specificity indicators (things that make a prompt more concrete)
    SPECIFICITY_INDICATORS = {
        'quoted_text': r'["\'].*?["\']',  # Quoted strings
        'line_numbers': r'\bline\s+\d+\b|\bl\.\s*\d+\b',  # Line references
        'section_refs': r'\b(?:section|paragraph|page|chapter)\s+\d+\b',  # Section refs
        'named_sections': r'\b(?:introduction|methods?|results?|discussion|conclusion)\b',  # Paper sections
        'reviewer_refs': r'\b(?:reviewer|R)\s*[123]\b',  # Reviewer references
        'file_names': r'\b\w+\.(?:csv|xlsx?|txt|pdf|py|r)\b',  # File names
        'specific_terms': r'\b(?:table|figure|equation|variable|metric|parameter)\s+\w+\b',  # Specific elements
    }
    
    prompt_data = []
    
    for i, chunk in enumerate(chunks):
        role = chunk.get('role', '')
        text = chunk.get('text', '')
        
        # Only analyze user prompts
        if role != 'user':
            continue
        
        # Skip empty text (file upload chunks)
        if not text.strip():
            continue
        
        # Basic metrics
        prompt_length = len(text)
        words = text.split()
        word_count = len(words)
        has_question = '?' in text
        
        # Check for imperative command
        is_command = False
        text_lower = text.lower().strip()
        # Check first few words for imperative verbs
        first_words = text_lower.split()[:3]
        for verb in IMPERATIVE_VERBS:
            if any(word.startswith(verb) for word in first_words):
                is_command = True
                break
        
        # Check for follow-up patterns
        is_followup = False
        for pattern in FOLLOWUP_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                is_followup = True
                break
        
        # Calculate specificity score (0-1)
        specificity_score = _calculate_specificity_score(
            text, 
            word_count, 
            SPECIFICITY_INDICATORS
        )
        
        # Categorize prompt type
        prompt_type = _categorize_prompt_type(
            text, 
            has_question, 
            is_command, 
            is_followup
        )
        
        prompt_data.append({
            'chunk_index': i,
            'user_text': text,
            'prompt_length': prompt_length,
            'word_count': word_count,
            'has_question': has_question,
            'is_command': is_command,
            'is_followup': is_followup,
            'specificity_score': specificity_score,
            'prompt_type': prompt_type,
        })
    
    return pd.DataFrame(prompt_data)


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


__all__ = [
    'analyze_prompt_patterns',
]
