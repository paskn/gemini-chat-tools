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
            - has_file_context (bool): Whether prompt follows file upload chunks
            - file_context_tokens (int): Total tokens from uploaded files
            - specificity_score (float): 0-1, based on concrete details (includes file context)
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
        
        # Check for file upload context (look backward for file uploads)
        has_file_context, file_context_tokens = _detect_file_upload_context(chunks, i)
        
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
            SPECIFICITY_INDICATORS,
            has_file_context,
            file_context_tokens
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
            'has_file_context': has_file_context,
            'file_context_tokens': file_context_tokens,
            'specificity_score': specificity_score,
            'prompt_type': prompt_type,
        })
    
    return pd.DataFrame(prompt_data)


def _detect_file_upload_context(chunks: List[Dict[str, Any]], current_index: int) -> tuple[bool, int]:
    """Detect if a prompt follows file upload chunks.
    
    This function looks backward from the current chunk to see if there are
    recent file upload chunks (driveDocument or driveImage) that provide
    context for this prompt.
    
    Args:
        chunks: List of all chunks
        current_index: Index of the current prompt chunk
        
    Returns:
        Tuple of (has_file_context, total_tokens_from_files)
    """
    has_file_context = False
    file_context_tokens = 0
    
    # Look backward up to 5 chunks (but stop at model responses)
    lookback_limit = 5
    for offset in range(1, lookback_limit + 1):
        if current_index - offset < 0:
            break
        
        prev_chunk = chunks[current_index - offset]
        prev_role = prev_chunk.get('role', '')
        
        # Stop if we hit a model response (files are uploaded in sequence before user message)
        if prev_role == 'model':
            break
        
        # Check for file uploads (both documents and images)
        has_drive_doc = 'driveDocument' in prev_chunk
        has_drive_image = 'driveImage' in prev_chunk
        
        if has_drive_doc or has_drive_image:
            has_file_context = True
            file_context_tokens += prev_chunk.get('tokenCount', 0)
    
    return has_file_context, file_context_tokens


def _calculate_specificity_score(
    text: str, 
    word_count: int, 
    indicators: Dict[str, str],
    has_file_context: bool = False,
    file_context_tokens: int = 0
) -> float:
    """Calculate a specificity score for a prompt (0-1).
    
    Higher scores indicate more concrete, specific prompts with:
    - Quoted text
    - Line/section references
    - Named sections or reviewers
    - File names
    - Specific terms
    - Longer length
    - File upload context (provides concrete material to reference)
    
    Args:
        text: The prompt text
        word_count: Number of words in prompt
        indicators: Dictionary of indicator names to regex patterns
        has_file_context: Whether this prompt follows file upload chunks
        file_context_tokens: Total tokens from uploaded files
        
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
    
    # 4. File upload context bonus (uploaded files provide concrete context)
    # Even short prompts like "Please review this draft" are specific when
    # they reference uploaded files with thousands of tokens
    if has_file_context:
        # Base bonus for having file context
        score += 2.0
        
        # Additional bonus based on amount of uploaded content
        # 0-1000 tokens: +0 points
        # 1000-5000 tokens: +0 to +1 points (linear)
        # 5000+ tokens: +1 point
        if file_context_tokens >= 5000:
            score += 1.0
        elif file_context_tokens > 1000:
            score += (file_context_tokens - 1000) / 4000  # Linear scale
    
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


def detect_prompt_fatigue(prompt_df: pd.DataFrame, window_size: int = 3) -> Dict[str, Any]:
    """Detect signs of declining prompt quality over time (prompt fatigue).
    
    This function analyzes how prompt quality changed from the beginning to the
    end of the conversation, identifying patterns like:
    - Declining length (shorter prompts)
    - Declining specificity (vaguer prompts)
    - Increase in "lazy prompts" (minimal effort)
    
    Args:
        prompt_df: DataFrame from analyze_prompt_patterns()
        window_size: Number of segments to divide conversation into (default: 3 for thirds)
        
    Returns:
        Dictionary with:
            - avg_length_first_segment (float): Average word count in first segment
            - avg_length_last_segment (float): Average word count in last segment
            - length_decline_percentage (float): Percentage change (negative = decline)
            - avg_specificity_first_segment (float): Average specificity in first segment
            - avg_specificity_last_segment (float): Average specificity in last segment
            - specificity_decline_percentage (float): Percentage change
            - lazy_prompts (List[int]): Chunk indices of lazy prompts
            - lazy_prompt_count (int): Total number of lazy prompts
            - lazy_prompts_first_segment (int): Count in first segment
            - lazy_prompts_last_segment (int): Count in last segment
            - examples (List[Dict]): Examples of lazy prompts with context
            - has_fatigue (bool): Whether significant fatigue detected
            - segment_stats (List[Dict]): Detailed stats for each segment
            
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.meta import analyze_prompt_patterns, detect_prompt_fatigue
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> prompt_df = analyze_prompt_patterns(analysis._chunks)
        >>> fatigue = detect_prompt_fatigue(prompt_df)
        >>> 
        >>> if fatigue['has_fatigue']:
        >>>     print(f"Prompt quality declined {fatigue['length_decline_percentage']:.1f}%")
        >>>     print(f"Found {fatigue['lazy_prompt_count']} lazy prompts")
    """
    
    if len(prompt_df) == 0:
        return {
            'avg_length_first_segment': 0,
            'avg_length_last_segment': 0,
            'length_decline_percentage': 0,
            'avg_specificity_first_segment': 0,
            'avg_specificity_last_segment': 0,
            'specificity_decline_percentage': 0,
            'lazy_prompts': [],
            'lazy_prompt_count': 0,
            'lazy_prompts_first_segment': 0,
            'lazy_prompts_last_segment': 0,
            'examples': [],
            'has_fatigue': False,
            'segment_stats': []
        }
    
    # Divide conversation into segments
    n_prompts = len(prompt_df)
    segment_size = n_prompts // window_size
    
    segment_stats = []
    for i in range(window_size):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < window_size - 1 else n_prompts
        segment = prompt_df.iloc[start_idx:end_idx]
        
        segment_stats.append({
            'segment_number': i + 1,
            'start_index': start_idx,
            'end_index': end_idx,
            'prompt_count': len(segment),
            'avg_word_count': segment['word_count'].mean(),
            'avg_specificity': segment['specificity_score'].mean(),
            'median_word_count': segment['word_count'].median(),
            'median_specificity': segment['specificity_score'].median(),
        })
    
    # Get first and last segments
    first_segment = prompt_df.iloc[:segment_size]
    last_segment = prompt_df.iloc[-segment_size:]
    
    # Calculate averages
    avg_length_first = first_segment['word_count'].mean()
    avg_length_last = last_segment['word_count'].mean()
    avg_spec_first = first_segment['specificity_score'].mean()
    avg_spec_last = last_segment['specificity_score'].mean()
    
    # Calculate decline percentages
    if avg_length_first > 0:
        length_decline_pct = ((avg_length_last - avg_length_first) / avg_length_first) * 100
    else:
        length_decline_pct = 0
    
    if avg_spec_first > 0:
        spec_decline_pct = ((avg_spec_last - avg_spec_first) / avg_spec_first) * 100
    else:
        spec_decline_pct = 0
    
    # Identify lazy prompts
    lazy_prompts = _identify_lazy_prompts(prompt_df)
    lazy_prompt_indices = lazy_prompts['chunk_index'].tolist()
    
    # Count lazy prompts in first vs last segment
    first_segment_chunks = set(first_segment['chunk_index'])
    last_segment_chunks = set(last_segment['chunk_index'])
    
    lazy_first = sum(1 for idx in lazy_prompt_indices if idx in first_segment_chunks)
    lazy_last = sum(1 for idx in lazy_prompt_indices if idx in last_segment_chunks)
    
    # Get examples of lazy prompts
    examples = []
    for idx, row in lazy_prompts.head(10).iterrows():
        examples.append({
            'chunk': row['chunk_index'],
            'text': row['user_text'],
            'category': row['lazy_category'],
            'word_count': row['word_count'],
            'specificity': row['specificity_score']
        })
    
    # Determine if significant fatigue detected
    # Criteria: length decline > 20% OR specificity decline > 15% OR significant increase in lazy prompts
    has_fatigue = (
        length_decline_pct < -20 or 
        spec_decline_pct < -15 or
        (lazy_last > lazy_first * 2 and lazy_last > 3)  # Doubled lazy prompts and more than 3
    )
    
    return {
        'avg_length_first_segment': avg_length_first,
        'avg_length_last_segment': avg_length_last,
        'length_decline_percentage': length_decline_pct,
        'avg_specificity_first_segment': avg_spec_first,
        'avg_specificity_last_segment': avg_spec_last,
        'specificity_decline_percentage': spec_decline_pct,
        'lazy_prompts': lazy_prompt_indices,
        'lazy_prompt_count': len(lazy_prompts),
        'lazy_prompts_first_segment': lazy_first,
        'lazy_prompts_last_segment': lazy_last,
        'examples': examples,
        'has_fatigue': has_fatigue,
        'segment_stats': segment_stats
    }


def _identify_lazy_prompts(prompt_df: pd.DataFrame) -> pd.DataFrame:
    """Identify prompts that show minimal effort (lazy prompts).
    
    Lazy prompts are characterized by:
    - Very short length (< 20 characters or < 5 words)
    - Low specificity score (< 0.15)
    - Vague terms without context ("fix", "improve", "better")
    - Acknowledgments without follow-up ("ok", "thanks", "yes")
    
    Args:
        prompt_df: DataFrame from analyze_prompt_patterns()
        
    Returns:
        DataFrame of lazy prompts with added 'lazy_category' column
    """
    
    lazy_mask = pd.Series([False] * len(prompt_df), index=prompt_df.index)
    categories = [''] * len(prompt_df)
    
    for idx, row in prompt_df.iterrows():
        text = row['user_text']
        text_lower = text.lower().strip()
        word_count = row['word_count']
        prompt_length = row['prompt_length']
        specificity = row['specificity_score']
        
        # Category 1: Extremely short (< 20 chars or < 5 words)
        # But exclude prompts with file context (they're actually specific)
        if (prompt_length < 20 or word_count < 5) and not row['has_file_context']:
            lazy_mask[idx] = True
            categories[idx] = 'extremely_short'
            continue
        
        # Category 2: Low specificity and short (< 10 words, specificity < 0.15)
        # But exclude prompts with file context (they're actually specific)
        if word_count < 10 and specificity < 0.15 and not row['has_file_context']:
            lazy_mask[idx] = True
            categories[idx] = 'short_and_vague'
            continue
        
        # Category 3: Vague imperatives without context
        vague_patterns = [
            (r'^fix\s+(this|that|it)\.?$', 'vague_imperative'),
            (r'^(improve|enhance|make\s+better)\s+(this|that|it)\.?$', 'vague_imperative'),
            (r'^(change|update|modify)\s+(this|that|it)\.?$', 'vague_imperative'),
        ]
        
        for pattern, category in vague_patterns:
            if re.match(pattern, text_lower):
                lazy_mask[idx] = True
                categories[idx] = category
                break
        
        # Category 4: Pure acknowledgments (but only if they're the full prompt)
        ack_patterns = [r'^ok\.?$', r'^thanks?\.?$', r'^thank\s+you\.?$', 
                       r'^yes\.?$', r'^no\.?$', r'^sure\.?$', r'^good\.?$', r'^great\.?$']
        
        for pattern in ack_patterns:
            if re.match(pattern, text_lower):
                lazy_mask[idx] = True
                categories[idx] = 'acknowledgment_only'
                break
    
    lazy_prompts = prompt_df[lazy_mask].copy()
    lazy_prompts['lazy_category'] = [categories[idx] for idx in lazy_prompts.index]
    
    return lazy_prompts


def categorize_prompt_types(prompt_df: pd.DataFrame, include_plot: bool = False) -> Dict[str, Any]:
    """Categorize prompts by intent and return distribution statistics.
    
    This function analyzes the prompt_type column from analyze_prompt_patterns()
    and provides a breakdown of how prompts are distributed across different
    categories of intent.
    
    Args:
        prompt_df: DataFrame from analyze_prompt_patterns()
        include_plot: If True, create and display a visualization
        
    Returns:
        Dictionary with:
            - counts (Dict[str, int]): Count of prompts per category
            - percentages (Dict[str, float]): Percentage per category
            - total (int): Total number of prompts
            - most_common (str): Most frequently used prompt type
            - least_common (str): Least frequently used prompt type (excluding zeros)
            - diversity_score (float): 0-1, higher = more diverse prompt types
            
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.meta import analyze_prompt_patterns, categorize_prompt_types
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> prompt_df = analyze_prompt_patterns(analysis._chunks)
        >>> categories = categorize_prompt_types(prompt_df, include_plot=True)
        >>> 
        >>> print(f"Most common: {categories['most_common']}")
        >>> print(f"Distribution: {categories['percentages']}")
    """
    
    if len(prompt_df) == 0:
        return {
            'counts': {},
            'percentages': {},
            'total': 0,
            'most_common': None,
            'least_common': None,
            'diversity_score': 0.0
        }
    
    # Count prompt types
    type_counts = prompt_df['prompt_type'].value_counts().to_dict()
    total = len(prompt_df)
    
    # Calculate percentages
    percentages = {ptype: (count / total) * 100 for ptype, count in type_counts.items()}
    
    # Identify most and least common
    most_common = max(type_counts.items(), key=lambda x: x[1])[0]
    least_common = min(type_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate diversity score (Shannon entropy normalized to 0-1)
    # Higher score = more diverse usage of different prompt types
    import math
    entropy = 0.0
    for count in type_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    # Normalize by max possible entropy (log2 of number of categories)
    max_entropy = math.log2(len(type_counts)) if len(type_counts) > 1 else 1.0
    diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0
    
    result = {
        'counts': type_counts,
        'percentages': percentages,
        'total': total,
        'most_common': most_common,
        'least_common': least_common,
        'diversity_score': diversity_score
    }
    
    # Create visualization if requested
    if include_plot:
        _plot_prompt_type_distribution(type_counts, total)
    
    return result


def _plot_prompt_type_distribution(type_counts: Dict[str, int], total: int):
    """Create visualization of prompt type distribution.
    
    Args:
        type_counts: Dictionary mapping prompt type to count
        total: Total number of prompts
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return
    
    # Sort by count (descending)
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    types = [t[0] for t in sorted_types]
    counts = [t[1] for t in sorted_types]
    percentages = [(c / total) * 100 for c in counts]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Prompt Type Distribution', fontsize=16, fontweight='bold')
    
    # 1. Bar chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    bars = ax1.barh(types, counts, color=colors)
    ax1.set_xlabel('Count', fontsize=11)
    ax1.set_ylabel('Prompt Type', fontsize=11)
    ax1.set_title('Prompt Type Frequency', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add count labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f' {count} ({pct:.1f}%)',
                ha='left', va='center', fontsize=9)
    
    # 2. Pie chart
    ax2.pie(counts, labels=types, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Prompt Type Proportions', fontsize=13)
    
    plt.tight_layout()
    plt.show()


def plot_prompt_quality_trend(prompt_df: pd.DataFrame, 
                             show_lazy_prompts: bool = True,
                             window_size: int = 5,
                             figsize: tuple = (14, 8)) -> None:
    """Create comprehensive visualization of prompt quality trends over time.
    
    This function generates a multi-panel plot showing:
    - Word count over time with moving average
    - Specificity score over time with quality zones
    - Segment-level comparison (first vs middle vs last)
    - Lazy prompt highlights on timeline
    
    Args:
        prompt_df: DataFrame from analyze_prompt_patterns()
        show_lazy_prompts: If True, highlight lazy prompts on plots
        window_size: Size of rolling window for moving averages
        figsize: Figure size as (width, height) tuple
        
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.meta import analyze_prompt_patterns, plot_prompt_quality_trend
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> prompt_df = analyze_prompt_patterns(analysis._chunks)
        >>> plot_prompt_quality_trend(prompt_df)
    """
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib required for plotting. Install with: uv add matplotlib")
        return
    
    if len(prompt_df) == 0:
        print("No prompts to plot")
        return
    
    # Detect lazy prompts if requested
    lazy_prompts = None
    if show_lazy_prompts:
        lazy_prompts = _identify_lazy_prompts(prompt_df)
    
    # Detect fatigue for segment stats
    fatigue = detect_prompt_fatigue(prompt_df)
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Prompt Quality Trend Analysis', fontsize=16, fontweight='bold')
    
    # Setup x-axis (conversation progress)
    x = np.arange(len(prompt_df))
    
    # === Panel 1: Word Count Over Time ===
    word_counts = prompt_df['word_count'].values
    
    ax1.plot(x, word_counts, 'o-', alpha=0.5, markersize=4, label='Actual', color='steelblue')
    
    # Add rolling average
    if len(word_counts) >= window_size:
        rolling_avg = pd.Series(word_counts).rolling(window=window_size, center=True).mean()
        ax1.plot(x, rolling_avg, linewidth=2, label=f'{window_size}-prompt average', color='darkblue')
    
    # Highlight lazy prompts
    if show_lazy_prompts and len(lazy_prompts) > 0:
        lazy_indices = [list(prompt_df['chunk_index']).index(idx) for idx in lazy_prompts['chunk_index'] 
                       if idx in list(prompt_df['chunk_index'])]
        if lazy_indices:
            lazy_word_counts = [word_counts[i] for i in lazy_indices]
            ax1.scatter(lazy_indices, lazy_word_counts, color='red', s=100, 
                       marker='x', linewidths=2, label='Lazy prompts', zorder=5)
    
    ax1.set_xlabel('Prompt Number', fontsize=10)
    ax1.set_ylabel('Word Count', fontsize=10)
    ax1.set_title('Prompt Length Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: Specificity Over Time ===
    specificity = prompt_df['specificity_score'].values
    
    # Create quality zones (background shading)
    ax2.axhspan(0, 0.2, alpha=0.1, color='red', label='Low quality')
    ax2.axhspan(0.2, 0.4, alpha=0.1, color='yellow', label='Medium quality')
    ax2.axhspan(0.4, 1.0, alpha=0.1, color='green', label='High quality')
    
    ax2.plot(x, specificity, 'o-', alpha=0.5, markersize=4, label='Actual', color='darkorange')
    
    # Add rolling average
    if len(specificity) >= window_size:
        rolling_avg = pd.Series(specificity).rolling(window=window_size, center=True).mean()
        ax2.plot(x, rolling_avg, linewidth=2, label=f'{window_size}-prompt average', color='darkred')
    
    # Highlight lazy prompts
    if show_lazy_prompts and len(lazy_prompts) > 0:
        lazy_indices = [list(prompt_df['chunk_index']).index(idx) for idx in lazy_prompts['chunk_index'] 
                       if idx in list(prompt_df['chunk_index'])]
        if lazy_indices:
            lazy_specificity = [specificity[i] for i in lazy_indices]
            ax2.scatter(lazy_indices, lazy_specificity, color='red', s=100, 
                       marker='x', linewidths=2, label='Lazy prompts', zorder=5)
    
    ax2.set_xlabel('Prompt Number', fontsize=10)
    ax2.set_ylabel('Specificity Score (0-1)', fontsize=10)
    ax2.set_title('Prompt Specificity Over Time', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Segment Comparison ===
    segment_stats = fatigue['segment_stats']
    segment_numbers = [s['segment_number'] for s in segment_stats]
    avg_words = [s['avg_word_count'] for s in segment_stats]
    avg_spec = [s['avg_specificity'] for s in segment_stats]
    
    x_seg = np.arange(len(segment_numbers))
    width = 0.35
    
    bars1 = ax3.bar(x_seg - width/2, avg_words, width, label='Avg Word Count', color='steelblue')
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x_seg + width/2, avg_spec, width, label='Avg Specificity', color='darkorange')
    
    ax3.set_xlabel('Conversation Segment', fontsize=10)
    ax3.set_ylabel('Average Word Count', fontsize=10, color='steelblue')
    ax3_twin.set_ylabel('Average Specificity', fontsize=10, color='darkorange')
    ax3.set_title('Quality Metrics by Segment', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_seg)
    ax3.set_xticklabels([f'Segment {i}' for i in segment_numbers])
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3_twin.tick_params(axis='y', labelcolor='darkorange')
    ax3_twin.set_ylim(0, 1)
    
    # Add combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === Panel 4: Fatigue Summary ===
    ax4.axis('off')
    
    # Create summary text
    summary_lines = [
        "FATIGUE DETECTION SUMMARY",
        "=" * 40,
        "",
        f"Total Prompts: {len(prompt_df)}",
        f"Lazy Prompts: {fatigue['lazy_prompt_count']} ({fatigue['lazy_prompt_count']/len(prompt_df)*100:.1f}%)",
        "",
        "Length Trend:",
        f"  First segment: {fatigue['avg_length_first_segment']:.1f} words",
        f"  Last segment: {fatigue['avg_length_last_segment']:.1f} words",
        f"  Change: {fatigue['length_decline_percentage']:+.1f}%",
        "",
        "Specificity Trend:",
        f"  First segment: {fatigue['avg_specificity_first_segment']:.3f}",
        f"  Last segment: {fatigue['avg_specificity_last_segment']:.3f}",
        f"  Change: {fatigue['specificity_decline_percentage']:+.1f}%",
        "",
        f"Fatigue Detected: {'YES ⚠️' if fatigue['has_fatigue'] else 'NO ✓'}",
    ]
    
    # Color-code the fatigue detection result
    fatigue_color = 'red' if fatigue['has_fatigue'] else 'green'
    
    summary_text = '\n'.join(summary_lines)
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add fatigue indicator box
    if fatigue['has_fatigue']:
        ax4.text(0.5, 0.05, 'FATIGUE DETECTED', transform=ax4.transAxes,
                fontsize=14, fontweight='bold', color='white',
                horizontalalignment='center', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    else:
        ax4.text(0.5, 0.05, 'NO FATIGUE', transform=ax4.transAxes,
                fontsize=14, fontweight='bold', color='white',
                horizontalalignment='center', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def identify_conversation_segments(
    timeline: pd.DataFrame,
    prompt_df: pd.DataFrame,
    method: str = 'token_burst',
    n_segments: int = None
) -> List[Dict[str, Any]]:
    """Group conversation into distinct segments based on activity patterns.
    
    This function identifies natural phases in the conversation by detecting
    patterns like topic shifts, file upload boundaries, or token usage bursts.
    
    Args:
        timeline: DataFrame from ChatAnalysis.timeline()
        prompt_df: DataFrame from analyze_prompt_patterns()
        method: Segmentation method:
            - 'token_burst': High-activity clusters (default)
            - 'topic_shift': Major topic changes based on prompt patterns
            - 'file_upload': Split on file upload boundaries
            - 'equal_chunks': Divide into N equal-sized segments
        n_segments: Number of segments (required for 'equal_chunks', optional for others)
        
    Returns:
        List of segment dictionaries with:
            - segment_id (int): Segment number (1 to N)
            - start_prompt (int): First prompt number in segment
            - end_prompt (int): Last prompt number in segment
            - start_chunk (int): First chunk index in segment
            - end_chunk (int): Last chunk index in segment
            - sequence_start (float): Normalized start position (0.0-1.0)
            - sequence_end (float): Normalized end position (0.0-1.0)
            - message_count (int): Total messages in segment
            - user_messages (int): User message count
            - model_messages (int): Model message count
            - total_tokens (int): Total tokens in segment
            - avg_tokens_per_message (float): Average tokens per message
            - avg_word_count (float): Average user prompt word count
            - avg_specificity (float): Average user prompt specificity
            - dominant_prompt_types (List[str]): Most common prompt types
            - has_file_uploads (bool): Whether segment includes file uploads
            - characteristics (List[str]): Detected segment characteristics
            - description (str): Human-readable segment description
            
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.meta import analyze_prompt_patterns, identify_conversation_segments
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> timeline = analysis.timeline()
        >>> prompt_df = analyze_prompt_patterns(analysis._chunks)
        >>> 
        >>> # Detect segments by topic shift
        >>> segments = identify_conversation_segments(timeline, prompt_df, method='topic_shift')
        >>> for seg in segments:
        >>>     print(f"Segment {seg['segment_id']}: {seg['description']}")
    """
    
    if method == 'equal_chunks':
        if n_segments is None:
            n_segments = 3  # Default to thirds
        return _segment_equal_chunks(timeline, prompt_df, n_segments)
    
    elif method == 'token_burst':
        return _segment_token_burst(timeline, prompt_df, n_segments)
    
    elif method == 'topic_shift':
        return _segment_topic_shift(timeline, prompt_df, n_segments)
    
    elif method == 'file_upload':
        return _segment_file_upload(timeline, prompt_df)
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
__all__ = [
    'analyze_prompt_patterns',
    'detect_prompt_fatigue',
    'categorize_prompt_types',
    'plot_prompt_quality_trend',
]
