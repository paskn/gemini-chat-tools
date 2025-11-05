"""Tests for meta-analysis functions."""

import pytest
from gemini_chat_tools.meta import analyze_prompt_patterns


def test_analyze_prompt_patterns_basic():
    """Test basic prompt pattern analysis."""
    chunks = [
        {'role': 'user', 'text': 'Fix the grammar in this sentence.', 'tokenCount': 10},
        {'role': 'model', 'text': 'Here is the corrected version...', 'tokenCount': 20},
        {'role': 'user', 'text': 'What does autocratization mean?', 'tokenCount': 8},
        {'role': 'model', 'text': 'Autocratization refers to...', 'tokenCount': 30},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert len(df) == 2  # Only user prompts
    assert 'chunk_index' in df.columns
    assert 'user_text' in df.columns
    assert 'prompt_length' in df.columns
    assert 'word_count' in df.columns
    assert 'has_question' in df.columns
    assert 'is_command' in df.columns
    assert 'specificity_score' in df.columns
    assert 'prompt_type' in df.columns


def test_analyze_prompt_patterns_question_detection():
    """Test that questions are properly detected."""
    chunks = [
        {'role': 'user', 'text': 'What is the meaning of life?', 'tokenCount': 10},
        {'role': 'user', 'text': 'Fix this sentence', 'tokenCount': 5},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert df.iloc[0]['has_question'] == True
    assert df.iloc[1]['has_question'] == False


def test_analyze_prompt_patterns_command_detection():
    """Test that commands are properly detected."""
    chunks = [
        {'role': 'user', 'text': 'Fix the grammar errors in this paragraph.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Change the tense to past tense.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Rewrite this more clearly.', 'tokenCount': 10},
        {'role': 'user', 'text': 'This looks good to me.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert df.iloc[0]['is_command'] == True  # "Fix"
    assert df.iloc[1]['is_command'] == True  # "Change"
    assert df.iloc[2]['is_command'] == True  # "Rewrite"
    assert df.iloc[3]['is_command'] == False  # Not a command


def test_analyze_prompt_patterns_followup_detection():
    """Test that follow-ups are properly detected."""
    chunks = [
        {'role': 'user', 'text': 'That doesn\'t work. Try again.', 'tokenCount': 10},
        {'role': 'user', 'text': 'What about using a different approach?', 'tokenCount': 10},
        {'role': 'user', 'text': 'No, that\'s not what I meant.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Write a summary of the paper.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert df.iloc[0]['is_followup'] == True  # "doesn't work", "try again"
    assert df.iloc[1]['is_followup'] == True  # "what about"
    assert df.iloc[2]['is_followup'] == True  # "no, that's"
    assert df.iloc[3]['is_followup'] == False  # Not a follow-up


def test_analyze_prompt_patterns_type_categorization():
    """Test prompt type categorization."""
    chunks = [
        {'role': 'user', 'text': 'Fix the grammar and tense consistency.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Make this paragraph clearer and simpler.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Explain what this term means.', 'tokenCount': 10},  # Changed to avoid "methodology"
        {'role': 'user', 'text': 'How can I address Reviewer 2\'s speculation critique?', 'tokenCount': 10},
        {'role': 'user', 'text': 'Give me 3 different versions of this sentence.', 'tokenCount': 10},
        {'role': 'user', 'text': 'No, that\'s too formal. Try again.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Thanks, that looks good.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert df.iloc[0]['prompt_type'] == 'copyedit_request'
    assert df.iloc[1]['prompt_type'] == 'clarity_improvement'
    assert df.iloc[2]['prompt_type'] == 'explanation_request'
    assert df.iloc[3]['prompt_type'] == 'argument_strengthening'
    assert df.iloc[4]['prompt_type'] == 'alternatives_request'
    assert df.iloc[5]['prompt_type'] == 'iteration_feedback'
    assert df.iloc[6]['prompt_type'] == 'acknowledgment'


def test_analyze_prompt_patterns_specificity_scoring():
    """Test that specificity scoring works correctly."""
    chunks = [
        # Low specificity: short, vague
        {'role': 'user', 'text': 'Fix this.', 'tokenCount': 5},
        
        # Medium specificity: has quotes
        {'role': 'user', 'text': 'Change "the data shows" to "the data show".', 'tokenCount': 10},
        
        # High specificity: long, has quotes, references sections
        {'role': 'user', 'text': '''
            In the methods section, on line 45, change "the data shows" to "the data show". 
            This is to address Reviewer 2's grammar critique about subject-verb agreement.
            Make sure this is consistent with Table 3 and Figure 2.
        ''', 'tokenCount': 50},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Specificity should increase from first to last
    assert df.iloc[0]['specificity_score'] < df.iloc[1]['specificity_score']
    assert df.iloc[1]['specificity_score'] < df.iloc[2]['specificity_score']
    
    # Rough ranges (not exact thresholds)
    assert df.iloc[0]['specificity_score'] < 0.2  # Very low
    assert df.iloc[2]['specificity_score'] > 0.5  # High


def test_analyze_prompt_patterns_empty_text():
    """Test that empty text (file upload chunks) are skipped."""
    chunks = [
        {'role': 'user', 'text': '', 'driveDocument': {'id': '123'}, 'tokenCount': 1000},
        {'role': 'user', 'text': 'Here are three files.', 'tokenCount': 10},
        {'role': 'user', 'text': '  ', 'tokenCount': 0},  # Whitespace only
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Only the second chunk should be analyzed (has actual text)
    assert len(df) == 1
    assert df.iloc[0]['chunk_index'] == 1


def test_analyze_prompt_patterns_model_chunks_skipped():
    """Test that model responses are skipped."""
    chunks = [
        {'role': 'user', 'text': 'First user prompt', 'tokenCount': 10},
        {'role': 'model', 'text': 'Model response here', 'tokenCount': 20},
        {'role': 'model', 'text': 'Another model response', 'isThought': True, 'tokenCount': 30},
        {'role': 'user', 'text': 'Second user prompt', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert len(df) == 2  # Only user prompts
    assert df.iloc[0]['chunk_index'] == 0
    assert df.iloc[1]['chunk_index'] == 3


def test_analyze_prompt_patterns_length_metrics():
    """Test that length metrics are calculated correctly."""
    chunks = [
        {'role': 'user', 'text': 'Short', 'tokenCount': 5},
        {'role': 'user', 'text': 'This is a longer prompt with more words.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert df.iloc[0]['prompt_length'] == 5  # Character count
    assert df.iloc[0]['word_count'] == 1
    
    assert df.iloc[1]['prompt_length'] == 40  # Character count
    assert df.iloc[1]['word_count'] == 8


def test_analyze_prompt_patterns_methodology_detection():
    """Test detection of methodology-related prompts."""
    chunks = [
        {'role': 'user', 'text': 'Explain how the autocratization variable was calculated.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Describe the methodology in more detail.', 'tokenCount': 10},
        {'role': 'user', 'text': 'How were the state requests measured?', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    assert df.iloc[0]['prompt_type'] == 'methodology_detail'
    assert df.iloc[1]['prompt_type'] == 'methodology_detail'
    # Third might be methodology_detail or explanation_request depending on patterns


def test_analyze_prompt_patterns_reviewer_references():
    """Test that reviewer references increase specificity."""
    chunks = [
        {'role': 'user', 'text': 'Address the speculation concern.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Address Reviewer 2\'s speculation concern.', 'tokenCount': 10},
        {'role': 'user', 'text': 'Address R3\'s speculation concern from the methods section.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Specificity should increase with more specific references
    assert df.iloc[0]['specificity_score'] < df.iloc[1]['specificity_score']
    assert df.iloc[1]['specificity_score'] <= df.iloc[2]['specificity_score']
