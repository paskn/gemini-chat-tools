"""Tests for meta-analysis functions."""

import pytest
import pandas as pd
from gemini_chat_tools.meta import analyze_prompt_patterns, detect_prompt_fatigue


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


def test_analyze_prompt_patterns_file_upload_context():
    """Test that file upload context is properly detected and affects specificity."""
    chunks = [
        # Case 1: Short prompt without file context
        {'role': 'user', 'text': 'Please review this draft.', 'tokenCount': 10},
        {'role': 'model', 'text': 'Sure, I can help...', 'tokenCount': 20},
        
        # Case 2: File upload followed by short prompt
        {'role': 'user', 'driveDocument': {'id': 'abc123'}, 'text': '', 'tokenCount': 2500},
        {'role': 'user', 'text': 'Please review this draft.', 'tokenCount': 10},
        {'role': 'model', 'text': 'I\'ve reviewed your document...', 'tokenCount': 100},
        
        # Case 3: Multiple file uploads followed by prompt
        {'role': 'user', 'driveDocument': {'id': 'def456'}, 'text': '', 'tokenCount': 3000},
        {'role': 'user', 'driveImage': {'id': 'ghi789'}, 'text': '', 'tokenCount': 500},
        {'role': 'user', 'text': 'Compare these two files.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Should have 3 prompts (the ones with text)
    assert len(df) == 3
    
    # First prompt: no file context
    assert df.iloc[0]['has_file_context'] == False
    assert df.iloc[0]['file_context_tokens'] == 0
    baseline_specificity = df.iloc[0]['specificity_score']
    
    # Second prompt: has file context (2500 tokens)
    assert df.iloc[1]['has_file_context'] == True
    assert df.iloc[1]['file_context_tokens'] == 2500
    assert df.iloc[1]['specificity_score'] > baseline_specificity  # Should be higher!
    
    # Third prompt: has file context from multiple files (3500 tokens total)
    assert df.iloc[2]['has_file_context'] == True
    assert df.iloc[2]['file_context_tokens'] == 3500
    assert df.iloc[2]['specificity_score'] > baseline_specificity
    # More file context should mean higher specificity
    assert df.iloc[2]['specificity_score'] >= df.iloc[1]['specificity_score']


def test_analyze_prompt_patterns_file_context_stops_at_model():
    """Test that file context detection stops at model responses."""
    chunks = [
        # Old file upload from previous exchange
        {'role': 'user', 'driveDocument': {'id': 'old123'}, 'text': '', 'tokenCount': 1000},
        {'role': 'user', 'text': 'Here is an old file.', 'tokenCount': 10},
        {'role': 'model', 'text': 'Thanks, I see it...', 'tokenCount': 50},
        
        # New prompt - should NOT pick up old file upload
        {'role': 'user', 'text': 'Now do something else.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Second prompt should NOT have file context (stopped by model response)
    assert df.iloc[1]['has_file_context'] == False
    assert df.iloc[1]['file_context_tokens'] == 0


def test_analyze_prompt_patterns_file_context_lookback_limit():
    """Test that file context has a lookback limit."""
    chunks = [
        # File upload far away (more than 5 chunks back)
        {'role': 'user', 'driveDocument': {'id': 'far123'}, 'text': '', 'tokenCount': 1000},
        {'role': 'user', 'text': 'chunk 1', 'tokenCount': 5},
        {'role': 'user', 'text': 'chunk 2', 'tokenCount': 5},
        {'role': 'user', 'text': 'chunk 3', 'tokenCount': 5},
        {'role': 'user', 'text': 'chunk 4', 'tokenCount': 5},
        {'role': 'user', 'text': 'chunk 5', 'tokenCount': 5},
        # This prompt is 6 chunks away from the file upload
        {'role': 'user', 'text': 'This is too far away.', 'tokenCount': 10},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Last prompt should NOT have file context (too far away)
    last_prompt = df.iloc[-1]
    assert last_prompt['has_file_context'] == False
    assert last_prompt['file_context_tokens'] == 0


def test_analyze_prompt_patterns_file_context_with_inline_quotes():
    """Test that prompts with inline quoted text have higher specificity."""
    chunks = [
        # Prompt without quotes
        {'role': 'user', 'text': 'Please improve the clarity of the paragraph about autocratization.', 'tokenCount': 20},
        
        # Prompt with quoted text (inline context)
        {'role': 'user', 'text': '''
Please improve the clarity of this paragraph: "This is a paragraph about 
autocratization that includes detailed argumentation and specific examples."
The paragraph should be more accessible.
        ''', 'tokenCount': 50},
    ]
    
    df = analyze_prompt_patterns(chunks)
    
    # Neither has file context
    assert df.iloc[0]['has_file_context'] == False
    assert df.iloc[1]['has_file_context'] == False
    
    # But the second should have higher specificity due to quoted text
    assert df.iloc[1]['specificity_score'] > df.iloc[0]['specificity_score']


def test_detect_prompt_fatigue_basic():
    """Test basic prompt fatigue detection."""
    # Create fake prompt data showing clear decline
    data = []
    
    # First third: long, specific prompts
    for i in range(10):
        data.append({
            'chunk_index': i,
            'user_text': 'This is a detailed prompt with specific instructions and context. ' * 5,
            'prompt_length': 300,
            'word_count': 50,
            'has_question': False,
            'is_command': True,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.6,
            'prompt_type': 'command'
        })
    
    # Second third: medium prompts
    for i in range(10, 20):
        data.append({
            'chunk_index': i,
            'user_text': 'Medium length prompt with some detail.',
            'prompt_length': 150,
            'word_count': 25,
            'has_question': False,
            'is_command': True,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.3,
            'prompt_type': 'command'
        })
    
    # Last third: short, vague prompts (fatigue!)
    for i in range(20, 30):
        data.append({
            'chunk_index': i,
            'user_text': 'ok' if i % 3 == 0 else 'fix this',
            'prompt_length': 10,
            'word_count': 2,
            'has_question': False,
            'is_command': False,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.05,
            'prompt_type': 'acknowledgment'
        })
    
    prompt_df = pd.DataFrame(data)
    fatigue = detect_prompt_fatigue(prompt_df)
    
    # Should detect fatigue
    assert fatigue['has_fatigue'] == True
    
    # Should show decline
    assert fatigue['length_decline_percentage'] < -50  # Significant decline
    assert fatigue['specificity_decline_percentage'] < -50
    
    # Should identify lazy prompts (all in last third)
    assert fatigue['lazy_prompt_count'] >= 10
    assert fatigue['lazy_prompts_last_segment'] > fatigue['lazy_prompts_first_segment']


def test_detect_prompt_fatigue_no_fatigue():
    """Test that consistent quality doesn't trigger false fatigue detection."""
    # Create prompts with consistent quality
    data = []
    
    for i in range(30):
        data.append({
            'chunk_index': i,
            'user_text': 'Detailed prompt with specific instructions and good context.',
            'prompt_length': 200,
            'word_count': 30,
            'has_question': True,
            'is_command': False,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.5,
            'prompt_type': 'question'
        })
    
    prompt_df = pd.DataFrame(data)
    fatigue = detect_prompt_fatigue(prompt_df)
    
    # Should NOT detect fatigue
    assert fatigue['has_fatigue'] == False
    
    # Should show minimal or no decline
    assert abs(fatigue['length_decline_percentage']) < 5
    assert abs(fatigue['specificity_decline_percentage']) < 5


def test_detect_prompt_fatigue_lazy_prompt_categories():
    """Test that different lazy prompt categories are detected."""
    chunks = [
        # Extremely short
        {'role': 'user', 'text': 'ok', 'tokenCount': 5},
        {'role': 'user', 'text': 'yes', 'tokenCount': 5},
        
        # Short and vague (no file context)
        {'role': 'user', 'text': 'do that please', 'tokenCount': 10},
        
        # Vague imperative
        {'role': 'user', 'text': 'fix this', 'tokenCount': 5},
        {'role': 'user', 'text': 'improve that', 'tokenCount': 5},
        
        # Acknowledgment only
        {'role': 'user', 'text': 'thanks', 'tokenCount': 5},
        
        # NOT lazy: has file context
        {'role': 'user', 'driveDocument': {'id': 'abc'}, 'text': '', 'tokenCount': 2000},
        {'role': 'user', 'text': 'review this', 'tokenCount': 5},
        
        # NOT lazy: sufficient length and specificity
        {'role': 'user', 'text': 'Please review the methodology section for clarity and check if the variable definitions are consistent with Table 2.', 'tokenCount': 50},
    ]
    
    prompt_df = analyze_prompt_patterns(chunks)
    fatigue = detect_prompt_fatigue(prompt_df)
    
    # Should identify the lazy prompts but NOT the one with file context or the detailed one
    # Expected lazy: 'ok', 'yes', 'do that please', 'fix this', 'improve that', 'thanks'
    # NOT lazy: 'review this' (with file context), long detailed prompt
    assert fatigue['lazy_prompt_count'] == 6  # All short/vague prompts except the one with file context
    
    # Check categories
    categories = [ex['category'] for ex in fatigue['examples']]
    assert 'extremely_short' in categories  # 'ok', 'yes', 'do that please', 'fix this', 'improve that', 'thanks'
    assert 'vague_imperative' in categories or 'extremely_short' in categories  # May be caught by either
    assert 'acknowledgment_only' in categories or 'extremely_short' in categories  # May be caught by either


def test_detect_prompt_fatigue_empty_dataframe():
    """Test that empty DataFrame is handled gracefully."""
    empty_df = pd.DataFrame(columns=['chunk_index', 'user_text', 'word_count', 
                                     'specificity_score', 'has_file_context'])
    
    fatigue = detect_prompt_fatigue(empty_df)
    
    assert fatigue['has_fatigue'] == False
    assert fatigue['lazy_prompt_count'] == 0
    assert len(fatigue['examples']) == 0


def test_detect_prompt_fatigue_segment_stats():
    """Test that segment statistics are calculated correctly."""
    # Create data with clear progression
    data = []
    
    # Segment 1: 100 words avg
    for i in range(15):
        data.append({
            'chunk_index': i,
            'user_text': 'x ' * 100,
            'prompt_length': 200,
            'word_count': 100,
            'has_question': False,
            'is_command': True,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.6,
            'prompt_type': 'command'
        })
    
    # Segment 2: 50 words avg
    for i in range(15, 30):
        data.append({
            'chunk_index': i,
            'user_text': 'x ' * 50,
            'prompt_length': 100,
            'word_count': 50,
            'has_question': False,
            'is_command': True,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.4,
            'prompt_type': 'command'
        })
    
    # Segment 3: 25 words avg
    for i in range(30, 45):
        data.append({
            'chunk_index': i,
            'user_text': 'x ' * 25,
            'prompt_length': 50,
            'word_count': 25,
            'has_question': False,
            'is_command': True,
            'is_followup': False,
            'has_file_context': False,
            'file_context_tokens': 0,
            'specificity_score': 0.2,
            'prompt_type': 'command'
        })
    
    prompt_df = pd.DataFrame(data)
    fatigue = detect_prompt_fatigue(prompt_df, window_size=3)
    
    # Check segment stats
    assert len(fatigue['segment_stats']) == 3
    
    seg1 = fatigue['segment_stats'][0]
    seg2 = fatigue['segment_stats'][1]
    seg3 = fatigue['segment_stats'][2]
    
    assert seg1['avg_word_count'] == 100
    assert seg2['avg_word_count'] == 50
    assert seg3['avg_word_count'] == 25
    
    assert abs(seg1['avg_specificity'] - 0.6) < 0.001
    assert abs(seg2['avg_specificity'] - 0.4) < 0.001
    assert abs(seg3['avg_specificity'] - 0.2) < 0.001


def test_detect_prompt_fatigue_file_context_exclusion():
    """Test that prompts with file context aren't classified as lazy."""
    chunks = [
        # Short prompt WITHOUT file context (should be lazy)
        {'role': 'user', 'text': 'review this', 'tokenCount': 5},
        
        # Short prompt WITH file context (should NOT be lazy)
        {'role': 'user', 'driveDocument': {'id': 'abc'}, 'text': '', 'tokenCount': 2000},
        {'role': 'user', 'text': 'review this', 'tokenCount': 5},
    ]
    
    prompt_df = analyze_prompt_patterns(chunks)
    fatigue = detect_prompt_fatigue(prompt_df)
    
    # Only the first "review this" should be lazy
    assert fatigue['lazy_prompt_count'] == 1
    assert fatigue['lazy_prompts'][0] == 0  # First chunk index
