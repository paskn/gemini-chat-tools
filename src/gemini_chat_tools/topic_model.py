"""Topic modeling utilities for Gemini chat conversations.

This module provides tools for analyzing conversation topics using BERTopic,
including text preprocessing and comprehensive visualization capabilities.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP


def preprocess_timeline(
    timeline: pd.DataFrame,
    remove_urls: bool = True,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    keep_only_alpha: bool = True,
    custom_stopwords: Optional[List[str]] = None,
    spacy_model: str = "en_core_web_sm"
) -> pd.DataFrame:
    """Preprocess timeline text data for topic modeling.
    
    This function applies standard NLP preprocessing steps to prepare
    conversation text for topic modeling with BERTopic.
    
    Args:
        timeline: DataFrame from ChatAnalysis.timeline()
        remove_urls: Remove HTTP/HTTPS URLs from text
        remove_stopwords: Remove common English stop words
        lemmatize: Apply lemmatization using spaCy
        keep_only_alpha: Keep only alphabetic characters (removes punctuation, numbers)
        custom_stopwords: Additional stop words to remove (e.g., ['gemini', 'ai'])
        spacy_model: spaCy model to use for lemmatization
        
    Returns:
        DataFrame with preprocessed 'text' column
        
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.topic_model import preprocess_timeline
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> timeline = analysis.timeline()
        >>> preprocessed = preprocess_timeline(timeline)
        >>> print(preprocessed['text'].head())
    """
    # Work on a copy
    df = timeline.copy()
    
    # Load spaCy model if lemmatization or stopword removal is needed
    nlp = None
    if lemmatize:
        nlp = spacy.load(spacy_model)
    
    # Build stopword set
    stopwords = set(STOP_WORDS) if remove_stopwords else set()
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    
    # Apply preprocessing steps
    if remove_urls:
        df['text'] = df['text'].apply(lambda x: re.sub(r"http\S+", "", x))
    
    # Convert to lowercase
    df['text'] = df['text'].str.lower()
    
    # Keep only alphabetic characters
    if keep_only_alpha:
        df['text'] = df['text'].apply(
            lambda x: " ".join(re.sub("[^a-zA-Z]+", " ", x).split())
        )
    
    # Lemmatize
    if lemmatize and nlp:
        df['text'] = df['text'].apply(
            lambda x: " ".join([token.lemma_ for token in nlp(x)])
        )
    
    # Remove stopwords (after lemmatization for better results)
    if stopwords:
        df['text'] = df['text'].apply(
            lambda x: " ".join([word for word in x.split() if word not in stopwords])
        )
    
    return df


