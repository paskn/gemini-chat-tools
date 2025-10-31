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


class TopicModelAnalysis:
    """Wrapper for BERTopic analysis of chat conversations.
    
    This class provides a clean interface for topic modeling with
    comprehensive visualization capabilities. It gives users full
    control over BERTopic parameters while simplifying common tasks.
    
    Attributes:
        model: The fitted BERTopic model
        topics: Topic assignments for each document
        probs: Topic probabilities for each document
        timeline: Original or preprocessed timeline DataFrame
        embeddings: Document embeddings (if computed)
        reduced_embeddings: 2D UMAP embeddings for visualization
        topics_over_time: Temporal topic evolution data
    """
    
    def __init__(
        self,
        model: BERTopic,
        topics: List[int],
        probs: List[float],
        timeline: pd.DataFrame,
        embeddings: Optional[Any] = None,
        reduced_embeddings: Optional[Any] = None
    ):
        """Initialize TopicModelAnalysis.
        
        Args:
            model: Fitted BERTopic model
            topics: Topic assignments
            probs: Topic probabilities
            timeline: Timeline DataFrame with text data
            embeddings: Optional document embeddings
            reduced_embeddings: Optional 2D UMAP embeddings
        """
        self.model = model
        self.topics = topics
        self.probs = probs
        self.timeline = timeline
        self.embeddings = embeddings
        self.reduced_embeddings = reduced_embeddings
        self.topics_over_time = None
        
    @classmethod
    def from_timeline(
        cls,
        timeline: pd.DataFrame,
        preprocess: bool = True,
        preprocess_config: Optional[Dict[str, Any]] = None,
        bertopic_config: Optional[Dict[str, Any]] = None,
        compute_embeddings: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        umap_config: Optional[Dict[str, Any]] = None
    ) -> "TopicModelAnalysis":
        """Create TopicModelAnalysis from a timeline DataFrame.
        
        This is the main entry point for topic modeling. It handles
        preprocessing, model fitting, and optional embedding computation.
        
        Args:
            timeline: DataFrame from ChatAnalysis.timeline()
            preprocess: Apply standard preprocessing
            preprocess_config: Config dict for preprocess_timeline()
            bertopic_config: Config dict for BERTopic() constructor
            compute_embeddings: Compute embeddings for visualizations
            embedding_model: SentenceTransformer model name
            umap_config: Config dict for UMAP dimensionality reduction
            
        Returns:
            TopicModelAnalysis instance with fitted model
            
        Example:
            >>> from gemini_chat_tools import analyze_gemini_chat
            >>> from gemini_chat_tools.topic_model import TopicModelAnalysis
            >>> 
            >>> analysis = analyze_gemini_chat("chat.json")
            >>> timeline = analysis.timeline()
            >>> 
            >>> # Quick start with defaults
            >>> topic_analysis = TopicModelAnalysis.from_timeline(timeline)
            >>> 
            >>> # Custom configuration
            >>> topic_analysis = TopicModelAnalysis.from_timeline(
            ...     timeline,
            ...     preprocess_config={'custom_stopwords': ['gemini', 'claude']},
            ...     bertopic_config={'nr_topics': 10, 'verbose': True},
            ...     umap_config={'n_neighbors': 15, 'min_dist': 0.1}
            ... )
            >>> 
            >>> # Access results
            >>> print(topic_analysis.model.get_topic_info())
            >>> fig = topic_analysis.visualize_topics_over_time()
        """
        # Preprocess if requested
        if preprocess:
            config = preprocess_config or {}
            processed_timeline = preprocess_timeline(timeline, **config)
        else:
            processed_timeline = timeline.copy()
        
        # Extract messages and timestamps
        messages = processed_timeline['text'].tolist()
        # Use timeline index as timestamps (sequential ordering)
        timestamps = processed_timeline.index.tolist()
        
        # Create and fit BERTopic model
        bertopic_params = bertopic_config or {'verbose': True}
        model = BERTopic(**bertopic_params)
        topics, probs = model.fit_transform(messages)
        
        # Compute embeddings if requested
        embeddings = None
        reduced_embeddings = None
        if compute_embeddings:
            sentence_model = SentenceTransformer(embedding_model)
            embeddings = sentence_model.encode(messages, show_progress_bar=False)
            
            # Compute 2D embeddings for visualization
            umap_params = umap_config or {
                'n_neighbors': 10,
                'n_components': 2,
                'min_dist': 0.0,
                'metric': 'cosine'
            }
            reduced_embeddings = UMAP(**umap_params).fit_transform(embeddings)
        
        # Create instance
        instance = cls(
            model=model,
            topics=topics,
            probs=probs,
            timeline=processed_timeline,
            embeddings=embeddings,
            reduced_embeddings=reduced_embeddings
        )
        
        # Compute topics over time
        instance.topics_over_time = model.topics_over_time(messages, timestamps)
        
        return instance
