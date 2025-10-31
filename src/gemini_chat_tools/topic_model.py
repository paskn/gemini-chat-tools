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
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about all topics.
        
        Returns:
            DataFrame with topic information (topic ID, count, words)
        """
        return self.model.get_topic_info()
    
    def get_topic(self, topic_id: int) -> List[Tuple[str, float]]:
        """Get words and weights for a specific topic.
        
        Args:
            topic_id: Topic ID to query
            
        Returns:
            List of (word, weight) tuples
        """
        return self.model.get_topic(topic_id)
    
    def get_topic_freq(self, topic_id: int) -> int:
        """Get frequency (document count) for a specific topic.
        
        Args:
            topic_id: Topic ID to query
            
        Returns:
            Number of documents assigned to this topic
        """
        return self.model.get_topic_freq(topic_id)
    
    def visualize_topics_over_time(self, **kwargs):
        """Visualize how topics evolve over the conversation.
        
        Args:
            **kwargs: Additional arguments passed to BERTopic.visualize_topics_over_time()
            
        Returns:
            Plotly Figure object (can be displayed or saved with .write_html())
            
        Example:
            >>> fig = topic_analysis.visualize_topics_over_time()
            >>> fig.write_html("topics_over_time.html")
        """
        if self.topics_over_time is None:
            messages = self.timeline['text'].tolist()
            timestamps = self.timeline.index.tolist()
            self.topics_over_time = self.model.topics_over_time(messages, timestamps)
        
        return self.model.visualize_topics_over_time(self.topics_over_time, **kwargs)
    
    def visualize_document_datamap(self, format: str = 'interactive', **kwargs):
        """Visualize documents in 2D embedding space.
        
        Creates a 2D scatter plot of documents colored by topic using datamapplot.
        
        Args:
            format: 'interactive' (default) or 'static'
            **kwargs: Additional arguments passed to BERTopic.visualize_document_datamap()
            
        Returns:
            Matplotlib Figure (static) or Plotly Figure (interactive)
            
        Example:
            >>> fig = topic_analysis.visualize_document_datamap()
            >>> fig.savefig("document_map.svg", bbox_inches="tight")
        """
        if self.reduced_embeddings is None:
            raise ValueError(
                "Embeddings not computed. Set compute_embeddings=True when creating TopicModelAnalysis."
            )
        
        messages = self.timeline['text'].tolist()
        return self.model.visualize_document_datamap(
            messages,
            reduced_embeddings=self.reduced_embeddings,
            **kwargs
        )
    
    def visualize_barchart(self, n_words: int = 9, **kwargs):
        """Visualize top words per topic as bar charts.
        
        Args:
            n_words: Number of words to show per topic (max 9 recommended)
            **kwargs: Additional arguments passed to BERTopic.visualize_barchart()
            
        Returns:
            Plotly Figure object
            
        Example:
            >>> fig = topic_analysis.visualize_barchart(n_words=7)
            >>> fig.write_html("topic_barchart.html")
        """
        return self.model.visualize_barchart(n_words=n_words, **kwargs)
    
    def visualize_documents(self, **kwargs):
        """Create interactive 3D visualization of documents.
        
        Requires WebGL to display properly in browser.
        
        Args:
            **kwargs: Additional arguments passed to BERTopic.visualize_documents()
            
        Returns:
            Plotly Figure object
            
        Example:
            >>> fig = topic_analysis.visualize_documents()
            >>> fig.write_html("document_visualization.html")
        """
        if self.reduced_embeddings is None:
            raise ValueError(
                "Embeddings not computed. Set compute_embeddings=True when creating TopicModelAnalysis."
            )
        
        messages = self.timeline['text'].tolist()
        return self.model.visualize_documents(
            messages,
            reduced_embeddings=self.reduced_embeddings,
            **kwargs
        )
    
    def visualize_heatmap(self, **kwargs):
        """Visualize topic similarity as a heatmap.
        
        Args:
            **kwargs: Additional arguments passed to BERTopic.visualize_heatmap()
            
        Returns:
            Plotly Figure object
            
        Example:
            >>> fig = topic_analysis.visualize_heatmap()
            >>> fig.write_html("topic_heatmap.html")
        """
        return self.model.visualize_heatmap(**kwargs)
    
    def visualize_topics(self, **kwargs):
        """Visualize intertopic distance map.
        
        Args:
            **kwargs: Additional arguments passed to BERTopic.visualize_topics()
            
        Returns:
            Plotly Figure object
            
        Example:
            >>> fig = topic_analysis.visualize_topics()
            >>> fig.write_html("intertopic_distance.html")
        """
        return self.model.visualize_topics(**kwargs)
    
    def visualize_topics_per_class(self, classes: Optional[List[str]] = None, **kwargs):
        """Visualize topic distribution across classes (e.g., user vs model).
        
        Args:
            classes: List of class labels for each document. If None, uses 'role' from timeline.
            **kwargs: Additional arguments passed to BERTopic.visualize_topics_per_class()
            
        Returns:
            Plotly Figure object
            
        Example:
            >>> # Compare user vs model topics
            >>> fig = topic_analysis.visualize_topics_per_class()
            >>> fig.write_html("topics_per_role.html")
        """
        messages = self.timeline['text'].tolist()
        
        if classes is None:
            # Use role from timeline by default
            classes = self.timeline['role'].tolist()
        
        topics_per_class = self.model.topics_per_class(messages, classes)
        return self.model.visualize_topics_per_class(topics_per_class, **kwargs)
    
    def save_all_visualizations(
        self,
        output_dir: str = "topic_visualizations",
        formats: Dict[str, str] = None,
        n_words: int = 9
    ) -> Dict[str, Path]:
        """Generate and save all visualizations to a directory.
        
        Args:
            output_dir: Directory to save visualizations (created if doesn't exist)
            formats: Dict mapping visualization names to file extensions.
                    Default: all HTML for interactive plots, SVG for datamap
            n_words: Number of words for barchart visualization
            
        Returns:
            Dict mapping visualization names to saved file paths
            
        Example:
            >>> saved_files = topic_analysis.save_all_visualizations(
            ...     output_dir="my_analysis",
            ...     formats={'topics_over_time': 'html', 'document_datamap': 'svg'}
            ... )
            >>> print(f"Saved {len(saved_files)} visualizations")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Default formats (interactive HTML, static SVG for datamap)
        if formats is None:
            formats = {
                'topics_over_time': 'html',
                'document_datamap': 'svg',
                'barchart': 'html',
                'documents': 'html',
                'heatmap': 'html',
                'topics': 'html',
                'topics_per_class': 'html'
            }
        
        saved_files = {}
        
        # Topics over time
        if 'topics_over_time' in formats:
            fig = self.visualize_topics_over_time()
            file_path = output_path / f"topics_over_time.{formats['topics_over_time']}"
            fig.write_html(str(file_path))
            saved_files['topics_over_time'] = file_path
        
        # Document datamap
        if 'document_datamap' in formats and self.reduced_embeddings is not None:
            fig = self.visualize_document_datamap()
            file_path = output_path / f"document_datamap.{formats['document_datamap']}"
            if formats['document_datamap'] == 'svg':
                fig.savefig(str(file_path), bbox_inches="tight")
            else:
                fig.write_html(str(file_path))
            saved_files['document_datamap'] = file_path
        
        # Barchart
        if 'barchart' in formats:
            fig = self.visualize_barchart(n_words=n_words)
            file_path = output_path / f"barchart.{formats['barchart']}"
            fig.write_html(str(file_path))
            saved_files['barchart'] = file_path
        
        # Documents (interactive)
        if 'documents' in formats and self.reduced_embeddings is not None:
            fig = self.visualize_documents()
            file_path = output_path / f"documents.{formats['documents']}"
            fig.write_html(str(file_path))
            saved_files['documents'] = file_path
        
        # Heatmap
        if 'heatmap' in formats:
            fig = self.visualize_heatmap()
            file_path = output_path / f"heatmap.{formats['heatmap']}"
            fig.write_html(str(file_path))
            saved_files['heatmap'] = file_path
        
        # Topics (intertopic distance)
        if 'topics' in formats:
            fig = self.visualize_topics()
            file_path = output_path / f"intertopic_distance.{formats['topics']}"
            fig.write_html(str(file_path))
            saved_files['topics'] = file_path
        
        # Topics per class
        if 'topics_per_class' in formats:
            fig = self.visualize_topics_per_class()
            file_path = output_path / f"topics_per_class.{formats['topics_per_class']}"
            fig.write_html(str(file_path))
            saved_files['topics_per_class'] = file_path
        
        return saved_files


__all__ = ['TopicModelAnalysis', 'preprocess_timeline']
