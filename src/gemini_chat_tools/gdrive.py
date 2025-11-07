"""Google Drive integration for fetching file metadata.

This module provides functions to fetch metadata for files referenced in
Gemini chat conversations via Google Drive API.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path


def get_drive_service():
    """Initialize and return a Google Drive API service object.
    
    This function handles OAuth2 authentication using stored credentials.
    On first run, it will open a browser for authorization.
    
    Returns:
        Google Drive API service object
        
    Raises:
        ImportError: If google-auth or google-api-python-client not installed
        FileNotFoundError: If gdrive_credentials.json not found
        
    Note:
        Requires gdrive_credentials.json in the current directory.
        Creates token.json to store access/refresh tokens.
        
    Example:
        >>> service = get_drive_service()
        >>> # Use service to fetch file metadata
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        raise ImportError(
            "Google Drive API dependencies not installed. "
            "Install with: uv add google-auth-oauthlib google-api-python-client"
        ) from e
    
    # Scope for read-only metadata access
    SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    
    creds = None
    
    # Load existing credentials from token.json
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Need gdrive_credentials.json from Google Cloud Console
            if not os.path.exists("gdrive_credentials.json"):
                raise FileNotFoundError(
                    "gdrive_credentials.json not found. "
                    "Download OAuth 2.0 credentials from Google Cloud Console."
                )
            
            flow = InstalledAppFlow.from_client_secrets_file(
                "gdrive_credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    
    return build("drive", "v3", credentials=creds)


def get_file_metadata(service, file_id: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata for a specific Google Drive file.
    
    Args:
        service: Google Drive API service object from get_drive_service()
        file_id: Google Drive file ID
        
    Returns:
        Dictionary with file metadata:
            - id: File ID
            - name: File name
            - mimeType: MIME type
            - createdTime: Creation timestamp
            - modifiedTime: Last modification timestamp
        Returns None if file not found or error occurs
        
    Example:
        >>> service = get_drive_service()
        >>> metadata = get_file_metadata(service, "1h-8ZvgE_hHPSisPGUoO8qsvP1CkQ9BwM")
        >>> print(metadata['name'])
        'paper-reviews_special-issue.md'
    """
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        return None
    
    try:
        file_metadata = (
            service.files()
            .get(fileId=file_id, fields="id, name, mimeType, createdTime, modifiedTime")
            .execute()
        )
        return file_metadata
    except HttpError as error:
        print(f"Error fetching file {file_id}: {error}")
        return None


def get_files_metadata_batch(service, file_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch metadata for multiple files.
    
    Args:
        service: Google Drive API service object
        file_ids: List of Google Drive file IDs
        
    Returns:
        Dictionary mapping file_id -> metadata dict
        Files that couldn't be fetched are excluded
        
    Example:
        >>> service = get_drive_service()
        >>> file_ids = ['1h-8ZvgE...', '1H9lLqS...']
        >>> metadata_map = get_files_metadata_batch(service, file_ids)
        >>> for file_id, metadata in metadata_map.items():
        >>>     print(f"{metadata['name']}: {metadata['mimeType']}")
    """
    metadata_map = {}
    
    for file_id in file_ids:
        metadata = get_file_metadata(service, file_id)
        if metadata:
            metadata_map[file_id] = metadata
    
    return metadata_map


def get_conversation_files_metadata(analysis, service=None) -> Dict[int, List[Dict[str, Any]]]:
    """Fetch metadata for all files used in a conversation.
    
    This is a convenience function that combines analysis.files_used with
    Drive API calls to get file names and metadata.
    
    Args:
        analysis: ChatAnalysis object from analyze_gemini_chat()
        service: Optional pre-initialized Drive service. If None, will create one.
        
    Returns:
        Dictionary mapping chunk_index -> list of file metadata dicts
        Each metadata dict contains: id, name, mimeType, createdTime, modifiedTime
        
    Example:
        >>> from gemini_chat_tools import analyze_gemini_chat
        >>> from gemini_chat_tools.gdrive import get_conversation_files_metadata
        >>> 
        >>> analysis = analyze_gemini_chat("chat.json")
        >>> files_metadata = get_conversation_files_metadata(analysis)
        >>> 
        >>> for chunk_idx, files in files_metadata.items():
        >>>     print(f"Message {chunk_idx}:")
        >>>     for file in files:
        >>>         print(f"  - {file['name']}")
    """
    if service is None:
        service = get_drive_service()
    
    # Get all unique file IDs
    all_file_ids = set()
    for file_ids in analysis.files_used.values():
        all_file_ids.update(file_ids)
    
    # Fetch metadata for all files
    metadata_map = get_files_metadata_batch(service, list(all_file_ids))
    
    # Build result mapping chunk_index -> list of metadata
    result = {}
    for chunk_idx, file_ids in analysis.files_used.items():
        result[chunk_idx] = [
            metadata_map[file_id] 
            for file_id in file_ids 
            if file_id in metadata_map
        ]
    
    return result


def format_file_names(files_metadata: List[Dict[str, Any]], max_length: int = 50) -> str:
    """Format file names for display in visualizations.
    
    Args:
        files_metadata: List of file metadata dicts
        max_length: Maximum length for each file name (truncate if longer)
        
    Returns:
        Formatted string with file names, one per line
        
    Example:
        >>> files = [
        >>>     {'name': 'paper-reviews.md'},
        >>>     {'name': 'very-long-filename-that-needs-truncation.csv'}
        >>> ]
        >>> print(format_file_names(files, max_length=30))
        paper-reviews.md
        very-long-filename-that-nee...
    """
    lines = []
    for file_meta in files_metadata:
        name = file_meta.get('name', 'unknown')
        if len(name) > max_length:
            name = name[:max_length-3] + '...'
        lines.append(name)
    return '\n'.join(lines)


__all__ = [
    'get_drive_service',
    'get_file_metadata',
    'get_files_metadata_batch',
    'get_conversation_files_metadata',
    'format_file_names',
]
