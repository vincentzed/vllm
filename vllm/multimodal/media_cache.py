# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union
import hashlib
from vllm.utils import LRUCache, GiB_bytes
from vllm.logger import init_logger

logger = init_logger(__name__)

MediaData = Union[object, str, bytes]  # Can be image, audio, video, embeddings, etc.

class MediaCache:
    """
    Global cache for media data, using UUID or content hash as key.
    This allows reusing media data across multiple requests.
    """
    
    def __init__(self, capacity_gb: float = 1.0) -> None:
        self._cache = LRUCache[str, MediaData](
            capacity=int(GiB_bytes * capacity_gb)
        )
    
    def _get_url_hash(self, url: str) -> str:
        """Generate a hash for a URL to use as cache key."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def get(self, uuid: Optional[str], url: Optional[str] = None) -> Optional[MediaData]:
        """
        Get media data from cache.
        
        Args:
            uuid: User-provided UUID for the media
            url: URL of the media (used as fallback key if UUID not provided)
        
        Returns:
            Cached media data if found, None otherwise
        """
        if uuid:
            cache_key = uuid
        elif url:
            cache_key = self._get_url_hash(url)
        else:
            return None
            
        return self._cache.get(cache_key)
    
    def put(self, data: MediaData, uuid: Optional[str] = None, url: Optional[str] = None) -> str:
        """
        Store media data in cache.
        
        Args:
            data: The media data to cache
            uuid: User-provided UUID for the media
            url: URL of the media (used to generate key if UUID not provided)
            
        Returns:
            The cache key used
        """
        if uuid:
            cache_key = uuid
        elif url:
            cache_key = self._get_url_hash(url)
        else:
            raise ValueError("Either uuid or url must be provided")
            
        self._cache[cache_key] = data
        return cache_key
    
    def clear(self) -> None:
        """Clear all cached media data."""
        self._cache.clear()


# Global instance
_global_media_cache = MediaCache()

def get_global_media_cache() -> MediaCache:
    """Get the global media cache instance."""
    return _global_media_cache