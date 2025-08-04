# import json
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# third-party libraries
# from ollama import chat, ListResponse, list as ollama_list
import wikipediaapi
from ddgs import DDGS

from nlp.text_processing import TextProcessor

# configure logging to monitor tool usage by LLM
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enums and dataclass ---
class SearchProvider(Enum):
    """Enumeration of available search providers"""
    DDGSWEB = "web_search"
    DDGSNEWS = "news_search"
    WIKIPEDIA = "wikipedia"

@dataclass
class SearchResult:
    """Standardized search result format"""
    title: str
    url: str
    snippet: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@runtime_checkable
class SearchProviderProtocol(Protocol):
    """Protocol for search providers to ensure consistent interfaces"""
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform search and returns standardized results"""
        ...

class BaseSearchProvider(ABC):
    """Base class for search providers"""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform search and returns standardized results"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the search provider"""
        pass

# --- Search Provider implementations ---
class WebSearchProvider(BaseSearchProvider):
    """Web search from ddgs"""
    def __init__(self, ddgs_client: DDGS, text_processor: TextProcessor):
        # initialize the DDGS client
        self._ddgs = ddgs_client
        self.text_processor = text_processor

    @property
    def provider_name(self) -> str:
        return SearchProvider.DDGSWEB.value
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search for webpages"""
        try:
            processed_query = self.text_processor.preprocess_text_for_search(query, search_type='ddgs')
            logger.info(f"Search the web for '{processed_query}'")
            search_results = self._ddgs.text(processed_query, max_results=max_results, safesearch='off')

            return [
                SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    source=self.provider_name,
                    metadata={'accessed_date': datetime.now().isoformat()}
                )
                for result in search_results
            ]
        
        except Exception as e:
            logger.error(f"Error in ddgs web search: {e}", exc_info=True)
            return []
        
class NewsSearchProvider(BaseSearchProvider):
    """News search from ddgs"""
    def __init__(self, ddgs_client: DDGS, text_processor: TextProcessor):
        # initialize the DDGS client
        self._ddgs = ddgs_client
        self.text_processor = text_processor

    @property
    def provider_name(self) -> str:
        return SearchProvider.DDGSNEWS.value
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search for webpages"""
        try:
            processed_query = self.text_processor.preprocess_text_for_search(query, search_type='ddgs')
            logger.info(f"Search the news for '{processed_query}'")
            search_results = self._ddgs.text(processed_query, max_results=max_results, safesearch='off')

            return [
                SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    source=self.provider_name,
                    metadata={'accessed_date': datetime.now().isoformat(),
                              'news_date': result.get("date", ''),
                              'news_source': result.get("source", '')}
                )
                for result in search_results
            ]
        
        except Exception as e:
            logger.error(f"Error in ddgs web search: {e}", exc_info=True)
            return []
        
class WikipediaProvider(BaseSearchProvider):
    """Search provider using wikipedia API"""
    def __init__(self, text_processor: TextProcessor, user_agent: str = 'WikiSearch (jacky@example.com)', language: str = 'en'):
        # create client once for reuse
        self._wiki_api = wikipediaapi.Wikipedia(user_agent=user_agent, language=language)
        self.text_processor = text_processor

    @property
    def provider_name(self) -> str:
        return SearchProvider.WIKIPEDIA.value
    
    def search(self, query: str, max_results: int = 1) -> List[SearchResult]:
        """Search using Wikipedia API. Note that wikipedia API doesn't support 'max_results' as ddgs;
        it fetches only one page"""
        try:
            processed_query = self.text_processor.preprocess_text_for_search(query, search_type='ddgs')
            logger.info(f'Search the wikedia article for {processed_query}')
            page = self._wiki_api.page(query)

            if not page.exists():
                logger.warning(f"No Wikipedia page found for {query}")
                return []
            
            return [
                SearchResult(
                    title=page.title,
                    url=page.fullurl if page.fullurl else '',
                    snippet=f"{page.summary[:1200]}..." if len(page.summary) > 1200 else page.summary,
                    source=self.provider_name,
                    metadata={'accessed_date': datetime.now().isoformat()}
                )
            ]

        except Exception as e:
            logger.error(f"Error in wikipedia web search: {e}", exc_info=True)
            return []
        
# --- Search management --- 
class SearchManager:
    """Manager class for handling multiple search providers"""
    def __init__(self):
        """Use dictionary to match Enum provider instances"""
        # create a shared clients and injects them into the providers

        # create shared instances once
        shared_ddgs_client = DDGS()
        shared_text_processor = TextProcessor()

        self.providers: Dict[SearchProvider, BaseSearchProvider] = {
            SearchProvider.DDGSWEB: WebSearchProvider(
                ddgs_client=shared_ddgs_client,
                text_processor=shared_text_processor
            ),
            SearchProvider.DDGSNEWS: NewsSearchProvider(
                ddgs_client=shared_ddgs_client,
                text_processor=shared_text_processor
            ),
            SearchProvider.WIKIPEDIA: WikipediaProvider(
                text_processor=shared_text_processor
            )
        }

    def add_provider(self, provider_enum: SearchProvider, provider_instance: BaseSearchProvider):
        """Add a new search provider"""
        logger.info(f"Adding / updating provider: {provider_enum.value}")
        self.providers[provider_enum] = provider_instance

    def search(self, provider: SearchProvider, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search using the specified provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not found")
        return self.providers[provider].search(query, max_results)
    
    def multi_search(self, query: str, providers: List[SearchProvider] = [], max_results_per_provider: int = 5) -> Dict[str, List[SearchResult]]:
        """Search using multiple providers in parallel"""
        if not providers:
            providers = list(self.providers.keys())
        
        results: Dict[str, List[SearchResult]] = {}
        # Use ThreadPool Executor to perform network bound search concurrently
        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            future_to_provider = {
                executor.submit(self.search, provider, query, max_results_per_provider): provider
                for provider in providers if provider in self.providers
            }

            for future in as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    provider_results = future.result()
                    results[provider.value] = provider_results
                except Exception as e:
                    logger.error(f"Error in multi_search provided by {provider.value}: {e}", exc_info=True)
                    results[provider.value] = []
        
        return results