# text processing class to process the text for mongodb, wikipedia and ddgs search
# import string
# import urllib.parse
import re
from typing import List, Optional, Set
from nltk.corpus import stopwords
import nltk

class TextProcessor:
    """
    Custom text preprocessing workflows for ddgs, wikipedia search and 
    indexing mongodb documents: 
    1. URL & email removal
    2. Punctuation removal
    3. Tokenization
    4. Stopword removal
    5. named entities recognition (NER)
    6. Part-of-speech (POS) tagging
    """
    def __init__(self, stopwords_set: Optional[Set[str]] = None):
        if stopwords is None:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = stopwords_set

        if not self.stopwords:
            self.stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }

    def clean_text(self, text: str) -> str:
        """Remove URLs, emails, punctuations"""
        # remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
        # remove emails
        text = re.sub(r'\S+@\S+', '', text)
    
        # remove most punctuations except hypens and underscores
        text = re.sub(r"[^\w\s\-_]", '', text)

        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def tokenize(self, text: str) -> List[str]:
        # word tokenization
        return nltk.word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def process_text(self, text: str) -> List[str]:
        tokens = self.tokenize(self.clean_text(text))
        return self.remove_stopwords(tokens)
    
    def preprocess_text_for_search(self, text: str, search_type: str = "") -> str:
        """
        Full text processing pipelines for ddgs and wikipedia search

        Args:
            text (str): input text
            type (str): ddgs or wiki
        """
        # tokenize, POS tagging and NER
        tokens = self.process_text(text)
        pos_tags = nltk.pos_tag(tokens)
        entities = nltk.ne_chunk(pos_tags, binary=False)

        named_entities = []
        for entity in entities:
            if isinstance(entity, nltk.Tree):
                # named entities (e.g. person, organizations, buildings, etc)
                entity_text = " ".join(word for word, _ in entity.leaves())
                named_entities.append(entity_text)

        if named_entities:
            search_query = " ".join(named_entities)

        else:
            nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
            if nouns:
                search_query = " ".join(nouns)
            else:
                search_query = " ".join(tokens)

        # # URL
        # if search_type == 'ddgs':
        #     return urllib.parse.quote_plus(search_query)
        
        if search_type == 'wiki':
            title = search_query.replace(' ', '_')
            # capitalize first character
            if title:
                title = title[0].upper() + title[1:]

            return title
        
        else:
            return search_query