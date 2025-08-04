# Contain classes or functions that handles local knowledge management:
# 1. get the YouTube video transcripts and the metadata
# 2. store them in local disk as document database (Mongodb) and vector database (chromadb)
# 3. setup methods that facilitate document search based on user query

import re
from datetime import datetime
from typing import Dict, List, Optional, Callable
from urllib.parse import parse_qs, urlparse

# database (local)
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pymongo import MongoClient, TEXT
from youtube_transcript_api import YouTubeTranscriptApi
import requests

from nlp.text_processing import TextProcessor

class YouTubeTranscriptProcessor:
    def __init__(self,
                 mongodb_uri: str = "mongodb://127.0.0.1:27017",
                 database_name: str = "why_files",
                 chroma_path: str = "./chromadb_wf_docs",
                 embedding_function: Callable = DefaultEmbeddingFunction):
        """
        Processor with multi-vector RAG capabilities

        Args:
            mongodb_uri (str): MongoDB connection URI.
            database_name (str): Name of the document database to use.
            chroma_path (str): Path for chroma vector database
            embedding_function (Optional[Callable]): Function to generate embeddings.
        """

        # mongodb database setup
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]

        # mongodb collections
        self.transcripts_collection = self.db["transcripts"]    # original transcript pulled from the api
        self.doc_metadata_collection = self.db["doc_metadata"]   # document (transcript) metadata
        self.chunks_collection = self.db["chunks"]   # chunks collection (chunk_txt, chunk_id and etc)

        # create indexes for easy search and retrieval
        self.transcripts_collection.create_index("video_id", unique=True)
        self.doc_metadata_collection.create_index("video_id", unique=True)
        # self.chunks_collection.create_index("video_id", unique=True)
        # self.chunks_collection.create_index([("start_time", 1), ("end_time", 1)])

        # chromadb setup
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="wf_transcripts_vectors",
            metadata={"hnsw:space": "cosine"}
        )

        # store embedding function
        self.embedding_function = embedding_function

        # might need text processor
        self.text_processor = TextProcessor()

        self._ensure_text_index()

    def _ensure_text_index(self):
        """Create text search for the transcripts collection"""
        try:
            self.transcripts_collection.create_index(
                [("title", TEXT), ("content", TEXT)],
                default_language="english",
                weights={"title": 10, "content": 5}
            )
        except Exception:
            pass   # index might already exist

    def extract_video_id(self, url: str) -> str:
        """
        Extracts the video ID from a YouTube URL.
        
        Args:
            url (str): YouTube URL.

        Returns:
            str: Extracted video ID.
        """
        # handle different YouTibe URL formats
        if "youtu.be" in url:
            return url.split("youtube.be/")[1].split("?")[0]
        elif "youtube.com/watch" in url:
            parsed_url = urlparse(url)
            return parse_qs(parsed_url.query)["v"][0]
        elif "youtube.com/embed/" in url:
            return url.split("youtube.com/embed/")[1].split("?")[0]
        else:
            # assume it is the video ID
            return url
        
    def get_video_metadata(self, video_id: str) -> Dict:
        """
        Fetch video metadata from YouTube API (title and channel)

        Args:
            video_id (str): YouTube video ID.

        Returns:
            Dict: Video metadata.
        """
        try:
            # use YouTube's oEmbed API for basic info
            url = f"https://www.youtube.com/oembed?url=https%3A//www.youtube.com/watch%3Fv%3D{video_id}&format=json"
            response = requests.get(url, timeout=10)

            response.raise_for_status()
        
            if response.status_code == 200:
                data = response.json()
                return {
                    "title": data["title"],
                    "channel": data["author_name"]
                }
        
        except Exception as e:
            print(f"Error fetching video metadata: {e}")
        
        return {
            "title": "Unknown",
            "channel": "Unknown"
        }
    
    def get_transcript_from_yt(self, video_id: str) -> List[Dict]:
        """
        Fetch transcript from YouTubeTranscriptApi.

        Args:
            video_id (str): YouTube video ID.

        Returns:
            List[Dict]: Transcript segments.
        """
        try:
            return YouTubeTranscriptApi().fetch(video_id).to_raw_data()
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return []
        
    def process_transcript(self, transcript: List[Dict], ads_times: tuple, end_time: float) -> Dict:
        """
        Process the transcript list into a structured format

        Args:
            transcript (List[Dict]): List of transcript segments from YouTubeTranscriptApi.
            ads_times (tuple): Start and end times of ads.
            end_time (float): when to stop processing the transcript

        Returns:
            Dict: Processed transcript.
        """
        full_text = ""
        duration = 0
        for snippet in transcript:
            # start time of snippet
            start_time = snippet["start"]

            # skip the ads time window
            if ads_times[0] <= start_time <= ads_times[1]:
                continue

            # stop when the video transcript reaches a certain timestamp
            if start_time >= end_time:
                break

            # extract the text
            text = snippet["text"].replace(">> ", "").strip()

            full_text += text + " "
            duration += snippet["duration"]

        # further process and clean up text
        full_text = re.sub(r'\s+\[.*?\]', ' ', full_text).strip()

        return {
            'content': full_text,
            'duration': duration,
            'word_count': len(full_text.split()) if full_text else 0
        }
    
    def create_transcript_chunks(self, 
                                 transcript: List[Dict],
                                 ads_times: tuple = (0, 0),
                                 end_time: float = float('inf'),
                                 chunk_duration: float = 180.0,
                                 overlap_duration: float = 20.0):
        """
        Create overlapping chunks from transcript for vector database

        Args:
            transcript (List[Dict]): List of transcript segments.
            ads_times (tuple): Start and end times of ads.
            end_time (float): When to stop processing the transcript.
            chunk_duration (float): Duration of each chunk.
            overlap_duration (float): Overlap duration between chunks.

        Returns:
            List[Dict]: List of chunks with metadata.
        """
        chunks = []
        # chunks_text_list = []

        valid_snippets = [
            s for s in transcript
            if not (ads_times[0] <= s["start"] <= ads_times[1]) and s["start"] < end_time
        ]

        if not valid_snippets:
            return []
        
        current_chunk_snippets = []
        current_duration = 0.0

        for i, snippet in enumerate(valid_snippets):
            current_chunk_snippets.append(snippet)
            current_duration += snippet.get('duration', 0.0)

            # if chunk is long enough, or last snippet, finalize the chunk and update the duration and chunk snippets
            if current_duration >= chunk_duration or i == len(valid_snippets) - 1:
                # join the list of text and clean up
                chunk_text = " ".join(s['text'].replace(">> ", "").strip() for s in current_chunk_snippets)
                clean_text = re.sub(r'\s+', ' ', chunk_text).strip()

                # store chunk in list of dictionary
                if clean_text:
                    start_time = current_chunk_snippets[0]['start']
                    end_time_of_chunk = current_chunk_snippets[-1]['start'] + current_chunk_snippets[-1].get('duration', 0.0)

                    # chunk metadata for chromadb
                    chunks.append({
                        "document": clean_text,
                        "start_time": start_time,
                        "end_time": end_time_of_chunk,
                        "duration": end_time_of_chunk - start_time,
                        "word_count": len(clean_text.split())
                    })

                    # prepare for next chunk: update the current chunk snippets and duration
                    overlap_start_time = end_time_of_chunk - overlap_duration

                    # find the index to slice (keep)
                    slice_index = 0
                    for j, s in enumerate(current_chunk_snippets):
                        if s['start'] >= overlap_start_time:
                             slice_index = j
                             break
                        
                    current_chunk_snippets = current_chunk_snippets[slice_index:]
                    current_duration = sum(s.get('duration', 0.0) for s in current_chunk_snippets)

                else:
                    # reset if chunk was empty
                    current_chunk_snippets = []
                    current_duration = 0.0
                
        return chunks   # documents and their metadata for chromadb
    
    def _store_vector_chunks(self,
                             video_id: str,
                             raw_transcript: List[Dict],
                             video_metadata: Dict,
                             ads_times: tuple,
                             end_time: float,
                             chunk_duration: float,
                             overlap_duration: float, 
                             overwrite: bool):
        """Store transcript chunks with vector embeddings"""
        try:
            # clean up the existing data in mongodb and chromadb if overwriting
            if overwrite:
                results = self.chroma_collection.get(
                    where={'video_id': video_id}
                )

                if results and results["ids"]:
                    print(f"Overwrite enabled: Deleting {len(results['ids'])} existing chunks for video {video_id}")
                    self.chroma_collection.delete(ids=results["ids"])
                    # self.doc_metadata_collection.delete_one({"video_id": video_id})
                    # self.transcripts_collection.delete_one({"video_id": video_id})

            # create chunks
            chunks = self.create_transcript_chunks(raw_transcript, ads_times=ads_times, end_time=end_time,
                                                    chunk_duration=chunk_duration, overlap_duration=overlap_duration)
            
            
            if not chunks:
                print("No chunk created")
                return
            
            doc_record = {
                "video_id": video_id,
                "title": video_metadata.get("title", "Unknown"),
                "channel": video_metadata.get("channel", "Unknown"),
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "created_date": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "metadata": {
                    "chunk_duration": chunk_duration,
                    "overlap_duration": overlap_duration
                }
            }

            self.doc_metadata_collection.replace_one({"video_id": video_id}, doc_record, upsert=True)

            # create the vector data
            vector_data = {
                "ids": [],
                "documents": [],
                "metadatas": []
            }

            print(f"Processing {len(chunks)} chunks for video {video_id}")
            for i, chunk in enumerate(chunks):
                chunk_id = f"{video_id}_chunk_{i}"

                # store the document chunks into chromadb
                try:
                    vector_data["ids"].append(chunk_id)
                    vector_data["documents"].append(chunk["document"])
                    vector_data["metadatas"].append(
                        {
                            "video_id": video_id,
                            "chunk_index": i,
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                            "title": video_metadata.get("title", "Unknown")
                        }
                    )

                except Exception as e:
                    print(f"Error generating embedding for chunk {chunk_id}: {e}")
                    continue

            if vector_data["ids"]:
                self.chroma_collection.add(**vector_data)
                print(f"Stored {len(vector_data['ids'])} vectors for video {video_id}")

        except Exception as e:
            print(f"Error processing chunks for video {video_id}: {e}")
            # rollback on error
            # self.doc_metadata_collection.delete_one({"video_id": video_id})
            # self.transcripts_collection.delete_one({"video_id": video_id})
            raise
    
    def fetch_and_store_transcript(self, 
                                   video_url_or_id: str,
                                   ads_times: tuple = (0, 0), 
                                   end_time: float = float('inf'),
                                   overwrite: bool = False,
                                   enable_vector_storage: bool = True,
                                   chunk_duration: float = 180.0,
                                   overlap_duration: float = 20.0) -> Dict:
        """
        Fetch transcript for a YouTube video and store it in MongoDB and chormaDB.

        Args:
            video_url_or_id (str): YouTube video URL or ID.
            ads_times (tuple): Start and end times of ads.
            end_time (float): When to stop processing the transcript.
            overwrite (bool): Whether to overwrite existing transcript in the db.
            enable_vector_storage (bool): Whether to enable vector storage.
            chunk_duration (float): Duration of each chunk for vector storage
            overlap_duration (float): Overlap duration between chunks for vector storage

        Returns:
            Dict: document metadata
        """
        try:
            # extract video id
            video_id = self.extract_video_id(video_url_or_id)

            # check if transcript already exists
            if not overwrite:
                existing_doc = self.transcripts_collection.find_one({"video_id": video_id})
                if existing_doc:
                    print(f"Transcript for video {video_id} already exists in the database. Use overwrite=True to update it.")
                    return existing_doc
                
            # fetch transcript
            print(f"Fetching raw transcript for video {video_id}")
            raw_transcript = self.get_transcript_from_yt(video_id)

            if not raw_transcript:
                raise Exception(f"No transcript found for video {video_id}")
            
            # get video metadata
            print(f"Fetching video metadata for video {video_id}")
            video_metadata = self.get_video_metadata(video_id)

            # process transcript
            processed_transcript = self.process_transcript(raw_transcript, ads_times=ads_times, end_time=end_time)

            # create transcript dictionary
            transcript_doc = {
                "video_id": video_id,
                "title": video_metadata.get("title", "Unknown"),
                "content": processed_transcript["content"],
                "duration": processed_transcript["duration"],
                "word_count": processed_transcript["word_count"]
            }

            # store in mongodb collection
            result = self.transcripts_collection.replace_one(
                {"video_id": video_id},
                transcript_doc,
                upsert=True
            )

            # vector storage if enabled and embedding function provided (TODO)
            if enable_vector_storage:
                print(f"Storing vectors for video {video_id}")
                # update vector database and mongodb collection (doc_metadata)
                self._store_vector_chunks(
                    video_id=video_id,
                    raw_transcript=raw_transcript,
                    video_metadata=video_metadata,
                    ads_times=ads_times,
                    end_time=end_time,
                    chunk_duration=chunk_duration,
                    overlap_duration=overlap_duration,
                    overwrite=overwrite
                )

            print(f"Successfully stored transcript for {video_metadata.get('title', 'Unknown')}")
            return transcript_doc    

        except Exception as e:
            print(f"Error fetching and storing transcript for video {video_url_or_id}: {e}")
            raise

    def semantic_search(self, 
                        query: str,
                        n_results: int = 5,
                        where: Optional[Dict] = None,
                        video_id_filter: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search across transcript chunks

        Args:
            query (str): search query
            n_results (int): number of search results (chunks)
            where (Optional[Dict]): metadata filters
            video_id_filter (Optional[Dict]): video id filters

        Returns:
            List[Dict]: search results with video and chunk info
        """
        if not self.embedding_function:
            raise ValueError("Embedding function not provided for semantic search")
        
        # build metadata filter
        query_where = {}
        if where:
            query_where.update(where)
        if video_id_filter:
            query_where["video_id"] = video_id_filter

        try:
            chroma_results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=query_where if query_where else None
            )

            if not chroma_results or not chroma_results.get("ids"):
                return []

            enriched_results = []

            for i, chunk_id in enumerate(chroma_results["ids"][0]):
                # get transcript data
                metadata = chroma_results["metadatas"][0][i]
                video_id = metadata["video_id"]
                # in semantic search, return the chunk text
                chunk_text = chroma_results["documents"][0][i]

                similarity_score = 1 / (1.0 + chroma_results["distances"][0][i])

                result = {
                    "chunk_id": chunk_id,
                    "video_id": video_id,
                    "text": chunk_text,
                    "similarity": similarity_score,
                    "start_time": metadata.get("start_time"),
                    "title": metadata.get("title", "Unknown")
                }
                enriched_results.append(result)

            return enriched_results
        
        except Exception as e:
            print(f"Error performing semantic search: {e}")
            return []
        
    def safe_text_search(self, query: str, limit: int = 2) -> List[Dict]:
        """Perform safe text search with fallback strategies
        
        Args:
            query (str): search query
            limit (int): maximum number of results to return
        
        Returns:
            List of search results    
        """
        results = []

        # try preprocessed query first
        processed_query = self.text_processor.preprocess_text_for_search(query)

        if not processed_query:
            print("Query preprocessing results in empty string. Skipping text search")
            return []
        
        # strategy 1: try full text search with preprocessed query
        try:
            search_query = {"$text": {"$search": processed_query}}
            projection = {"score": {"$meta": "textScore"}}

            cursor = (
                self.transcripts_collection.find(search_query, projection)
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )

            results = list(cursor)
            if results:
                print("Text search successful with proprocessed query") 
                return results
            
        except Exception as e:
            print(f"Full text search failed: {e}")

        # strategy 2: search word by word
        words = processed_query.split()
        if len(words) > 1:
            try:
                # search for each word and combine results
                word_queries = []
                for word in words:
                    word_queries.append({"$text": {"$search": word}})

                # use $or to find documents matching any of the words
                search_query = {"$or": word_queries}
                projection = {"score": {"$meta": "textScore"}}

                cursor = (
                    self.transcripts_collection.find(search_query, projection)
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(limit * 2)   # get more results to account for duplicates
                )

                # deduplicate results by video id
                seen_videos = set()
                unique_results = []
                for doc in cursor:
                    if doc["video_id"] not in seen_videos:
                        seen_videos.add(doc["video_id"])
                        unique_results.append(doc)

                if unique_results:
                    print("Indivdiual words search successful")
                    return unique_results
            
            except Exception as e:
                print(f"Invididual words search failed: {e}")

        # strategy 3: Regex search
        try:
            # create case insensitive regex pattern
            regex_pattern = "|".join([re.escape(word) for word in words])
            search_query = {
                "$or": [
                    {"title": {"$regex": regex_pattern, "$options": "i"}},
                    {"content": {"$regex": regex_pattern, "$options": "i"}}
                ]
            }

            cursor = (
                self.transcripts_collection.find(search_query)
                .limit(limit)
            )

            results = list(cursor)
            if results:
                print("Fallback regex search successful")
                # for consistency
                for result in results:
                    result["score"] = 1.0
                return results
        
        except Exception as e:
            print(f"Fallback regex search failed: {e}")

        print("All text search failed!")
        return []
    
    def hybrid_search(self, 
                      query: str,
                      limit: int = 2,
                      text_weight: float = 0.3,
                      semantic_weight: float = 0.7,
                      enable_text_search: bool = True,
                      enable_semantic_search: bool = True) -> List[Dict]:
        """
        Combine text and semantic search results

        Args:
            query: Search query
            text_weight: Weight for text search results
            semantic_weight: Weight for semantic search results
            limit: Number of final results
            enable_text_search: Whether to enable text search
            enable_semantic_search: Whether to enable semantic search
            
        Returns:
            Combined and reranked results        
        """
        if not query.strip():
            print("Query is empty for hybrid search!")
            return []
        
        results = {}

        # get text search results
        # Get text search results
        if enable_text_search:
            try:
                text_results = self.safe_text_search(query, limit * 2)
                for result in text_results:
                    video_id = result["video_id"]
                    score = result.get("score", 0) * text_weight
                    
                    if video_id not in results:
                        results[video_id] = {
                            "video_id": video_id,
                            "title": result["title"],
                            "content": result["content"],
                            "combined_score": score,
                            "text_score": result.get("score", 0),
                            "semantic_score": 0
                        }
                    else:
                        results[video_id]["combined_score"] += score
                        results[video_id]["text_score"] = max(result.get("score", 0), results[video_id]["text_score"])
            except Exception as e:
                print(f"Error in text search: {e}")

        # Get semantic search results
        if enable_semantic_search:
            try:
                semantic_results = self.semantic_search(query, limit * 2)
                for result in semantic_results:
                    video_id = result["video_id"]
                    score = result["similarity"] * semantic_weight
                    
                    if video_id not in results:
                        results[video_id] = {
                            "video_id": video_id,
                            "title": result["title"],
                            "content": result["content"],
                            "combined_score": score,
                            "text_score": 0,
                            "semantic_score": result["similarity"]
                        }
                    else:
                        results[video_id]["combined_score"] += score
                        results[video_id]["semantic_score"] = max(
                            results[video_id]["semantic_score"],
                            result["similarity"]
                        )
            except Exception as e:
                print(f"Error in semantic search: {e}")

        if not results:
            print("No results from both text and semantic search!")

        # Sort by combined score and return top results
        sorted_results = sorted(
            results.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        return sorted_results[:limit]