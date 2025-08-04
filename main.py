# main entry point of chat app

import asyncio
from typing import List, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
from ollama import AsyncClient, ChatResponse

# custom modules
from local_knowledge.youtube_transcripts import YouTubeTranscriptProcessor
from local_chatbot.ollama_chatbot import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IntelligentChatBot:
    """
    A chatbot that integrates Ollama for language model interactions,
    YouTube transcript processing for local knowledge, and web search
    for external information.
    """
    def __init__(self,
                 model_name: str = "llama3.2",
                 summarizer_name: str = "phi4-mini",
                 mongodb_uri: str = "mongodb://127.0.0.1:27017",
                 database_name: str = "why_files",
                 chroma_path: str = "./chromadb_wf_docs", 
                 local_search_limit: int = 2):
        """
        Initialize the chatbot

        Args:
            model_name (str): Name of the Ollama model to use.
            mongodb_uri (str): MongoDB connection URI.
            database_name (str): Name of the document database to use.
            chroma_path (str): Path for chroma vector database
            local_search_limit (int): Maximum number of local search results (documents retrieved)
            similarity_threshold (float): Minimum similarity score for local results
        """
        self.model_name = model_name
        self.summarizer_model_name = summarizer_name
        self.local_search_limit = local_search_limit
        # self.similarity_threshold = similarity_threshold

        # initialize components
        logger.info("Initializing chatbot components...")

        # initialize Ollama client with web search capabilities
        self.chat_client = OllamaClient(model=self.model_name, stream=True)
        self.summarizer_client = AsyncClient()

        # initialize transcript processor with embedding function
        self.transcript_processor = YouTubeTranscriptProcessor(
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            chroma_path=chroma_path,
        )

        # System prompt for the LLM
        self.system_prompt = """
        You are an intelligent assistant with access to both local knowledge and web search capabilities.

        Your workflow:
        1. First, examine any local documents/transcripts provided to you
        2. If the local content contains sufficient information to answer the user's query, use it as your primary source
        3. If the local content is insufficient, irrelevant, or missing, use the available search tools:
        - web_search: for web information and general content
        - news_search: for current events and trends
        - wikipedia_search: for encyclopedic information and detailed explanations

        Guidelines:
        - Be honest about your sources - clearly indicate when you're using local knowledge vs web search
        - If you cannot find relevant information from either source, say so clearly

        Remember to provide comprehensive and accurate responses while being transparent about your information sources.
        """
        logger.info("Initialization complete.")

    def search_local_knowledge(self, query: str) -> tuple[List[Dict], str]:
        """
        Search local knowledge base and returns results with context

        Args:
            query (str): User query
        
        Returns:
            Tuple of (search_results, formatted_context)
        """
        try:
            logger.info("Searching local knowledge base...")
            
            # perform hybrid search
            results = self.transcript_processor.hybrid_search(
                query=query,
                limit=self.local_search_limit
            )

            if not results:
                logger.info("No local results found.")
                return [], ""
            
            # format context for LLM
            context_parts = []
            context_parts.append("=== LOCAL KNOWLEDGE BASE RESULTS ===")

            for i, result in enumerate(results, start=1):
                context_parts.append(f"Document {i}:")
                context_parts.append(f"Title: {result.get('title', 'Unknown')}")
                context_parts.append(f"Content: {result.get('content', '')}")
                context_parts.append("---")

            context_parts.append("=== END OF LOCAL RESULTS ===\n")

            formatted_context = "\n".join(context_parts)
            return results, formatted_context
        
        except Exception as e:
            logger.error(f"Error searching local knowledge: {e}")
            return [], ""
        
    async def summarize_history(self, history: List[Dict]) -> str:
        """
        Summarizes the conversation history using another LLM

        Args:
            history (List[Dict]): A list of conversation turns ({'role': ..., 'content': ...})

        Returns:
            str: Summarized conversation history
        """
        if not history:
            return ""
        
        logger.info("Summarizing conversation history...")
        history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

        summary_prompt = f"""
        Please summarize the following conversation. Condense the key points and the main topic of discussion into
        a concise paragraph. This summary will be used as context for another AI.

        Conversation:
        {history_text}
        ---

        CONCISE SUMMARY:
        """

        messages = [{"role": "user", "content": summary_prompt}]

        try:
            response: ChatResponse = await self.summarizer_client.chat(
                model=self.summarizer_model_name, 
                messages=messages,
                options={
                    "temperature": 0.1,    # prevent hallucination
                    "num_predict": 500     # prevent verbose outputs
                }    
            )

            if response and response.get('message', {}).get('content'):
                return response['message']['content'].strip()
            else:
                logger.warning("Failed to summarize history")
                return ""
            
        except Exception as e:
            logger.error(f"Error summarizing history: {e}")
            return ""
        
    def create_enhanced_messages(self, query: str, local_context: str, history_summary: str) -> List[Dict[str, str]]:
        """
        Create message list with system prompt, local context, and history summary

        Args:
            query (str): User query
            local_context (str): Context from local knowledge
            history_summary (str): Summary of conversation history

        Returns:
            List[Dict[str, str]]: List of messages
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        user_content_parts = []

        if history_summary:
            user_content_parts.append(
                "Here is the summary of previous conversation:\n" + history_summary
            )

        if local_context:
            user_content_parts.append(
                "Here is the context from local knowledge base:\n" + local_context + "\n"
            )
        else:
            user_content_parts.append(
                "No relevant info from local knowledge base\n"
            )

        user_content_parts.append(f"Based on all the above info and a web search (if necessary), please answer the new question:\n{query}")
        
        messages.append({"role": "user", "content": "\n".join(user_content_parts)})
        return messages
    
    async def chat(self, user_query: str, history: List[Dict]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main async chat generator that handles the chat workflow and stream the response:
        1. Search local knowledge
        2. Summarize history
        3. Create enhanced messages
        4. Send to Ollama for response

        Args:
            user_query (str): User's question
            history (str): Conversation history (optional)

        Yields:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()

        try:
            # step 1: RAG (local knowledge search): can be run in executor if slow
            local_results, local_context = self.search_local_knowledge(user_query)

            # step 2: Summarize history asynchronously
            if history:
                history_summary_task = asyncio.create_task(self.summarize_history(history))
                history_summary = await history_summary_task
            else:
                history_summary = ""

            # step 3: Create enhanced messages
            enhanced_messages = self.create_enhanced_messages(user_query, local_context, history_summary)

            # step 4: Send to Ollama for streaming response
            response_stream = self.chat_client.chat( 
                messages=enhanced_messages,
                options_kwargs={
                    "temperature": 0.4,
                    "top_k": 0.9
                }
            )

            full_answer = ""
            for chunk in response_stream:
                if chunk.get("error"):
                    yield {"type": "error", "data": chunk}
                    return
                
                if content_piece := chunk.get("message", {}).get("content"):
                    full_answer += content_piece
                    yield {"type": "content", "data": content_piece}

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            yield {
                "type": "finale",
                "data": {
                    "success": True,
                    "answer": full_answer,
                    "response_time": response_time,
                    "local_results_count": len(local_results),
                    "local_sources":[r.get('title', 'Unknown') for r in local_results]
                }
            }

        except Exception as e:
            logger.error(f"Error in chat method: {e}")
            yield {
                "type": "error",
                "data": {
                    "success": False,
                    "error": str(e)
                }
            }
        
    async def interactive_chat(self):
        """
        Interactive chat loop with memory management
        """
        print("=" * 60)
        print("🤖 Intelligent Chatbot with Local & Web Search")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print("Commands: 'quit', 'exit', 'help', 'stats'")
        print("-" * 60)

        session_stats = {
            "queries": 0,
            "total_time": 0
        }

        chat_history = []

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nUser: ")
                user_input = user_input.strip()
                # handle special commands
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                elif user_input.lower() == "help":
                    self.show_help()
                    continue
                elif user_input.lower() == "stats":
                    self.show_stats(session_stats)
                    continue
                elif user_input.lower() == "clear":
                    chat_history = []
                    session_stats = {
                        "queries": 0,
                        "total_time": 0
                    }
                    print("\nChat history and stats cleared!")
                    continue
                elif not user_input:
                    continue

                print("\n🤖 Assistant:")
                print("-" * 50, flush=True)

                full_answer = ""
                final_result = None

                async for response_part in self.chat(user_input, chat_history):
                    if response_part["type"] == "content":
                        full_answer += response_part["data"]
                        print(response_part["data"], end="", flush=True)
                    elif response_part["type"] == "final":
                        final_result = response_part["data"]
                        break
                    elif response_part["type"] == "error":
                        err_msg = response_part["data"].get("error", "Unknown error")
                        print(f"\n❌ Error: {err_msg}")
                        final_result = response_part["data"]
                        break
                    
                print()
                print("-" * 50, flush=True)
                
                if final_result and final_result.get("success"):
                    session_stats["queries"] += 1
                    session_stats["total_time"] += final_result.get("response_time", 0)

                    # append the latest exchange to history
                    chat_history.append({"role": "user", "content": user_input})
                    chat_history.append({"role": "assistant", "content": full_answer})

                    # show metadata
                    local_count = final_result.get("local_results_count", 0)
                    processing_time = final_result.get("response_time", 0)
                    
                    metadata_parts = []
                    if local_count > 0:
                        metadata_parts.append(f"{local_count} local results 📚")
                        sources = final_result.get("local_sources", [])
                        
                        if sources:
                            display_sources = ', '.join(sources[:2])
                            if len(sources) > 2:
                                display_sources += '...'
                            metadata_parts.append(f"Source: {display_sources}")
                    
                    metadata_parts.append(f"⏱️Processing time: {processing_time:.2f} seconds")

                    print("ℹ️ " + " | ".join(metadata_parts))

            except KeyboardInterrupt:
                print("\nKeyboard interrupt. Exiting...")
                break

            except Exception as e:
                print(f"An error occurred: {e}")
                logger.error(f"Error occurred during interactive chat: {e}")

    def show_help(self):
        """Display helpful info"""
        help_text = """
        📖 HELP - Intelligent Chatbot Commands

        Basic Usage:
        • Type any question or query to get an answer
        • The bot will first check local knowledge, then search the web if needed

        Special Commands:
        • 'help' - Show this help message
        • 'stats' - Show session statistics  
        • 'clear' - Clear the chat history
        • 'quit' or 'exit' - End the chat session

        Features:
        • 🔍 Hybrid search combining local documents and web search
        • 📚 Local knowledge from video transcripts and documents
        • 🌐 Web search via DuckDuckGo and Wikipedia

        Examples:
        • "What is quantum computing?"
        • "Tell me about recent AI developments"
        • "Explain the content from the local videos about UFOs"
        """
        print(help_text)

    def show_stats(self, stats: Dict):
        """Display session statistics"""
        avg_time = stats["total_time"] / max(stats["queries"], 1)

        print(f"""
        📊 SESSION STATISTICS
        {'='*30}
        Queries processed: {stats['queries']}
        Average response time: {avg_time:.2f}s
        Total processing time: {stats['total_time']:.2f}s
        Model: {self.model_name}
        """)

    def single_query(self, query: str) -> str:
        """
        Process a single query and return just the answer

        Args:
            query: The question to ask
            
        Returns:
            The assistant's answer
        """
        result = self.chat(query)

        if result["success"]:
            return result["answer"]
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
        
async def main():
    """Main async function to run the chatbot"""
    chatbot = IntelligentChatBot()
    await chatbot.interactive_chat()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Exiting...")
