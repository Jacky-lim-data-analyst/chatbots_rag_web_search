# Ollama chatbot
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, List, Union, Iterator
from tools.websearch import SearchManager, SearchProvider
from ollama import Client, ListResponse, list as ollama_list

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FunctionTool:
    """Represents a function tool that can be called by the model"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function_callable: Callable

class OllamaClient:
    """Enhanced Ollama client (chatbot) with RAG and function calling capabilities"""
    def __init__(self, model: str, stream: bool = False):
        self.model_name = model
        self.stream = stream
        self.search_manager = SearchManager()
        self.function_tools = self._initialize_tools()
        self.client = Client()

    def _initialize_tools(self) -> Dict[str, FunctionTool]:
        """Initialize the built-in function tools"""
        return {
            "web_search": FunctionTool(
                name="web search",
                description="web search services from various backends",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, a topic or entity name"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "The maximum number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                },
                function_callable=self._web_search
            ),
            "news_search": FunctionTool(
                name="news search",
                description="web search services from duckduckgo and yahoo search engines",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, a topic or entity name"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "The maximum number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                },
                function_callable=self._news_search
            ),
            "wikipedia_search": FunctionTool(
                name="wikipedia articles search",
                description="Search wikipedia for informative articles",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query, typically a topic or entity name"
                        }
                    },
                    "required": ["query"]
                },
                function_callable=self._wikipedia_search
            )
        }
    
    def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Wrapper for web text search by ddgs to be used as tool"""
        results = self.search_manager.search(SearchProvider.DDGSWEB, query=query, max_results=max_results)
        # convert to dataclass back to dictionary for json serialization
        return {"results": [asdict(result) for result in results]}
    
    def _news_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Wrapper for news search by ddgs to be used as tool"""
        results = self.search_manager.search(SearchProvider.DDGSNEWS, query=query, max_results=max_results)
        # convert to dataclass back to dictionary for json serialization
        return {"results": [asdict(result) for result in results]}
    
    def _wikipedia_search(self, query: str) -> Dict[str, Any]:
        """Wrapper for wikipedia search to be used as tool"""
        results = self.search_manager.search(SearchProvider.WIKIPEDIA, query=query)
        # convert to dataclass back to dictionary for json serialization
        return {"results": [asdict(result) for result in results]}
    
    def add_function_tool(self, tool: FunctionTool):
        """Add a new function tool"""
        if tool.name in self.function_tools:
            logger.warning(f"Overwriting existing function tool: {tool.name}")
        
        self.function_tools[tool.name] = tool

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get a list of available function tools"""
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
        for tool in self.function_tools.values()]
    
    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute function tool"""
        if function_name not in self.function_tools:
            return {
                "error": f"Function {function_name} not found"
            }
        
        tool = self.function_tools[function_name]
        try:
            return tool.function_callable(**arguments)
        except TypeError as e:
            logger.error(f"Error executing {function_name} with args {arguments}: {e}")
            return {
                "error": f"Error executing {function_name} with args {arguments}: {str(e)}"
            }
        except Exception as e:
            return {
                "error": f"Error executing {function_name}: {str(e)}"
            }
        
    def chat(self, messages: List[Dict[str, str]], **options_kwargs) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Chat with ollama model with optional function calling and streaming

        This method sends a request to the Ollama model. If the model decides to call
        a tool, this method handles the tool execution and sends the result back to 
        model to get the final response

        Args:
            messages (List[Dict[str, str]]): List of messages in the chat
            options_kwargs: Additional options to pass to the model, like temperature, top_p and etc

        Raises:
            Exception: If there is an error in the chat method

        Returns:
            if the stream is False, a dictionary with final model response.
            if the stream is True, a generator yielding response chunks
        """
        try:
            tools = self.get_available_tools()

            # first, make a non-streaming call to check for tool usage. 
            # Simpler and more reliable than trying to parse tool calls from a stream
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                tools=tools,
                stream=False
            )

            # check if the model wants to call a tool
            if response.get('message', {}).get('tool_calls'):
                logger.info("Model requested a tool call. Handling it.")

                # append the assistant's message with tool calls to the history
                assistant_message = response.get('message', {})
                messages.append(assistant_message)

                # execute all the tools called by the LLM
                for tool_call in assistant_message['tool_calls']:
                    function_name = tool_call.get('function', {}).get('name')
                    arguments = tool_call.get('function', {}).get('arguments', {})
                    # tool_call_id = tool_call.get('id')  # id is not needed anymore based on the current official GitHub code

                    logger.info(f"Executing tool: '{function_name}' with arguments: {arguments}")
                    tool_output = self._execute_function(function_name, arguments)

                    messages.append(
                        {
                            'role': 'tool',
                            'content': json.dumps(tool_output),   # because the tool output is dictionary object
                            'tool_name': function_name,
                        }
                    )

                # Send the tool results back to the model for the final response.
                # This final call respects the original 'stream' parameter.
                logger.info("Sending tool results back to model for final response.")
                final_response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                    options=options_kwargs
                )

                return final_response
            else:
                logger.info("No tool call requested. Return direct response")
                if self.stream: # make the call again
                    return self.client.chat(
                        model=self.model_name,
                        messages=messages,
                        stream=self.stream,
                        options=options_kwargs
                    )
                else:
                    return response
        # a more robust error handling (TODO)
        except Exception as e:
            logger.error(f"Error in chat method: {e}")
            error_response = {"error": f"Failed to get response from model: {str(e)}"}
            if self.stream:
                return iter([error_response])
            return error_response

    def _handle_function_calls(self, original_messages: List[Dict[str, str]],
                               response_with_calls: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle function calls in the model response
        
        Executes the requested functions, appends the result to the message history, and
        sends back to model to get a response
        
        Args:
            original_messages (List[Dict[str, str]]): The original message history
            response (Dict[str, Any]): model response with the tool calls

        Return:
            Dict[str, Any]: The final response from the model
        """
        assistant_message = response_with_calls.get('message', {})
        if not assistant_message.get('tool_calls'):
            return {"error": "Expected tool calls but none were found in the response."}
        
        # append the original messages with tool call response message
        messages = original_messages + [assistant_message]

        # execute all the tools called by llm
        for tool_call in assistant_message.get('tool_calls', []):
            function_name = tool_call.get('function', {}).get('name')
            arguments = tool_call.get('function', {}).get('arguments', {})
            tool_call_id = tool_call.get('id')

            logger.info(f"Executing tool: '{function_name}' with arguments: {arguments}")

            # execute the function and get the output
            tool_output = self._execute_function(function_name, arguments)

            messages.append(
                {
                    'role': 'tool',
                    'content': json.dumps(tool_output),
                    'tool_call_id': tool_call_id
                }
            )

        logger.info("Send tool results back to model for final response")
        final_response = self.client.chat(
            model=self.model_name,
            messages=messages
        )
        return final_response
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available local models with their details"""
        try:
            response: ListResponse = ollama_list()

            return [
                {
                    "name": model.get('name'),
                    "size": model.get('size'),
                    "details": model.get('details')
                }
                for model in response.get('models', [])
            ]

        except Exception as e:
            logger.error(f"Error in list_models (show Ollama models): {e}", exc_info=True)
            return []