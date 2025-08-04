# Intelligent chatbot with local & web search

This project is an advanced chatbot that combines local knowledge from YouTube video transcripts with real-time web search capabilities. It leverages large language models (LLMs) via Ollama, supports hybrid RAG search and provides an interactive command line interface (CLI).

## Features
* **Hybrid search**: Combines local document retrieval with web search to provide a more comprehensive and context-aware answers.
* **Local knowledge Base**: Uses YouTube video transcripts stored and indexed for fast semantic and text search.
* **Web search Integration**: Fallbacks to `ddgs` (Dux Distributed Global Search) and Wikipedia article snippets when local knowledge is not available.
* **Summarization**: Utilizes Ollama's `phi4` model for generating summaries of conversation history; `llama3.2` model for interact chat.
* **Streaming responses**: Streams LLM responses in real time.

## Project structure
```
- local_chatbot
    - __init__.py
    - ollama_chatbot.py    # Ollama Chat client
- local_knowledge
    - __init__.py
    - youtube_transcripts.py  # YouTube transcript loader
- nlp
    - __init__.py
    - nltk_downloads.py  # Run this file to download the nltk languages package
    - text_processing.py  # Text cleaning for search
- tools
    - __init__.py
    - websearch.py     # web search providers
main.py   # entry point of app
```

## Requirements
- Python 3.9+
- [Ollama Python SDK](https://github.com/ollama/ollama-python.git)
- [MongoDB](https://www.mongodb.com/docs/manual/administration/install-community/#std-label-install-mdb-community-edition). You can choose a cloud-based service like `Atlas`, but I use the community version for this project.
- [ChromaDB](https://docs.trychroma.com/docs/overview/getting-started)
- [NLTK](https://www.nltk.org/) and some of the [data](https://www.nltk.org/nltk_data/)
- Other dependencies can be found in [requirements.txt](./requirements.txt)

## Installation
1. Clone the repo
```sh
git clone <this-repo-url>
cd chat_rag
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Install Ollama and MongoDB:
* Make sure Ollama and MongoDB are running.

4. Download the required nltk data. Navigate to `./nlp/nltk_downloads.py` and run the script.

## Usage
Run the chatbot interactively.
```sh
python main.py
```
Commands:
- `help` - show help messages
- `exit` or `quit`- exit the chat
- `clear` - clear chat history and start a new session
- `stats` - show sessions statistics

## Configurations
You can customize Ollama model names, database URIs and other parameters by editing the arguments in `IntelligentChatBot`.

## Extending
- Add new tools in `./tools` directory
- Add new local knowledge in `./local_knowledge` directory
- Add new LLMs in `./ollama_chatbot` directory. You may want to use OpenAI, Google Gemini or Anthropic provided LLM API.
- Wrap the application in a web app, like streamlit, chainlit, Gradio and etc.

## License
MIT License