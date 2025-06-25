# Legal Document RAG Chatbot (Retrieval-Augmented Generation)

This project is a RAG (Retrieval-Augmented Generation) chatbot designed to answer legal questions based on the content of uploaded PDFs, such as laws, regulations, or internal manuals.

It uses FAISS for vector storage, reranks with Flashrank (MS Marco BERT), and responds via the `tinyllama` local language model.

## Features

- Extracts text from PDF legal documents
- Uses semantic similarity to retrieve relevant sections
- Reranks results using a lightweight reranker
- Generates natural-language answers using a local LLM
- Runs locally – no API keys or internet required

## Model Architecture

- **Embeddings**: `BAAI/bge-m3` via HuggingFace
- **Vector DB**: FAISS
- **LLM**: `tinyllama` via Ollama
- **Reranker**: `ms-marco-MultiBERT-L-12`
- **Prompt Type**: Custom LangChain template

## Requirements

```bash
pip install pdfplumber faiss-cpu langchain langchain-community langchain-huggingface langchain-ollama flashrank
```

You also need to:
- Have [`ollama`](https://ollama.com/) installed and running
- Pull the TinyLLaMA model:
```bash
ollama pull tinyllama
```

## How to Use

1. Replace the `pdf_path` variable in the code with your own local PDF.
2. Run the script.
3. Enter questions in Chinese, e.g.:
   ```
   公司法中董事的責任是什麼？
   ```

## Disclaimer

This chatbot uses **synthetic or public legal documents** only. Please **do not upload confidential or proprietary PDFs**.

