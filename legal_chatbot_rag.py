# 法律文件問答機器人 (RAG-based)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_ollama import OllamaLLM

# === Replace this path with your own legal document PDF ===
pdf_path = 'your_legal_document.pdf'

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

pdf_text = extract_text_from_pdf(pdf_path)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
)
texts = text_splitter.split_text(pdf_text)

# Embedding and vector store
HF_EMBEDDING_MODEL = 'BAAI/bge-m3'
hf_embeddings = HuggingFaceEmbeddings(
    model_name=HF_EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
vectordb = FAISS.from_texts(texts, hf_embeddings)

# Set up the LLM and reranker
LLM_MODEL = 'tinyllama'
RERANK_MODEL = 'ms-marco-MultiBERT-L-12'

llm = OllamaLLM(model=LLM_MODEL)
custom_prompt_template = """<your system instruction>
{context}
Question: {question}
Helpful Answer:"""
CUSTOMPROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 100})
compressor = FlashrankRerank(model=RERANK_MODEL, top_n=1)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True
)

qa.combine_documents_chain.llm_chain.prompt = CUSTOMPROMPT

def ask_questions():
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        answer = qa.invoke({"query": question})
        print("\nTop 1 most relevant responses:\n")
        if answer and 'source_documents' in answer:
            for i, doc in enumerate(answer['source_documents'][:1], start=1):
                print(f"Response {i}:\n{doc.page_content}\n")
        else:
            print("No relevant responses found.\n")

ask_questions()
