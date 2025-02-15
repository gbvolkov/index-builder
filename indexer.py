import config
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer

from nltk.tokenize import sent_tokenize

import pickle

import time
from tqdm import tqdm  # For progress tracking
import re
import json
import os
from functools import lru_cache
from typing import List, Dict, Tuple, Any

from sentence_splitter import chunk_sentences


import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_ref_content(content: str) -> str:
    pattern = r'##IMAGE##\s+\S+\.(?:jpg|jpeg|png|gif|bmp|svg)'
    cleaned_content = re.sub(pattern, '', content)
    return cleaned_content


def length_factory(tokenizer: Any = None):
    @lru_cache(maxsize=5000, typed=True)
    def _len(text: str) -> int:
        return len(tokenizer.encode(text))
    if tokenizer:
        return _len
    else:
        return len

def process_json_for_indexing(json_file_path: Any, max_chunk_size: int = 2000, overlap=0.75, embedding_model_name: str = None) -> List[Dict]:
    
    if type(json_file_path) == str:
        try:
            with open(json_file_path, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            logger.info(f"Error decoding JSON: {e}")
            return []
        except FileNotFoundError:
            logger.info(f"File not found: {json_file_path}")
            return []
    else:
        data = json_file_path

    tokenizer = None
    if embedding_model_name:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    _len = length_factory(tokenizer)

    processed_documents = []
    #for item in data:
    for item in tqdm(data, desc="Processing json data", unit="item"):
        core_content = (
            f"Problem Number: {item.get('problem_number', '')}\n"
            f"Problem Description: {item.get('problem_description', '')}\n"
            #f"Systems: {item.get('systems', '')}\n"
            f"Solution Steps: {item.get('solution_steps', '')}\n"
        )
        core_length = _len(f'{core_content}\nAdditional Information (Part 9999/9999):\n')
        current_chunk_max_size = max_chunk_size - core_length

        if max_chunk_size > 0:
            references = cleanup_ref_content(item.get('references', ''))
        else:
            references = item.get('references', '')
        additional_content = f"{references}"

        if max_chunk_size > 0 and _len(additional_content) >= current_chunk_max_size:
            sentences = sent_tokenize(additional_content, language='russian')
            additional_chunks = chunk_sentences(sentences, max_chunk_size=current_chunk_max_size, overlap_size=current_chunk_max_size * overlap, _len=_len)    
        else:
            additional_chunks = [additional_content] 
        
        for i, chunk in enumerate(additional_chunks):
            full_content = f"{core_content}\nAdditional Information (Part {i+1}/{len(additional_chunks)}):\n{chunk}"
            processed_documents.append({
                'content': full_content,
                'metadata': {
                    'problem_number': item.get('problem_number', ''),
                    'url':  item.get('url', ''),
                    'chunk_number': i+1,
                    'total_chunks': len(additional_chunks),
                    'actual_chunk_size': _len(full_content)
                }
            })
    
    return processed_documents

def get_documents(
        json_file_path: Any,
        max_chunk_size: int = 4000,
        overlap: int = 0.5,
        embedding_model_name: str = None
) -> List[Document]:
    # Step 1: Process the JSON file
    processed_docs = process_json_for_indexing(json_file_path, max_chunk_size=max_chunk_size, overlap=overlap, embedding_model_name=embedding_model_name)
    logger.info(f"Documents processed from {json_file_path}. {len(processed_docs)} documents found.")
    if not processed_docs:
        logger.error("No documents to process. Exiting vector store creation.")
        return []
    
    if logger.isEnabledFor(logging.DEBUG):
        # Save processed documents to a text file for debugging
        with open('./logs/processed_docs.txt', 'w', encoding='utf-8-sig') as f:
            f.write('\n\n======================\n'.join([doc['content'] for doc in processed_docs]))
        logger.info("Processed documents saved to 'processed_docs.txt'.")
    # Step 2: Convert processed docs to Document objects
    documents = [
        Document(page_content=doc['content'], metadata=doc.get('metadata', {}))
        for doc in processed_docs
    ]
    logger.info("Documents converted to LangChain Document objects.")
    return documents


# Helper function to determine the device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def create_vectorstore(json_file_path: str, embedding_model_name: str, batch_size: int = 500, max_retries: int = 3, max_chunk_size: int = 4000, overlap: int = 0.5) -> Tuple[FAISS, List[Document]]:
    """
    Creates a FAISS vectorstore from a JSON file using the specified embedding model, supporting GPU acceleration if available.

    Args:
        json_file_path (str): Path to the JSON file containing documents.
        embedding_model_name (str): Name or path of the embedding model.
        batch_size (int, optional): Number of documents to process in each batch. Defaults to 500.
        max_retries (int, optional): Maximum number of retry attempts for failed batches. Defaults to 3.

    Returns:
        FAISS: The created FAISS vectorstore.
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    try:
        # Step 1: Process documents
        documents = get_documents(json_file_path, max_chunk_size=max_chunk_size, overlap=overlap, embedding_model_name=embedding_model_name)
        if len(documents) == 0:
            logger.error("No documents to process. Exiting vector store creation.")
            return None

        # Step 2: Initialize the embedding model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})
        logger.info(f"Embeddings loaded from model: {embedding_model_name}.")

        # Step 3: Initialize FAISS vectorstore
        first_batch = documents[:batch_size]
        logger.info(f"Initializing FAISS vectorstore with the first batch of {len(first_batch)} documents.")
        vectorstore = FAISS.from_documents(documents=first_batch, embedding=embeddings)

        # Step 4: Transfer FAISS index to GPU if CUDA is available
        if device == "cuda":
            import faiss
            if hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                cpu_index = vectorstore.index
                vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("FAISS index transferred to GPU.")

        # Step 5: Process and add remaining documents in batches
        remaining_documents = documents[batch_size:]
        total_batches = (len(remaining_documents) + batch_size - 1) // batch_size
        logger.info(f"Adding remaining documents in {total_batches} batches of size {batch_size}.")

        for batch_num, i in enumerate(range(0, len(remaining_documents), batch_size), start=1):
            batch = remaining_documents[i:i + batch_size]
            attempt = 0
            while attempt < max_retries:
                try:
                    logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} documents.")
                    vectorstore.add_documents(batch)
                    logger.info(f"Batch {batch_num}/{total_batches} added successfully.")
                    break
                except Exception as e:
                    attempt += 1
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt}/{max_retries} failed for batch {batch_num}: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        logger.error(f"Batch {batch_num} failed after {max_retries} attempts. Skipping this batch.")

        logger.info("All batches processed. Vector store is ready.")
        full_documents = get_documents(json_file_path, max_chunk_size=-1, overlap=0.5)
        logger.info("Full documents store processed. Vector store and doc strore are ready.")
        return (vectorstore, full_documents)

    except Exception as general_e:
        logger.exception(f"An unexpected error occurred: {general_e}")
        raise

def save_vectorstore(vectorstore: FAISS, docstore: List[Document], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logger.info(f"Storing vectore store to {file_path}.")
    try:
        vectorstore.save_local(file_path)
        with open(f'{file_path}/docstore.pkl', 'wb') as file:
            pickle.dump(docstore, file)
    except Exception as e:
        logger.error(f"Unexpected error while storing vector store: {str(e)}")
        raise
    logger.info(f"Vectorstore saved to {file_path}")


if __name__ == '__main__':
    import nltk
    nltk.download('punkt_tab')

    json_name = "data/kb.json"
    embedding_model_name = 'intfloat/multilingual-e5-large'
    vectorestore_path = 'data/vectorstore_e5'
    (vectorstore, docstore) = create_vectorstore(json_name, embedding_model_name, batch_size=500, max_chunk_size=400, overlap=0.75) 
    save_vectorstore(vectorstore, docstore, vectorestore_path)