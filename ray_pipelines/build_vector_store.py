import time
import os
import ray
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from typing import List
from embeddings import LocalHuggingFaceEmbeddings

FAISS_INDEX_PATH = '../'
