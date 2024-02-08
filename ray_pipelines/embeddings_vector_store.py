import time
import datetime
from torch import cuda
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


class EmbeddingsVectorStore:
    def __init__(self):
        ct = datetime.datetime.fromtimestamp(time.time())
        print('Loading documents... ' + str(ct.strftime('%Y-%m-%d %H:%M:%S')))
        loader = ReadTheDocsLoader('../document_embeddings/docs.ray.io/en/master/')
        documents = loader.load()
        
        # TODO: Arbitrarily chose chunk_size and chunk_overlap based off of Medium articles
        ct = datetime.datetime.fromtimestamp(time.time())
        print('Splitting documents...' + str(ct.strftime('%Y-%m-%d %H:%M:%S')))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 64)
        all_splits = text_splitter.split_documents(documents)
        
        model_name = "sentence-transformers/all-mpnet-base-v2"
        device = 'cuda' if cuda.is_available() else 'cpu'
        if device == 'cuda': # TODO: Only supports GPU and CPU
            model_kwargs = {'device' : 'cuda'}
        else:
            model_kwargs = {'device' : 'cpu'}
        
        ct = datetime.datetime.fromtimestamp(time.time())
        print('Generating and storing embeddings...' + str(ct.strftime('%Y-%m-%d %H:%M:%S')))
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        
        ct = datetime.datetime.fromtimestamp(time.time())
        print('Done.' + str(ct.strftime('%Y-%m-%d %H:%M:%S')))
