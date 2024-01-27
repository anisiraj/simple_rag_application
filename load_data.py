from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

client=MongoClient(key_param.MONGO_URI)
dbName='langchain_demo'
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1024,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

collectionName='collection_of_text_blobs'
collection=client[dbName][collectionName]

def clean_collection(collection):
    collection.delete_many({})

#loader=DirectoryLoader('./sample_files',glob="./*.txt",show_progress=True)

loader=PyPDFLoader('/home/deepti/Documents/sg2b_smartgrid.pdf')
data=loader.load_and_split()

embeddings=OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2',
#                                        model_kwargs={'device': 'cpu'})

vectorStore=MongoDBAtlasVectorSearch.from_documents(data,embeddings,collection=collection)