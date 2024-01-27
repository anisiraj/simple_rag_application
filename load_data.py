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

client=MongoClient(key_param.MONGO_URI)
dbName='langchain_demo'


collectionName='collection_of_text_blobs'
collection=client[dbName][collectionName]

loader=DirectoryLoader('./sample_files',glob="./*.txt",show_progress=True)
data=loader.load()

embeddings=OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)
# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2',
#                                        model_kwargs={'device': 'cpu'})

vectorStore=MongoDBAtlasVectorSearch.from_documents(data,embeddings,collection=collection)