import os
import json
import gradio
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 環境変数の読み込み
from dotenv import load_dotenv
load_dotenv()

# LLMのインスタンスの作成
chat = ChatOpenAI(model_name=OPENAI_API_MODEL)

loader = PyPDFLoader()
pages = loader.load_and_split()