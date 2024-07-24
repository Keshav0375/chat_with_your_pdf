import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
OPENAI_API_TYPE = os.environ['OPENAI_API_TYPE']
OPENAI_API_BASE = os.environ['OPENAI_API_BASE']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
DEPLOYMENT_NAME = os.environ['DEPLOYMENT_NAME']


def get_conversation_chain(vector_stores):
    llm = AzureChatOpenAI(
        deployment_name=DEPLOYMENT_NAME,
        temperature=1,
        max_tokens=4000,
        azure_endpoint=OPENAI_API_BASE
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_stores.as_retriever(),
        memory=memory
    )
    return conversation_chain
