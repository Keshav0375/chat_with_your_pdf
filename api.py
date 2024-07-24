import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
OPENAI_API_TYPE = os.environ['OPENAI_API_TYPE']
OPENAI_API_BASE = os.environ['OPENAI_API_BASE']
OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
DEPLOYMENT_NAME = os.environ['DEPLOYMENT_NAME']
Text_DEPLOYMENT_NAME = os.environ['TEXT_DEPLOYMENT_NAME']


class UserInput(BaseModel):
    question: str
    chat_history: list = []


app = FastAPI()


# Load vector store and create conversation chain globally
def get_conversation_chain():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=Text_DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE
    )
    # Assuming the vector store is already created and saved in a file
    vector_stores = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

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


conversation_chain = get_conversation_chain()


@app.post("/ask")
async def ask_question(user_input: UserInput):
    try:
        response = conversation_chain.invoke({
            'question': user_input.question,
            'chat_history': user_input.chat_history
        })
        return {"response": response["answer"], "chat_history": response["chat_history"]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh_index")
async def refresh_index():
    global conversation_chain
    try:
        conversation_chain = get_conversation_chain()
        return {"message": "FAISS index refreshed successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))