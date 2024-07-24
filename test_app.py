import unittest
import streamlit as st
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock, mock_open
from app import get_pdf_text, get_text_chunk, get_vectors, process_pdfs, handle_userinput
from api import get_conversation_chain, ask_question
from fastapi.testclient import TestClient
from api import app as fastapi_app
from langchain.chains import ConversationalRetrievalChain


class TestAppFunctions(unittest.TestCase):

    def setUp(self):
        # Load environment variables
        load_dotenv()
        self.pdf_path = "test.pdf"
        self.pdf_docs = [self.pdf_path]

    @patch("app.PdfReader")
    def test_get_pdf_text(self, mock_pdf_reader):
        # Mock the PdfReader to return specific text
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="Test text."))]
        mock_pdf_reader.return_value = mock_pdf

        text = get_pdf_text(self.pdf_docs)
        self.assertEqual(text, "Test text.")

    def test_get_text_chunk(self):
        text = "This is a test text to be split into chunks."
        chunks = get_text_chunk(text)
        self.assertTrue(len(chunks) > 0)

    @patch("app.AzureOpenAIEmbeddings")
    @patch("app.FAISS.from_texts")
    def test_get_vectors(self, mock_faiss, mock_embeddings):
        mock_faiss.return_value = MagicMock()
        chunks = ["This is a chunk."]
        vector_store = get_vectors(chunks)
        self.assertIsNotNone(vector_store)

    @patch("app.get_pdf_text")
    @patch("app.get_text_chunk")
    @patch("app.get_vectors")
    def test_process_pdfs(self, mock_get_vectors, mock_get_text_chunk, mock_get_pdf_text):
        mock_get_pdf_text.return_value = "Test text."
        mock_get_text_chunk.return_value = ["Test text chunk."]
        mock_vector_store = MagicMock()
        mock_vector_store.save_local = MagicMock()
        mock_get_vectors.return_value = mock_vector_store

        result = process_pdfs(self.pdf_docs)
        self.assertEqual(result, "Processed Successfully")

    @patch("requests.post")
    def test_handle_userinput(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200,
                                           json=lambda: {"chat_history": [{"content": "Test response."}]})
        st.session_state.chat_history = []

        handle_userinput("Test question")
        self.assertEqual(len(st.session_state.chat_history), 1)


class TestAPIFunctions(unittest.TestCase):
    def setUp(self):
        # Load environment variables
        load_dotenv()
        self.client = TestClient(fastapi_app)

    @patch("api.FAISS.load_local")
    @patch("api.AzureChatOpenAI")
    @patch("api.ConversationalRetrievalChain.from_llm")
    def test_get_conversation_chain(self, mock_conversation_chain, mock_azure_chat_openai, mock_faiss):
        mock_vector_store = MagicMock()
        mock_faiss.return_value = mock_vector_store
        mock_llm = MagicMock()
        mock_azure_chat_openai.return_value = mock_llm
        mock_chain = MagicMock()
        mock_conversation_chain.return_value = mock_chain

        conversation_chain = get_conversation_chain()
        self.assertIsNotNone(conversation_chain)

    @patch.object(ConversationalRetrievalChain, 'invoke')
    def test_ask_question(self, mock_invoke):
        mock_invoke.return_value = {"answer": "Test answer", "chat_history": [{"content": "Test response."}]}
        response = self.client.post("/ask", json={"question": "Test question", "chat_history": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Test answer")


if __name__ == '__main__':
    unittest.main()
