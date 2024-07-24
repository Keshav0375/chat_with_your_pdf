# Conversational PDF Chatbot

This repository contains a Conversational PDF Chatbot that uses Retrieval-Augmented Generation (RAG) architecture and FastAPI to allow users to ask questions about their uploaded PDF documents and receive responses based on the document content.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [How to Use](#how-to-use)
5. [Deployment](#deployment)
6. [Endpoints](#endpoints)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

The Conversational PDF Chatbot leverages the power of RAG architecture, combining retrieval-based and generative-based models to provide accurate and contextually relevant answers to user queries. It uses FastAPI to create a simple API that handles user inputs and retrieves information from the uploaded PDFs.

## Architecture

### Retrieval-Augmented Generation (RAG)

RAG architecture integrates both retrieval and generation processes:

- **Retrieval**: Uses a pre-built FAISS (Facebook AI Similarity Search) index to quickly search and retrieve relevant chunks of text from the uploaded PDFs.
- **Generation**: Uses OpenAI's Azure OpenAI service to generate coherent responses based on the retrieved text chunks.

### Components

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **PyPDF2**: A library to read and extract text from PDF files.
- **Langchain**: A framework for building language models.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Streamlit**: An app framework for creating web apps using Python.

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/conversational-pdf-chatbot.git
    cd conversational-pdf-chatbot
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    Create a `.env` file in the root directory and add your Azure OpenAI and other necessary keys:
    ```env
    OPENAI_API_TYPE=your_openai_api_type
    OPENAI_API_BASE=your_openai_api_base
    OPENAI_API_VERSION=your_openai_api_version
    OPENAI_API_KEY=your_openai_api_key
    DEPLOYMENT_NAME=your_deployment_name
    TEXT_DEPLOYMENT_NAME=your_text_deployment_name
    ```

## How to Use

1. **Run the FastAPI server:**
    ```sh
    uvicorn api:app --reload
    ```

2. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

3. **Upload PDF documents and ask questions:**
    - Use the Streamlit interface to upload PDF documents.
    - Enter your question in the text input box.
    - The bot will respond based on the content of the uploaded PDFs.




## Endpoints

- **POST /ask**
    - Request:
        ```json
        {
            "question": "Your question here",
            "chat_history": []
        }
        ```
    - Response:
        ```json
        {
            "response": "The answer to your question",
            "chat_history": [...]
        }
        ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements.

## License

This project is licensed under the MIT License.
