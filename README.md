# PDF Question Answering Chatbot using RAG

## Project Overview
An AI-powered PDF Question Answering Chatbot that allows users to upload PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-aware answers using Google Gemini.

## Features
- Upload and process PDF documents.
- Extract text from PDF files.
- Split documents into smaller chunks.
- Generate embeddings using HuggingFace Sentence Transformers.
- Store and retrieve document embeddings using FAISS.
- Generate answers using Google Gemini.
- Maintain conversation history for follow-up questions.
- Display source page references.
- Clear chat without removing the uploaded document.

## Technologies Used
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- Google Gemini API
- PyPDF

## RAG Workflow
1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the text into chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on the user's question.
7. Pass the retrieved context to Gemini.
8. Generate an answer with source page references.

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/sampathi-24/PDF_RAG_ChatBot.git
cd PDF_RAG_ChatBot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment
```powershell
.venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Gemini API Key

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace `your_gemini_api_key` with your own API key.

### 6. Run the application
```bash
streamlit run streamlit_app.py
```

## Future Improvements
- Support multiple PDF uploads.
- Improve retrieval accuracy.
- Add document summarization.
- Deploy the application online.

## Author
**Sampathi Varaprasad**

GitHub: https://github.com/sampathi-24