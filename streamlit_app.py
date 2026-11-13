
import streamlit as st
import os
import re
import hashlib
from io import BytesIO
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------
# 1. Page configuration
# ---------------------------------

st.set_page_config(
    page_title="PDF Question Answering Chatbot",
    page_icon="📄"
)

st.title("📄 PDF Question Answering Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# ---------------------------------
# 2. Load Gemini API key
# ---------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ---------------------------------
# 3. Initialize chat history
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------
# 4. Load embedding model
# ---------------------------------

@st.cache_resource
def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


# ---------------------------------
# 5. Create FAISS vector database
# ---------------------------------

@st.cache_resource
def create_vector_store(pdf_bytes):

    reader = PdfReader(BytesIO(pdf_bytes))

    documents = []

    for page_number, page in enumerate(reader.pages):

        page_text = page.extract_text() or ""

        if page_text.strip():

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "page": page_number + 1
                    }
                )
            )

    if not documents:
        raise ValueError("No readable text found in this PDF.")

    # Split PDF text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be created.")

    # Store the original order of every chunk
    for chunk_index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = chunk_index

    # Generate embeddings
    embedding_model = get_embedding_model()

    # Create FAISS database
    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


# ---------------------------------
# 6. PDF upload
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    # Identify uploaded PDF
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Reset conversation for a different PDF
    if st.session_state.get("pdf_hash") != pdf_hash:

        st.session_state.pdf_hash = pdf_hash
        st.session_state.messages = []

    try:

        # Create or reuse FAISS database
        with st.spinner("Processing PDF..."):

            vector_store = create_vector_store(pdf_bytes)

        st.success("PDF is ready! You can ask questions.")

        # ---------------------------------
        # 7. Clear chat button
        # ---------------------------------

        if st.button("🗑️ Clear Chat"):

            st.session_state.messages = []

            st.rerun()

        # ---------------------------------
        # 8. Display previous conversation
        # ---------------------------------

        st.subheader("Chat History")

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message["role"] == "assistant":

                    source_pages = message.get("sources", [])

                    if source_pages:

                        st.caption(
                            "Source pages: " +
                            ", ".join(
                                str(page) for page in source_pages
                            )
                        )

        # ---------------------------------
        # 9. Receive user question
        # ---------------------------------

        question = st.chat_input(
            "Ask a question about your PDF..."
        )

        if question:

            if not api_key:

                st.error(
                    "Gemini API key not found. Check your .env file."
                )

            else:

                # Save user question
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):
                    st.markdown(question)

                # ---------------------------------
                # 10. Get previous conversation
                # ---------------------------------

                previous_messages = st.session_state.messages[:-1]

                history_text = "\n".join(
                    f"{message['role']}: {message['content']}"
                    for message in previous_messages[-8:]
                )

                question_lower = question.lower()

                # ---------------------------------
                # 11. Retrieve relevant PDF chunks
                # ---------------------------------

                with st.spinner("Finding the answer..."):

                    # Questions about complete sections
                    broad_question_keywords = [
                        "project",
                        "experience",
                        "internship",
                        "employment",
                        "education",
                        "qualification",
                        "skill",
                        "certification",
                        "responsibilit",
                        "responsibility",
                        "duties",
                        "duty",
                        "role",
                        "position",
                        "company",
                        "technolog"
                    ]

                    is_broad_question = any(
                        keyword in question_lower
                        for keyword in broad_question_keywords
                    )

                    if is_broad_question:

                        # Retrieve all chunks for complete coverage
                        results = vector_store.similarity_search(
                            question,
                            k=vector_store.index.ntotal
                        )

                        # Restore original PDF chunk order
                        results.sort(
                            key=lambda document:
                            document.metadata["chunk_index"]
                        )

                    else:

                        # Retrieve top 3 relevant chunks
                        results = vector_store.similarity_search(
                            question,
                            k=3
                        )

                    # Prepare PDF context
                    context = "\n\n".join(
                        f"[Page {document.metadata['page']}]\n"
                        f"{document.page_content}"
                        for document in results
                    )

                    # ---------------------------------
                    # 12. Initialize Gemini
                    # ---------------------------------

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash-lite",
                        google_api_key=api_key,
                        max_retries=3,
                        timeout=60
                    )

                    # ---------------------------------
                    # 13. Prepare prompt
                    # ---------------------------------

                    prompt = f"""
                    You are a helpful PDF question-answering assistant.

                    Previous conversation:
                    {history_text}

                    PDF context:
                    {context}

                    Current question:
                    {question}

                    Instructions:

                    1. Answer using only information from the PDF context.

                    2. Use conversation history to understand
                       follow-up questions and references.

                    3. For questions about projects, experience,
                       internships, education, skills or certifications,
                       include all relevant information in the context.

                    4. For responsibility questions, list every
                       responsibility bullet found in the relevant
                       experience section. Do not omit bullets or
                       shorten the list.

                    5. Preserve the original meaning of each
                       responsibility.

                    6. Include job title, company, duration,
                       responsibilities and work mode when available.

                    7. Organize skills and certifications clearly
                       when the user asks about them.

                    8. Do not confuse technical skills, projects,
                       certifications and work experience.

                    9. If information is not available in the PDF,
                       clearly say that it is not available.

                    10. Do not invent information.

                    11. Mention relevant PDF page numbers using
                        this exact format: [Page 1].

                    12. Only cite pages that support your answer.
                    """

                    # ---------------------------------
                    # 14. Generate answer
                    # ---------------------------------

                    response = llm.invoke(prompt)

                    answer = response.text

                    # ---------------------------------
                    # 15. Extract source page numbers
                    # ---------------------------------

                    retrieved_pages = {
                        document.metadata["page"]
                        for document in results
                    }

                    cited_pages = re.findall(
                        r"\[Page\s+(\d+)\]",
                        answer
                    )

                    source_pages = sorted({
                        int(page)
                        for page in cited_pages
                        if int(page) in retrieved_pages
                    })

                # ---------------------------------
                # 16. Save and display answer
                # ---------------------------------

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_pages
                })

                with st.chat_message("assistant"):

                    st.markdown(answer)

                    if source_pages:

                        st.caption(
                            "Source pages: " +
                            ", ".join(
                                str(page) for page in source_pages
                            )
                        )

    except Exception as e:

        st.error(f"An error occurred: {e}")


else:

    st.info("Please upload a PDF to start chatting.")
