from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
path_pdf="data/sample.pdf"

reader = PdfReader(path_pdf)
text=""
for page in reader.pages:
    text += page.extract_text() or ""
print("Pdf text extracted successfully")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

"""for i,chunk in enumerate(chunks):
    print(f"chunk {i+1}---")
    print(chunk)"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Embeddings model loaded successfully")

vectors = embeddings.embed_documents(chunks)
print("Embeddings generated successfully")

"""print("Total vectors:", len(vectors))
print("vector dimensions:", len(vectors[0]))
print("First vector:", vectors[0][:10])"""

vectore_store = FAISS.from_embeddings(text_embeddings=list(zip(chunks, vectors)),embedding = embeddings)

print("vectore store created successfully")

vectore_store.save_local("vector_index")

print("vectore database saved successfully")

"""query="What programming languages  does sampathi know?"

result=vectore_store.similarity_search(query, k=3)
print("\n relevent search result:")"""

"""for i, res in enumerate(result):
    print(f"Result {i+1}---")
    print(res.page_content)"""

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=api_key, temperature=0,
                           max_retries=3, timeout=60)
query = "What programming languages does sampathi know?"

results = vectore_store.similarity_search(query, k=3)

context = "\n\n".join(document.page_content for document in results)

prompt = f"""Answer the question using only the below context

Context:
{context}

Question:
{query}
If the answer is not available in the context
Say that the information is not available. """

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.text)
