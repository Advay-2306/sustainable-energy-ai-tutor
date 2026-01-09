from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 1. Set paths
PDF_FOLDER = "pdfs"
INDEX_FOLDER = "faiss_index" # Where the vector DB will be saved

# 2. Load all PDFs from the folder
print("Loading PDFs...")
loader = PyPDFDirectoryLoader(PDF_FOLDER)
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDFs.")

# 3. Split into chunks (better for retrieval)
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

# 4. Create embeddings (local model)
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Build and save FAISS vector database
print("Building FAISS index... (this may take a few minutes)")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Create folder if needed
if not os.path.exists(INDEX_FOLDER):
    os.makedirs(INDEX_FOLDER)

# Save locally
vectorstore.save_local(INDEX_FOLDER)
print(f"RAG index saved to '{INDEX_FOLDER}'")
print("Done! You can now use this index for questions and quizzes.")