<div align="center">

<h1>🌱 Sustainable Energy AI Tutor</h1>

<p>
  <strong>Personalized AI-powered learning platform with RAG & Adaptive Quizzing</strong><br/>
  100% local • No API costs • Runs on your laptop
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangChain-1F8CEF?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"/>
  <img src="https://img.shields.io/badge/FAISS-FF6F61?style=for-the-badge" alt="FAISS"/>
</p>

</div>

<br/>

## ✨ What is this project?

Educational AI Tutor focused on **Sustainable Energy** topics  
(renewables, solar/wind efficiency, carbon footprint, energy policy, etc)

It uses **Retrieval Augmented Generation (RAG)** + **local LLM**  
to give accurate answers from your documents and generate **adaptive difficulty quizzes**

**Completely free • 100% local • No cloud required**

## 🎯 Key Features

- **Accurate Q&A** powered by RAG (answers are always grounded in your documents)
- **Adaptive Quiz System** — difficulty automatically increases/decreases based on performance
- **Dynamic document upload** — add new PDFs anytime → knowledge base updates instantly
- **Source citations** — see exactly which document page the answer came from
- **Detailed explanation** after every quiz answer
- **Beautiful, clean Streamlit interface**
- **Works offline** after initial model download

## 🚀 Quick Start (Local)

```bash
# 1. Install Ollama & pull a fast & good model
# (Recommended: gemma2:2b-instruct-q5_K_M  or  qwen2.5:3b-instruct-q5_K_M)
ollama pull qwen2.5:1.5b-instruct-q5_K_M

# 2. Clone & install dependencies
git clone https://github.com/yourusername/sustainable-energy-ai-tutor.git
cd sustainable-energy-ai-tutor


pip install -r requirements.txt

# 3. Put your PDFs in the pdfs/ folder
# (You can start with the PDFs added to this repo)

# 4. Create the knowledge base (only once, or when you add new PDFs)
python build_rag_index.py

# 5. Launch the app
streamlit run app.py
```

## 🛠️ Tech Stack

- **UI / Frontend**  
  Streamlit

- **LLM Inference**  
  Ollama (local)  
  Models: Qwen 2.5 1.5B–3B Instruct (Alternatives: Phi-3 Mini, Gemma-2 2B Instruct)

- **RAG Framework**  
  LangChain  
  (langchain, langchain-community, langchain-huggingface, langchain-ollama, langchain-text-splitters)

- **Vector Database**  
  FAISS (CPU)

- **Embeddings**  
  sentence-transformers/all-MiniLM-L6-v2

- **PDF Processing**  
  pypdf + PyPDFLoader / PyPDFDirectoryLoader

- **Text Splitting**  
  RecursiveCharacterTextSplitter

All components run locally; no paid APIs, no cloud dependency.