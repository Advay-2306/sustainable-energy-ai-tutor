from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Set paths and load saved index
INDEX_FOLDER = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(INDEX_FOLDER, embeddings, allow_dangerous_deserialization=True)

# 2. Set up local LLM
llm = ChatOllama(model="qwen2.5:1.5b-instruct-q5_K_M", temperature=0.3)  # Low temp for factual answers

# 3. Custom prompt for grounded answers
prompt_template = """Use the following pieces of context to answer the question at the end and answer in brief (approx. 50 words). 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
Helpful Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Create RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Stuff all retrieved chunks into prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Top 5 chunks
    return_source_documents=True,  # Return sources for verification
    chain_type_kwargs={"prompt": PROMPT}
)

# 5. Test queries
queries = [
    "What is sustainable energy?",
    "Explain the efficiency of solar panels.",
    "What is energy efficiency?",
    "What are the major types of renewable energy sources?"
]

for query in queries:
    print(f"\nQuestion: {query}")
    result = qa_chain.invoke({"query": query})
    print(f"Answer: {result['result']}")
    print("Sources (page metadata):")
    for doc in result['source_documents']:
        print(f"- {doc.metadata} (snippet: {doc.page_content[:100]}...)")
    print("---")