import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# -------------------------------------------------
# 1. Constants and Setup
# -------------------------------------------------
INDEX_FOLDER = "faiss_index"
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="qwen2.5:1.5b-instruct-q5_K_M", temperature=0.3)

qa_prompt_template = """You are a helpful, accurate, and concise AI tutor for sustainable energy topics.

IMPORTANT RULES:
- Answer ONLY using the information provided in the context below.
- If the context doesn't contain the answer or you are not sure, respond ONLY with: "I don't have enough information to answer this question."
- Do NOT make up information, speculate, or use general knowledge outside the context.
- Keep answers clear, factual, and brief.
- Do NOT add greetings, conclusions, suggestions, or extra commentary.

Context:
{context}

Question:
{question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

quiz_prompt_template = """You are a strict quiz question generator. Your ONLY task is to create ONE multiple-choice question.

STRICT FORMATTING RULES - YOU MUST FOLLOW THEM EXACTLY:
1. Generate exactly ONE question based ONLY on the provided context
2. Difficulty: {difficulty} (easy = basic facts, medium = explanations, hard = multi-step reasoning)
3. Output MUST be EXACTLY 6 lines in this exact order with no extra spaces, lines, words, explanations, or text anywhere else:
Question: [clear, well-formed question]
A) [plausible wrong or correct option]
B) [option]
C) [option]
D) [option]
Correct: [only one uppercase letter: A, B, C, or D]

FORBIDDEN BEHAVIORS (NEVER DO THESE):
- Do NOT write any introduction ("Here is a question", "Question:", etc.)
- Do NOT explain the question or answer
- Do NOT repeat the context
- Do NOT add "The correct answer is..." anywhere except the last line
- Do NOT include the correct answer in the question text or options
- Do NOT add notes, hints, difficulty level, or any other text

Context (use this to create an accurate question):
{context}

Now output ONLY the 6 lines exactly as specified. Nothing else."""

QUIZ_PROMPT = PromptTemplate(
    template=quiz_prompt_template,
    input_variables=["context", "difficulty"]
)

# 2. Load Vector Store
@st.cache_resource
def load_vectorstore():
    if os.path.exists(INDEX_FOLDER):
        return FAISS.load_local(
            INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        st.error("Run build_rag_index.py first!")
        return None

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# 3. Quiz Utilities
def generate_quiz(difficulty):
    docs = retriever.invoke("sustainable energy concepts")
    context = " ".join(doc.page_content for doc in docs[:3])

    prompt = QUIZ_PROMPT.format(
        context=context,
        difficulty=difficulty
    )

    response = llm.invoke(prompt).content.strip()
    lines = response.split("\n")

    quiz_lines = [l for l in lines if not l.startswith("Correct:")]
    correct_line = next((l for l in lines if l.startswith("Correct:")), None)

    correct = correct_line.split(":")[1].strip()[0] if correct_line else "B"
    return "\n".join(quiz_lines), correct


def evaluate_answer(user, correct):
    return user.upper() == correct.upper()

# 4. Session State Initialization
defaults = {
    "active_tab": "Chat Q&A",
    "quiz_mode": False,
    "difficulty": "easy",
    "score": 0,
    "total": 0,
    "current_quiz": None,
    "correct_answer": None,
    "user_answer": None,
    "submitted": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 5. UI Layout
st.title("🌱 Sustainable Energy AI Tutor")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Manage Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vectorstore.add_documents(chunks)
        vectorstore.save_local(INDEX_FOLDER)

        st.success(f"Added {uploaded_file.name}")

# ---------- Navigation ----------
tab = st.radio(
    "Navigation",
    ["Chat Q&A", "Quiz Mode"],
    horizontal=True,
    index=0 if st.session_state.active_tab == "Chat Q&A" else 1
)
st.session_state.active_tab = tab

# 6. Chat Q&A Tab
if tab == "Chat Q&A":
    st.header("💬 Ask Questions")

    query = st.text_input("Ask about sustainable energy:")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})

        st.write("**Answer:**")
        st.write(result["result"])

        st.write("**Sources:**")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata.get('source', 'Unknown')}")

# -------------------------------------------------
# 7. Quiz Mode Tab
# -------------------------------------------------
elif tab == "Quiz Mode":
    st.header("🧠 Quiz Mode")

    if st.button("Start / Next Question"):
        st.session_state.current_quiz, st.session_state.correct_answer = generate_quiz(
            st.session_state.difficulty
        )

        st.session_state.submitted = False
        st.session_state.user_answer = None

        # CRITICAL: reset radio widget
        if "quiz_radio" in st.session_state:
            del st.session_state["quiz_radio"]

    if st.session_state.current_quiz:
        st.write(st.session_state.current_quiz)

        st.session_state.user_answer = st.radio(
            "Select your answer:",
            ["A", "B", "C", "D"],
            key="quiz_radio",
            disabled=st.session_state.submitted
        )

        if not st.session_state.submitted:
            submit = st.button(
                "Submit Answer",
                disabled=st.session_state.user_answer is None
            )
            if submit:
                st.session_state.submitted = True

                correct = st.session_state.correct_answer
                if evaluate_answer(st.session_state.user_answer, correct):
                    st.success("Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"Incorrect. Correct answer: {correct}")

                question = st.session_state.current_quiz.split("\n")[0]
                with st.spinner("Generating explanation..."):
                    expl = qa_chain.invoke({
                        "query": f"Explain why {correct} is correct for: {question}"
                    })

                st.write("**Explanation:**")
                st.write(expl["result"])

                st.session_state.total += 1

                acc = st.session_state.score / st.session_state.total
                if acc > 0.7 and st.session_state.total >= 3:
                    st.session_state.difficulty = (
                        "medium" if st.session_state.difficulty == "easy" else "hard"
                    )
                elif acc < 0.5:
                    st.session_state.difficulty = "easy"

        st.info(
            f"Score: {st.session_state.score}/{st.session_state.total} "
            f"| Difficulty: {st.session_state.difficulty}"
        )

    else:
        st.info("Click **Start / Next Question** to begin.")