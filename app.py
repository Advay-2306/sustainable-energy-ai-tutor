import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 1. Constants and Setup
INDEX_FOLDER = "faiss_index"
PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="qwen2.5:1.5b-instruct-q5_K_M", temperature=0.3)

qa_prompt_template = """<role>helpful_accurate_tutor</role>

<rules>
- Answer ONLY from the context.
- If context has no answer or unsure, output ONLY: "I don't have enough information to answer this question."
- NEVER make up info. NEVER speculate. NEVER use outside knowledge.
- Keep answers clear, factual, brief. NO greetings. NO conclusions. NO extra commentary.
- Repeat: NEVER make up info.
</rules>

<context>
{context}
</context>

<question>
{question}
</question>

<output>Start answer now:"""

QA_PROMPT = PromptTemplate(
    template=qa_prompt_template,
    input_variables=["context", "question"]
)

quiz_prompt_template = """<role>strict_quiz_generator</role>

<rules>
- Create EXACTLY ONE multiple-choice question ONLY from context.
- Difficulty: {difficulty} (easy: basic facts; medium: simple explanations; hard: reasoning or comparisons).
- Options: 1 correct, 3 plausible wrong. Mix them up.
- NEVER explain question or answer. NEVER add extra words, intros, notes, hints.
- Output EXACTLY 6 lines: Question line, A) B) C) D) lines, Correct line.
- NO extra spaces/lines/text. Repeat: NO extras anywhere.
</rules>

<example_for_format_only>
Question: What is the capital of France?
A) Berlin
B) Paris
C) London
D) Madrid
Correct: B
</example_for_format_only>

<context>
{context}
</context>

<output>BEGIN QUIZ NOW. Start with "Question:":"""

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
    lines = [line.strip() for line in response.split("\n") if line.strip()]

    # Extract first 5 lines (Question + 4 options)
    if len(lines) >= 5:
        question_part = "\n".join(lines[:5])
    else:
        question_part = "\n".join(lines)

    # Find correct answer
    correct = "A"  # safe default
    for line in lines:
        if line.lower().startswith("correct:"):
            try:
                correct = line.split(":", 1)[1].strip().upper()[0]
                if correct in "ABCD":
                    break
            except:
                pass

    return question_part, correct


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

# Sidebar
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

# Navigation
tab = st.radio(
    "Navigation",
    ["Chat Q&A", "Quiz Mode"],
    horizontal=True,
    index=0 if st.session_state.active_tab == "Chat Q&A" else 1
)
st.session_state.active_tab = tab

# 6. Chat Q&A Tab – Modern version
if tab == "Chat Q&A":
    # Initialize chat history if not exists
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Welcome message (only shown once)
    if len(st.session_state.chat_messages) == 0:
        welcome = {
            "role": "assistant",
            "content": (
                "🌱 **Welcome to your Sustainable Energy AI Tutor!**\n\n"
                "You can ask me anything about:\n"
                "• Solar and wind energy\n"
                "• Energy efficiency\n"
                "• Carbon footprints & climate solutions\n"
                "• Renewable technologies, storage, policies...\n\n"
                "All answers are based **only** on the documents you provided.\n"
                "Just type your question below 👇"
            )
        }
        st.session_state.chat_messages.append(welcome)

    # Display chat history
    for message in st.session_state.chat_messages:
        avatar = "👤" if message["role"] == "user" else "🌱"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # Show sources in nice expander (only for assistant messages)
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for src in message["sources"]:
                        st.caption(os.path.basename(src))

    # Chat input at the bottom
    if prompt := st.chat_input("Ask about sustainable energy..."):
        # Add user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(user_msg)

        # Show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant", avatar="🌱"):
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain.invoke({"query": prompt})
                    answer = result["result"].strip()

                    # Clean up possible unwanted prefixes from prompt
                    if answer.startswith("Start answer now:"):
                        answer = answer.replace("Start answer now:", "", 1).strip()

                    sources = [doc.metadata.get("source", "Unknown")
                              for doc in result["source_documents"]]

                    # Store both answer and sources
                    ai_msg = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for src in sources:
                                st.caption(os.path.basename(src))

                except Exception as e:
                    error_msg = f"Sorry, something went wrong: {str(e)}"
                    st.error(error_msg)
                    ai_msg = {"role": "assistant", "content": error_msg}

        # Save assistant response to history
        st.session_state.chat_messages.append(ai_msg)

        # Auto-scroll to bottom (nice touch)
        st.rerun()

# 7. Quiz Mode Tab
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