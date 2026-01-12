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
<critical_rules>
- Generate a COMPLETELY NEW multiple-choice question using ONLY the facts in the provided context.
- Difficulty level: {difficulty}
- Create exactly 1 question with 4 options (A,B,C,D).
- Output format: exactly these 7 lines:

Question: [your new question here]
A) [option]
B) [option]
C) [option]
D) [option]
Correct: [A/B/C/D]
Explanation: [Brief reason why this answer is correct]

- No introductions, no extra text. Start directly with "Question: ".
</critical_rules>

<context>
{context}
</context>

Produce the quiz question now."""

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

    docs = retriever.invoke(f"sustainable energy {difficulty} concepts")
    context = " ".join(doc.page_content for doc in docs[:3])

    prompt = QUIZ_PROMPT.format(context=context, difficulty=difficulty)
    response = llm.invoke(prompt).content.strip()

    try:
        lines = [l.strip() for l in response.split("\n") if l.strip()]

        # Simple extraction based on prefixes
        question = next((l for l in lines if l.startswith("Question:")), "Question not found")
        options = [l for l in lines if l[0] in ['A', 'B', 'C', 'D'] and ')' in l[:3]]
        correct_line = next((l for l in lines if l.startswith("Correct:")), None)
        explanation_line = next((l for l in lines if l.startswith("Explanation:")), "Explanation not available.")

        # specific parsing
        correct_char = correct_line.split(":")[1].strip().upper()[0] if correct_line else "A"
        explanation_text = explanation_line.split(":", 1)[1].strip() if ":" in explanation_line else explanation_line

        # Reconstruct the question block for display
        formatted_question = f"{question}\n" + "\n".join(options)

        return {
            "question_text": formatted_question,
            "correct_option": correct_char,
            "explanation": explanation_text
        }

    except Exception as e:
        st.error(f"Error parsing quiz: {e}")
        return None


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

            # Show sources in nice expander
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

        # Auto-scroll to bottom
        st.rerun()

# 7. Quiz Mode Tab
elif tab == "Quiz Mode":
    st.header("🧠 Quiz Mode")

    if st.button("Start / Next Question"):
        with st.spinner("Drafting a new question..."):
            quiz_data = generate_quiz(st.session_state.difficulty)

            if quiz_data:
                st.session_state.current_quiz = quiz_data
                st.session_state.submitted = False
                st.session_state.user_answer = None
                st.session_state.score_updated = False

                if "quiz_radio" in st.session_state:
                    del st.session_state.quiz_radio

                st.rerun()

    # Display
    if st.session_state.current_quiz:
        st.markdown(f"### {st.session_state.current_quiz['question_text']}")

        # Radio Button
        st.session_state.user_answer = st.radio(
            "Select your answer:",
            ["A", "B", "C", "D"],
            key="quiz_radio",
            index=None,
            disabled=st.session_state.submitted
        )

        # Submit Button
        if not st.session_state.submitted:
            if st.button("Submit Answer", disabled=st.session_state.user_answer is None):
                st.session_state.submitted = True
                st.rerun()

        # Result
        if st.session_state.submitted:
            user_ans = st.session_state.user_answer
            correct_ans = st.session_state.current_quiz['correct_option']

            # Calculate Score
            # We check a flag to ensure we don't increment total/score on every page reload
            if not st.session_state.get("score_updated", False):
                st.session_state.total += 1
                if user_ans == correct_ans:
                    st.session_state.score += 1

                # Adjust difficulty based on accuracy
                if st.session_state.total > 0:
                    acc = st.session_state.score / st.session_state.total
                    if acc > 0.7 and st.session_state.total >= 3:
                        st.session_state.difficulty = (
                            "medium" if st.session_state.difficulty == "easy" else "hard"
                        )
                    elif acc < 0.5:
                        st.session_state.difficulty = "easy"

                st.session_state.score_updated = True

            # Feedback
            if user_ans == correct_ans:
                st.success("🎉 Correct!")
            else:
                st.error(f"❌ Incorrect. The correct answer was **{correct_ans}**.")

            # Show the pre-generated explanation
            st.info(f"**Explanation:** {st.session_state.current_quiz['explanation']}")

        # Display current stats
        st.info(
            f"Score: {st.session_state.score}/{st.session_state.total} "
            f"| Difficulty: {st.session_state.difficulty}"
        )
    else:
        st.info("Click **Start / Next Question** to begin.")