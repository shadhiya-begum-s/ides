# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_file
from chunking import chunk_text
import retriever
from generator import generate_answer
import utils
import db

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# ================= ENV & DB =================
load_dotenv()
db.init_db()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_docs")
TOP_K_DEFAULT = int(os.getenv("TOP_K_RESULTS", 5))
CONFIDENCE_DEFAULT = float(os.getenv("CONFIDENCE_THRESHOLD", 0.15))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= UI =================
st.set_page_config(page_title="📄 Intelligent Document Q&A", layout="wide")
st.title("🤖 Intelligent Document Q&A — FAISS RAG System")
st.caption("Upload documents, ask questions, get cited answers.")

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = db.load_chat()

if "docs" not in st.session_state:
    st.session_state.docs = []

# ================= LOAD DOCUMENTS =================
for f in os.listdir(UPLOAD_DIR):
    path = os.path.join(UPLOAD_DIR, f)
    if os.path.isfile(path) and f not in [d["name"] for d in st.session_state.docs]:
        st.session_state.docs.append(ingest_file(path, from_disk=True))

# ================= SIDEBAR =================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Chat", "Documents", "Monitoring & Analytics", "Chat History"]
)

# =================================================
# ======================= CHAT ====================
# =================================================
if page == "Chat":
    st.header("💬 Ask a question")
    question = st.text_input("Your question")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top-K retrieval", 1, 10, TOP_K_DEFAULT)
    with col2:
        conf_thresh = st.slider(
            "Confidence threshold", 0.0, 1.0, CONFIDENCE_DEFAULT
        )

    if st.button("Get Answer"):
        if not GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY missing")
        elif not question.strip():
            st.warning("Enter a question")
        elif not st.session_state.docs:
            st.warning("Upload documents first.")
        else:
            with st.spinner("Retrieving..."):
                results = retriever.retrieve(question, top_k)

            answer, is_generated = generate_answer(question, results)

            refs = []
            if results and not is_generated:
                refs = [f"Ref: {utils.format_reference(meta)}" for meta, _, _ in results]
            else:
                answer += "\n\n(Generated — not in documents)"

            # Compute confidence from retrieval scores
            confidence = 0.0
            if results:
                confidence = max(score for _, score, _ in results)  # 0.0–1.0
            confidence_percent = round(confidence * 100, 2)

            # Save chat with temporary feedback placeholder
            chat_entry = {
                "question": question,
                "answer": answer,
                "refs": refs,
                "confidence": confidence_percent,
                "rating": None,
                "comment": None
            }
            st.session_state.chat.append(chat_entry)
            db.save_chat(question, answer, refs, confidence_percent)

    # -------- Latest Answer & Feedback --------
    if st.session_state.chat:
        st.markdown("---")
        st.subheader("📝 Latest Response")
        last = st.session_state.chat[-1]
        st.write(f"**Q:** {last['question']}")
        st.write(f"**A:** {last['answer']}")
        st.metric("Confidence", f"{last.get('confidence', 0)}%")
        for r in last["refs"]:
            st.write(r)

        # Feedback form (if not already rated)
        if last.get("rating") is None:
            with st.form("feedback_form"):
                rating = st.slider("Rate this answer (1–5)", 1, 5, 3)
                comment = st.text_input("Comment (optional)")
                submit = st.form_submit_button("Submit Feedback")
                if submit:
                    last["rating"] = rating
                    last["comment"] = comment
                    db.save_feedback(last["question"], last["answer"], rating, comment)
                    st.success("✅ Feedback submitted successfully")

# =================================================
# ===================== DOCUMENTS ==================
# =================================================
elif page == "Documents":
    st.header("📂 Document Management")

    files = st.file_uploader(
        "Upload PDF / DOCX / TXT / MD / HTML",
        type=["pdf", "docx", "txt", "md", "html", "htm"],
        accept_multiple_files=True
    )

    if files:
        new_docs = []
        for f in files:
            path = os.path.join(UPLOAD_DIR, f.name)
            if os.path.exists(path):
                st.warning(f"Already uploaded: {f.name}")
                continue
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            doc = ingest_file(path, from_disk=True)
            st.session_state.docs.append(doc)
            new_docs.append(doc)

        if new_docs:
            retriever.index_document_chunks(new_docs, chunk_text)
            st.success("Documents indexed")

    st.subheader("Uploaded Documents")
    for d in st.session_state.docs:
        st.write(d["name"])

    if st.button("🔄 Reset Docs & Chat"):
        st.session_state.docs = []
        st.session_state.chat = []
        retriever.reset_store()
        db.clear_chat()
        for f in os.listdir(UPLOAD_DIR):
            os.remove(os.path.join(UPLOAD_DIR, f))
        st.success("Reset completed")

# =================================================
# =============== MONITORING & ANALYTICS ===========
# =================================================
elif page == "Monitoring & Analytics":
    st.header("📊 Monitoring & Analytics")

    chats = db.load_chat()
    feedback = db.load_feedback()

    # ---------- System ----------
    st.subheader("System Metrics")
    st.metric("Documents Uploaded", len(os.listdir(UPLOAD_DIR)))
    st.metric("Total Queries", len(chats))

    # ---------- Top Questions ----------
    st.subheader("Top 10 Frequently Asked Questions")
    questions = [c["question"] for c in chats]
    top_q = Counter(questions).most_common(10)
    if top_q:
        df = pd.DataFrame(top_q, columns=["Question", "Count"])
        st.dataframe(df)
    else:
        st.info("No questions yet")

    # ---------- Document Analytics ----------
    st.subheader("Document Analytics")
    refs = []
    for c in chats:
        for r in c.get("refs", []):
            if "Ref:" in r:
                refs.append(r.replace("Ref: ", "").split(" — ")[0])
    if refs:
        df_docs = pd.DataFrame(Counter(refs).items(), columns=["Document", "References"])
        st.bar_chart(df_docs.set_index("Document"))

    # ---------- Document Types ----------
    st.subheader("Document Type Distribution")
    doc_types = [f.split(".")[-1] for f in os.listdir(UPLOAD_DIR)]
    if doc_types:
        df_types = pd.DataFrame(Counter(doc_types).items(), columns=["Type", "Count"])
        fig, ax = plt.subplots()
        ax.pie(df_types["Count"], labels=df_types["Type"], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

    # ---------- Answer Quality ----------
    st.subheader("Answer Quality Metrics")
    cited = [c for c in chats if any("Ref:" in r for r in c.get("refs", []))]
    citation_rate = len(cited) / len(chats) if chats else 0
    st.metric("Citation Rate", f"{citation_rate*100:.2f}%")

    # ---------- Feedback ----------
    st.subheader("Feedback Analytics")
    if feedback:
        avg_rating = sum(f["rating"] for f in feedback) / len(feedback)
        low = len([f for f in feedback if f["rating"] <= 2])
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        st.metric("Low Rated Answers (≤2)", low)
    else:
        st.info("No feedback yet")

# =================================================
# =================== CHAT HISTORY =================
# =================================================
elif page == "Chat History":
    st.header("📜 Full Chat History with Feedback")

    chats = db.load_chat()  # load all chat + feedback

    if not chats:
        st.info("No chat history available.")
    else:
        for i, msg in enumerate(reversed(chats), 1):
            st.markdown(f"### Chat #{i}")
            st.write(f"**Q:** {msg['question']}")
            st.write(f"**A:** {msg['answer']}")
            st.write(f"**Confidence:** {msg.get('confidence', 0)}%")
            
            if msg.get("refs"):
                st.write("**References:**")
                for r in msg["refs"]:
                    st.write(f"- {r}")
            else:
                st.write("**References:** None")
            
            if msg.get("rating"):
                st.write(f"**Rating:** ⭐ {msg['rating']}")
                if msg.get("comment"):
                    st.write(f"💬 Comment: {msg['comment']}")
            else:
                st.write("**Feedback:** Not provided")
            
            st.markdown("---")

    # -------- Export Chat with Feedback to PDF --------
    if st.button("⬇ Export Full Chat to PDF"):
        pdf_bytes = utils.export_chat_to_pdf_bytes(chats)
        st.download_button(
            "Download chat as PDF",
            data=pdf_bytes,
            file_name="full_chat_history.pdf",
            mime="application/pdf"
        )