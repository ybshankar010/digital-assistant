# import os
# os.environ["STREAMLIT_WATCH_DISABLE_MODULES"] = "1"

import streamlit as st
st.set_page_config(page_title="Digital Assistant", layout="centered")
# st.title("💬 Digital Assistant with Indexing")
# st.query_params["disableWatchdogModules"] = "torch"

from digital_assistant.db.assistant_db import ChromaDB
from digital_assistant.etl.data_transformer import DataTransformer
from digital_assistant.agents.query_retriever import AgenticRAG


# --- Init ChromaDB & Agent ---
db = ChromaDB()
if "rag" not in st.session_state:
    st.session_state.rag = AgenticRAG(db)

# --- Sidebar for Indexing ---
st.sidebar.header("📄 Data Indexing")

uploaded_file = st.sidebar.file_uploader("Upload `.jsonl` file", type=["jsonl"])
batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=50, value=5)

if uploaded_file:
    temp_path = "temp_uploaded_data.jsonl"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success("File uploaded!")

    if st.sidebar.button("Start Indexing"):
        with st.spinner("Indexing..."):
            transformer = DataTransformer(db, file_path=temp_path)
            transformer.BATCH_SIZE = batch_size
            transformer.load_data_into_db()
        st.sidebar.success("✅ Data indexed successfully!")

# --- Main Chat Interface ---
st.header("💬 Chat with Your Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input (chat-style)
if user_prompt := st.chat_input("Type your question here..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag.run(user_prompt)
            assistant_reply = response.get("answer", "I'm not sure how to respond to that.")
            st.markdown(assistant_reply)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
