import streamlit as st
import requests
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="News QA with DeepSeek", page_icon="📰")
st.title("📰 Ask Questions About News URLs")
st.caption("Paste news article URLs and ask questions about their content.")
st.markdown("---")

# Session state for storing vector index
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

# Text area for URLs
st.subheader("🔗 Step 1: Paste News Article URLs (one per line)")
url_input = st.text_area("Enter URLs:", height=150, placeholder="https://example.com/article1\nhttps://example.com/article2")

if st.button("📥 Load & Process Articles"):
    with st.spinner("Loading and indexing documents..."):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if not urls:
            st.error("Please enter at least one valid URL.")
        else:
            try:
                loader = UnstructuredURLLoader(urls=urls)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_index = FAISS.from_documents(split_docs, embeddings)

                st.session_state.vector_index = vector_index
                st.success(f"✅ Indexed {len(split_docs)} chunks from {len(urls)} URLs.")
            except Exception as e:
                st.error(f"Failed to load documents: {e}")

# Ask a question
if st.session_state.vector_index:
    st.markdown("---")
    st.subheader("Step 2: Ask a Question")
    user_question = st.text_input("Your Question:", placeholder="What did Tesla announce?")
    
    if user_question:
        with st.spinner("Searching and generating answer..."):
            def query_lmstudio(prompt, retrieved_docs, model="deepseek-r1-distill-qwen-7b"):
                url = "http://localhost:1234/v1/chat/completions"
                headers = {"Content-Type": "application/json"}

                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                final_prompt = f"""Use the following article excerpts to answer the question.

{context}

Question: {prompt}
Answer:"""

                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": final_prompt}],
                    "temperature": 0.9,
                    "max_tokens": 500
                }

                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                except Exception as e:
                    return f"Error: {e}"

            docs = st.session_state.vector_index.similarity_search(user_question, k=4)
            answer = query_lmstudio(user_question, docs)
            sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

            st.subheader("Answer")
            st.write(answer.strip())

            st.subheader("📚 Sources")
            for i, src in enumerate(sources, 1):
                st.markdown(f"{i}. [{src}]({src})")
