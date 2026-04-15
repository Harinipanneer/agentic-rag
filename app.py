import streamlit as st
import requests

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="wide")

# ==========================================
# SIDEBAR / TOGGLE
# ==========================================
st.sidebar.title("Navigation")
st.sidebar.markdown("Switch between User and Admin modes.")
app_mode = st.sidebar.radio("Select Mode", ["User Mode", "Admin Mode (Upload)"])

# ==========================================
# ADMIN MODE: FILE UPLOAD
# ==========================================
if app_mode == "Admin Mode (Upload)":
    st.title("🛡️ Admin: Document Ingestion")
    st.markdown("Upload multiple PDF documents to update the knowledge base.")

    uploaded_files = st.file_uploader(
        "Select PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if st.button("Upload and Ingest Documents"):
        if not uploaded_files:
            st.warning("Please select at least one PDF file to upload.")
        else:
            with st.spinner("Uploading and processing documents... This may take a while."):
                try:
                    files_payload = [
                        ("files", (file.name, file.getvalue(), "application/pdf")) 
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(f"{API_URL}/admin/upload", files=files_payload)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Successfully processed {res_data.get('files_processed')} files!")
                        with st.expander("View Ingestion Details"):
                            st.json(res_data)
                    else:
                        st.error(f"Failed to upload. Server responded with status code: {response.status_code}")
                        st.write(response.text)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Is your FastAPI server running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ==========================================
# USER MODE: QUERY
# ==========================================
elif app_mode == "User Mode":
    st.title("🤖 AI RAG Assistant")
    st.markdown("Ask questions about financial data, policies.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- CHAT HISTORY LOOP ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Sources & Metadata", expanded=False):
                    meta = message["metadata"]
                    
                    # Clean layout for the main metadata
                    st.markdown(f"**📄 Document:** {meta.get('Document Name') or 'N/A'}")
                    st.markdown(f"**📑 Page(s):** {meta.get('Page No') or 'N/A'}")
                    st.markdown(f"**🔖 Citation:** {meta.get('Policy Citations') or 'N/A'}")
                    
                    # Show SQL Query Executed
                    if meta.get("SQL Query Executed"):
                        st.markdown("**💻 SQL Query Executed:**")
                        st.code(meta["SQL Query Executed"], language="sql")
                    else:
                        st.markdown("**💻 SQL Query Executed:** N/A")
                    
                    # Vertical Raw Chunks Display
                    if meta.get("Source Chunks"):
                        st.markdown("---")
                        st.markdown("**🔍 Retrieved Context (Raw Chunks):**")
                        raw_text = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(meta["Source Chunks"])])
                        st.code(raw_text, language="text")

    # --- LIVE QUERY LOOP ---
    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})


        with st.spinner("Thinking..."):
            try:
                payload = {"query": prompt}
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    
                    metadata = {
                        "Document Name": data.get("document_name"),
                        "Page No": data.get("page_no"),
                        "Policy Citations": data.get("policy_citations"),
                        "SQL Query Executed": data.get("sql_query_executed"),
                        "Source Chunks": data.get("source_chunks") 
                    }
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        
                        with st.expander("Sources & Metadata", expanded=False):
                            
                            st.markdown(f"**📄 Document:** {metadata.get('Document Name') or 'N/A'}")
                            st.markdown(f"**📑 Page(s):** {metadata.get('Page No') or 'N/A'}")
                            st.markdown(f"**🔖 Citation:** {metadata.get('Policy Citations') or 'N/A'}")
                            
                            if metadata.get("SQL Query Executed"):
                                st.markdown("**💻 SQL Query Executed:**")
                                st.code(metadata["SQL Query Executed"], language="sql")
                            else:
                                st.markdown("**💻 SQL Query Executed:** N/A")
                            
                            if metadata.get("Source Chunks"):
                                st.markdown("---")
                                st.markdown("**🔍 Retrieved Context (Raw Chunks):**")
                                raw_text = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(metadata["Source Chunks"])])
                                st.code(raw_text, language="text")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": metadata
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure your FastAPI server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")