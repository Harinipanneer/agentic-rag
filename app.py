import streamlit as st
import requests

# ==========================================
# CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="", layout="wide")

# ==========================================
# SIDEBAR / TOGGLE
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["User Mode", "Admin Mode)"])

# ==========================================
# ADMIN MODE: FILE UPLOAD
# ==========================================
if app_mode == "Admin ":
    st.title(" Admin: Document Ingestion")
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
    st.title(" AI RAG Assistant")
    st.markdown("Ask questions about financial data and banking policies.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- CHAT HISTORY LOOP ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Sources & Metadata", expanded=False):
                    meta = message["metadata"]
                    st.markdown(f"** Document:** {meta.get('Document Name') or 'N/A'}")
                    st.markdown(f"** Page(s):** {meta.get('Page No') or 'N/A'}")
                    st.markdown(f"** Citation:** {meta.get('Citations') or 'N/A'}")
                    
                    if meta.get("SQL Query Executed"):
                        st.markdown("** SQL Query Executed:**")
                        st.code(meta["SQL Query Executed"], language="sql")
                    
                    if meta.get("Source Chunks"):
                        st.markdown("---")
                        st.markdown("** Retrieved Context (Raw Chunks):**")
                        # Join chunks with double newlines for vertical clarity
                        raw_text = "\n\n".join([f"{chunk}" for chunk in meta["Source Chunks"]])
                        st.code(raw_text, language="text")

    # --- LIVE QUERY LOOP ---
    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            try:
                payload = {"query": prompt}
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    raw_answer = data.get("answer", "No answer provided.")
                    
                    
                    clean_answer = raw_answer.replace("\\n", "\n")
                    
                    metadata = {
                        "Document Name": data.get("document_name"),
                        "Page No": data.get("page_no"),
                        "Citations": data.get("citation"),
                        "SQL Query Executed": data.get("sql_query_executed"),
                        "Source Chunks": data.get("source_chunks") 
                    }
                    
                    with st.chat_message("assistant"):
                      
                        with st.container():
                            st.markdown(clean_answer)
                        
                        with st.expander("Sources & Metadata", expanded=False):
                            st.markdown(f"** Document:** {metadata['Document Name'] or 'N/A'}")
                            st.markdown(f"** Page(s):** {metadata['Page No'] or 'N/A'}")
                            st.markdown(f"** Citation:** {metadata['Citations'] or 'N/A'}")
                            
                            if metadata["SQL Query Executed"]:
                                st.markdown("** SQL Query Executed:**")
                                st.code(metadata["SQL Query Executed"], language="sql")
                            
                            if metadata["Source Chunks"]:
                                st.markdown("---")
                                st.markdown("** Retrieved Context (Raw Chunks):**")
                                raw_chunks = "\n\n".join([f"{chunk}" for chunk in metadata["Source Chunks"]])
                                st.code(raw_chunks, language="text")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": clean_answer,
                        "metadata": metadata
                    })
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure your FastAPI server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")