import streamlit as st
import requests

# ==========================================
# CONFIGURATION
# ==========================================
# Update this to match your FastAPI server address and prefix.
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="", layout="wide")

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
    st.title("Admin: Document Ingestion")
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
                    # Format files for the 'requests' library
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

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display metadata if it's an AI response
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Sources & Metadata", expanded=False):
                    meta = message["metadata"]
                    if meta.get("SQL Query Executed"):
                        st.markdown("**Generated SQL:**")
                        st.code(meta["SQL Query Executed"], language="sql")
                    
                    st.markdown(f"**📄 Document:** {meta.get('Document Name') or 'N/A'}")
                    st.markdown(f"**📑 Page(s):** {meta.get('Page No') or 'N/A'}")
                    st.markdown(f"**🔖 Citation:** {meta.get('Policy Citations') or 'N/A'}")
                    st.markdown(f"** SQL Query Executed:** {meta.get('SQL Query Executed')or 'N/A'}")

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                # Prepare the request payload
                payload = {"query": prompt}
                
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    
                    # Extract metadata mapping to your QueryResponse schema
                    metadata = {
                        "Document Name": data.get("document_name"),
                        "Page No": data.get("page_no"),
                        "Policy Citations": data.get("policy_citations"),
                        "SQL Query Executed": data.get("sql_query_executed")
                    }
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        with st.expander("Sources & Metadata", expanded=False):
                            
                            # Show SQL if present
                            if metadata["SQL Query Executed"]:
                                st.markdown("**Generated SQL:**")
                                st.code(metadata["SQL Query Executed"], language="sql")
                            
                            # Show Document sources cleanly (No more JSON!)
                            st.markdown(f"**📄 Document:** {metadata['Document Name'] or 'N/A'}")
                            st.markdown(f"**📑 Page(s):** {metadata['Page No'] or 'N/A'}")
                            st.markdown(f"**🔖 Citation:** {metadata['Policy Citations'] or 'N/A'}")
                            st.markdown(f"** SQL Query Executed:** {metadata['SQL Query Executed'] or 'N/A'}")

                    # Add assistant response to chat history
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