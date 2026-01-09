import os
import re
import json
import time
import logging
import httpx
import uuid
from dotenv import load_dotenv
import streamlit as st

# --- CONFIG ---
st.set_page_config(
    initial_sidebar_state="expanded",
    layout="wide",
    page_title="My Document Chat",
    page_icon=":books:",
)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ENV ---
load_dotenv()
# load_dotenv(".env.server")
env_type = os.getenv("ENV_TYPE", "local")
backend_uri = os.getenv("BACKEND_URI", "http://localhost:8080")
chat_uri = f"{backend_uri}/info_chat"
doc_upload_uri = f"{backend_uri}/info_upload"
admin_list_uri = f"{backend_uri}/admin/collections/My-DOCU/documents"
admin_delete_uri = f"{backend_uri}/admin/collections/My-DOCU/documents/delete"

timeout = httpx.Timeout(60.0)

# --- INIT SESSION STATE ---
if "info_chat_history" not in st.session_state:
    st.session_state["info_chat_history"] = []

if "info_session_id" not in st.session_state:
    st.session_state["info_session_id"] = str(uuid.uuid4())

if "valid_email" not in st.session_state:
    st.session_state["valid_email"] = {}

# --- Require authentication from Home.py ---
if "jwt_token" not in st.session_state or "user_email" not in st.session_state:
    st.warning("Please log in from the Home page.")
    st.stop()

# --- UTILS ---
def create_jwt():
    return st.session_state["jwt_token"]

def new_chat():
    st.session_state["info_chat_history"] = []
    st.session_state["info_session_id"] = str(uuid.uuid4())
    # Change uploader key to force reset
    st.session_state["uploader_key"] = f"uploader_{uuid.uuid4()}"
    st.rerun()


def _get_admin_emails():
    """Load admin emails from admin.txt file."""
    try:
        with open(os.path.join(os.path.dirname(__file__), '../admin.txt'), 'r') as f:
            return [line.strip().lower() for line in f if line.strip()]
    except Exception:
        return []


def render_chat():
    for i, entry in enumerate(st.session_state["info_chat_history"]):
        if entry["sender"] == "user":
            st.markdown(
                f"<div style='background-color:#4DA6FF;padding:10px;border-radius:8px;color:white;width:fit-content;margin-left:auto;'>"
                f"{entry['message']}</div>",
                unsafe_allow_html=True
            )
        else:
            msg = entry['message']
            if "**Top Sources & Chunks:**" in msg:
                main, sources = msg.split("**Top Sources & Chunks:**", 1)
                st.markdown(
                    f"<div style='background-color:#E6F2FF;padding:10px;border-radius:8px;color:#333;width:fit-content;margin-right:auto;'>"
                    f"{main.strip()}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("**Top Sources & Contents:**")
                # Parse each numbered source and display with its index
                for match in re.finditer(r"(\d+)\. (\*\*.*?\*\* \(page .*?\)):\n(.*?)(?=\n\d+\. |\Z)", sources, re.DOTALL):
                    idx, src_block, chunk = match.groups()
                    preview = chunk.strip().replace("\n", " ")[:200] + ("..." if len(chunk.strip()) > 200 else "")
                    with st.expander(f"{idx}. {src_block}: {preview}"):
                        st.write(chunk.strip())
            else:
                st.markdown(
                    f"<div style='background-color:#E6F2FF;padding:10px;border-radius:8px;color:#333;width:fit-content;margin-right:auto;'>"
                    f"{msg}</div>",
                    unsafe_allow_html=True
                )
            if entry.get("feedback"):
                st.markdown(f"**Feedback:** {'👍' if entry['feedback']=='up' else '👎'}")

# --- MAIN ---
def main():
    st.title("Chat With Your Document")

    # Custom CSS for Upload Document button
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
        }
        </style>
    """, unsafe_allow_html=True)

    # Get authenticated user email
    email = st.session_state["user_email"]

    # --- NEW CHAT BUTTON before welcome ---
    if st.button("New Chat"):
        new_chat()

    # --- Welcome + session info ---
    st.markdown(f"Welcome **{email}**")
    st.markdown(f"<span style='font-size:small;color:gray;'>Session ID: {st.session_state['info_session_id']}</span>", unsafe_allow_html=True)

    # --- File upload form in sidebar only ---
    with st.sidebar:
        # # --- Model Selection Dropdown ---
        # st.markdown("## Select Chat Model")
        # if env_type == "server":
        #     model_options = ["gemma3", "Mistral-small3.2:latest","gpt-oss:20b","gpt-oss:120b"]
        # else:
        #     model_options = ["gemma3:latest", "mistral-small3.1:24b", "llama3.2:latest","gpt-oss:20b","gpt-oss:120b"]
        # selected_model = st.selectbox(
        #     "Models",
        #     model_options,
        #     index=0,
        #     help="Choose which LLM model to use for chat."
        # )
        # st.session_state["selected_model"] = selected_model
        # Only allow upload for users listed in admin.txt
        admin_emails = _get_admin_emails()
        if email.lower() in admin_emails:
            st.caption("Admin")
            st.subheader("Upload Documents")
            with st.form("upload_form"):
                uploaded_files = st.file_uploader(
                    "Upload Text/PDF files",
                    accept_multiple_files=True,
                    type=["pdf", "docx", "txt"],
                    label_visibility="collapsed",
                    key=st.session_state.get("uploader_key", "uploader_0")
                )
                submit_upload = st.form_submit_button("Upload Document")
        else:
            uploaded_files = None
            submit_upload = False

        # --- Admin Quick Link in Sidebar ---
        try:
            current_email = email.lower()
        except Exception:
            current_email = ""
        if current_email in admin_emails:
            st.divider()
            if st.button("Manage Documents", use_container_width=True, key="info_manage_docs_btn"):
                st.session_state["info_show_admin"] = True
                st.rerun()


    if uploaded_files and submit_upload:
        for file in uploaded_files:
            with st.spinner(f"Indexing {file.name}..."):
                try:
                    headers = {"Authorization": f"Bearer {create_jwt()}"}
                    files = {"file": (file.name, file, file.type)}
                    response = httpx.post(doc_upload_uri, files=files, headers=headers, timeout=75.0)
                    if response.status_code == 200:
                        success_placeholder = st.empty()
                        success_placeholder.success(f"{file.name} uploaded successfully!")
                        time.sleep(3)
                        success_placeholder.empty()
                    else:
                        st.error(f"Upload failed for {file.name}: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Upload error for {file.name}: {e}")

    # --- Admin Management Tab (in-page) ---
    admin_emails = _get_admin_emails()
    if st.session_state.get("info_show_admin") and (email.lower() in admin_emails):
        st.header("Manage Document Collection")
        cols = st.columns([1,1,6])
        with cols[0]:
            if st.button("Back to Chat", type="secondary", key="info_back_to_chat"):
                st.session_state["info_show_admin"] = False
                st.rerun()
        with cols[1]:
            if st.button("Refresh List", key="info_refresh_list"):
                st.session_state.pop("info_admin_docs", None)
                st.rerun()
        try:
            if "info_admin_docs" not in st.session_state:
                headers = {"Authorization": f"Bearer {create_jwt()}"}
                resp = httpx.get(admin_list_uri, headers=headers, timeout=30.0)
                if resp.status_code == 200:
                    st.session_state["info_admin_docs"] = resp.json().get("documents", [])
                else:
                    st.warning(f"Failed to load documents: {resp.status_code}")
            docs = st.session_state.get("info_admin_docs", [])
            if docs:
                filenames = [d.get("filename") for d in docs]
                counts_map = {d.get("filename"): d.get("point_count", 0) for d in docs}
                selected = st.multiselect(
                    "Select documents to delete",
                    filenames,
                    format_func=lambda x: f"{x} (points: {counts_map.get(x, 0)})",
                    key="info_delete_select"
                )
                if selected and st.button("Delete Selected", type="primary", key="info_delete_selected"):
                    headers = {
                        "Authorization": f"Bearer {create_jwt()}",
                        "Content-Type": "application/json",
                    }
                    payload = {"filenames": selected}
                    del_resp = httpx.post(admin_delete_uri, headers=headers, json=payload, timeout=60.0)
                    if del_resp.status_code == 200:
                        st.success(f"Deleted: {del_resp.json().get('deleted', {})}")
                        st.session_state.pop("info_admin_docs", None)
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {del_resp.status_code} - {del_resp.text}")
            else:
                st.info("No documents found in collection.")
        except Exception as e:
            st.error(f"Admin error: {e}")
        return

    # --- Chat Section ---
    st.markdown("---")
    render_chat()  # Render everything in history

    prompt = st.chat_input("Type your question...")
    if prompt:
        # Append user input to history
        st.session_state["info_chat_history"].append({"sender": "user", "message": prompt})

        # Immediately rerun so user input appears right away
        st.rerun()

    # The next rerun happens, now that user input is in history

    if st.session_state["info_chat_history"] and st.session_state["info_chat_history"][-1]["sender"] == "user":
        # This means we have a fresh user message and no assistant response yet
        assistant_placeholder = st.empty()
        raw_response = ""
        with st.spinner("Generating response..."):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {create_jwt()}"
                }
                data = {
                    "userInput": st.session_state["info_chat_history"][-1]["message"],
                    "temperature": 0.7,
                    "session_id": st.session_state["info_session_id"],
                    "selectedModel": st.session_state.get("selected_model", "llama3.2:latest")
                }
                with httpx.stream("POST", chat_uri, headers=headers, json=data, timeout=timeout) as response:
                    for chunk in response.iter_bytes():
                        if chunk:
                            decoded = chunk.decode("utf-8")
                            raw_response += decoded
                            assistant_placeholder.markdown(
                                f"<div style='background-color:#E6F2FF;padding:10px;border-radius:8px;color:#333;width:fit-content;margin-right:auto;'>"
                                f"{raw_response}</div>",
                                unsafe_allow_html=True
                            )
                            time.sleep(0.01)
                # Save to history
                st.session_state["info_chat_history"].append({
                    "sender": "assistant",
                    "message": raw_response.strip(),
                    "message_id": str(uuid.uuid4()),
                    "session_id": st.session_state["info_session_id"]
                })
                # Rerun so full chat shows in history
                st.rerun()
            except Exception as e:
                st.error(f"Chat error: {e}")



if __name__ == "__main__":
    main()
