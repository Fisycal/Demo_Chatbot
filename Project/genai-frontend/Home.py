import streamlit as st
import httpx
import os
from dotenv import load_dotenv
import logging
import re


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
# Load environment variables from .env file
load_dotenv()
access_password = os.getenv("ACCESS_PASSWORD", "password")
backend_uri = os.getenv("BACKEND_URI", "http://localhost:8080")
cookie_expiry_days = 100

client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")

def get_admin_emails():
    """Deprecated local admin loader kept for compatibility (not used).
    Admin status is determined by the backend user record (`is_admin`)."""
    return []

def login_with_credentials(email: str, password: str):
    """Authenticate user with email and password"""
    try:
        # Validate email format
        if not re.fullmatch(email_regex, email):
            st.error("Invalid email format")
            return False
        # Call backend to login
        login_url = f"{backend_uri}/auth/login"
        try:
            response = httpx.post(login_url, json={"email": email, "password": password}, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                st.session_state["jwt_token"] = data.get("access_token")
                st.session_state["user_email"] = email.lower()
                # Fetch user profile (is_admin) from backend
                try:
                    me_url = f"{backend_uri}/auth/me"
                    headers = {"Authorization": f"Bearer {st.session_state.get('jwt_token','') }"}
                    me_resp = httpx.get(me_url, headers=headers, timeout=5.0)
                    if me_resp.status_code == 200:
                        profile = me_resp.json()
                        st.session_state["is_admin"] = bool(profile.get("is_admin", False))
                    else:
                        st.session_state["is_admin"] = False
                except Exception:
                    st.session_state["is_admin"] = False
                logger.info(f"User {email} logged in successfully")
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                msg = response.json().get("detail") if response.headers.get("content-type",""
                                                                     ).startswith("application/json") else response.text
                st.error(f"Login failed: {msg}")
                return False
        except httpx.RequestError as e:
            logger.error(f"Backend not available: {e}")
            st.error("Authentication service not available. Please try again later.")
            return False
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        st.error(f"Login failed: {str(e)}")
        return False



def main():
    # Streamlit page config
    st.set_page_config(
        page_title="Doc Chat",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Hide Streamlit footer and menu
    st.markdown(
        "<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>",
        unsafe_allow_html=True
    )

    # --- Authentication Gate ---
    if "jwt_token" not in st.session_state:
        st.warning("You must be logged in to use this app.")
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email", placeholder="user@example.com")
            password = st.text_input("Password", type="password")
            cols = st.columns([1,1])
            with cols[0]:
                submit = st.form_submit_button("Login")
            with cols[1]:
                signup_clicked = st.form_submit_button("Sign up")

            if submit:
                if email and password:
                    login_with_credentials(email, password)
                else:
                    st.error("Please enter both email and password")
            if signup_clicked:
                if not re.fullmatch(email_regex, email):
                    st.error("Enter a valid email to sign up")
                elif not password or len(password) < 6:
                    st.error("Please provide a password with at least 6 characters")
                else:
                    signup_url = f"{backend_uri}/auth/signup"
                    try:
                        resp = httpx.post(signup_url, json={"email": email, "password": password}, timeout=10.0)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.session_state["jwt_token"] = data.get("access_token")
                            st.session_state["user_email"] = email.lower()
                            # Fetch profile from backend to get is_admin
                            try:
                                me_url = f"{backend_uri}/auth/me"
                                headers = {"Authorization": f"Bearer {st.session_state.get('jwt_token','') }"}
                                me_resp = httpx.get(me_url, headers=headers, timeout=5.0)
                                if me_resp.status_code == 200:
                                    profile = me_resp.json()
                                    st.session_state["is_admin"] = bool(profile.get("is_admin", False))
                                else:
                                    st.session_state["is_admin"] = False
                            except Exception:
                                st.session_state["is_admin"] = False
                            st.success("Signup successful and logged in!")
                            st.rerun()
                        else:
                            detail = resp.json().get("detail") if resp.headers.get("content-type",""
                                                                        ).startswith("application/json") else resp.text
                            st.error(f"Signup failed: {detail}")
                    except httpx.RequestError as e:
                        logger.error(f"Signup request failed: {e}")
                        st.error("Signup service not available. Please try again later.")
        st.stop()

    # --- Authenticated Content ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("# My DOC CHAT")
    with col2:
        if st.button("Log out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # If we reach here, user is authenticated:
    st.markdown("""
        Welcome to the My Doc Chat Bot!  
        This application allows you to interact with uploaded documents using a chat interface.  
        You can ask questions about the documents, and the system will provide answers based on the content of the documents.
    """)

    # --- Change Password ---
    with st.expander("Change Password"):
        old_pw = st.text_input("Old password", type="password")
        new_pw = st.text_input("New password", type="password")
        new_pw_confirm = st.text_input("Confirm new password", type="password")
        if st.button("Change Password"):
            if not old_pw or not new_pw:
                st.error("Please fill both old and new password fields")
            elif new_pw != new_pw_confirm:
                st.error("New password and confirmation do not match")
            elif len(new_pw) < 6:
                st.error("Please choose a password with at least 6 characters")
            else:
                change_url = f"{backend_uri}/auth/change_password"
                headers = {"Authorization": f"Bearer {st.session_state.get('jwt_token','')}", "Content-Type": "application/json"}
                try:
                    resp = httpx.post(change_url, json={"old_password": old_pw, "new_password": new_pw}, headers=headers, timeout=10.0)
                    if resp.status_code == 200:
                        st.success("Password changed successfully. Please use the new password next time.")
                    else:
                        try:
                            detail = resp.json().get("detail")
                        except Exception:
                            detail = resp.text
                        st.error(f"Failed to change password: {detail}")
                except httpx.RequestError as e:
                    st.error(f"Request failed: {e}")

if __name__ == "__main__":
    main()