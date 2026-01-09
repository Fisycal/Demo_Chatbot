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

def get_admin_emails():
    """Load admin emails from admin.txt file"""
    try:
        admin_file = os.path.join(os.path.dirname(__file__), 'admin.txt')
        with open(admin_file, 'r') as f:
            return [line.strip().lower() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading admin emails: {e}")
        return []

def login_with_credentials(email: str, password: str):
    """Authenticate user with email and password"""
    try:
        # Validate email format
        if not re.fullmatch(email_regex, email):
            st.error("Invalid email format")
            return False
        
        # Check if email is in admin list
        admin_emails = get_admin_emails()
        if email.lower() not in admin_emails:
            st.error("Email not authorized. Please contact administrator.")
            return False
        
        # Verify password
        if password != access_password:
            st.error("Incorrect password")
            return False
        
        # Call backend to get JWT token
        login_url = f"{backend_uri}/auth/login"
        try:
            response = httpx.post(
                login_url,
                json={"email": email, "password": password},
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state["jwt_token"] = data.get("access_token", "dummy_token")
                st.session_state["user_email"] = email.lower()
                st.session_state["is_admin"] = email.lower() in admin_emails
                logger.info(f"User {email} logged in successfully")
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                # If backend auth fails, use local auth as fallback
                logger.warning(f"Backend auth failed, using local auth for {email}")
                st.session_state["jwt_token"] = "local_token_" + email
                st.session_state["user_email"] = email.lower()
                st.session_state["is_admin"] = email.lower() in admin_emails
                st.success("Login successful!")
                st.rerun()
                return True
        except httpx.RequestError as e:
            # Backend not available, use local auth
            logger.warning(f"Backend not available, using local auth: {e}")
            st.session_state["jwt_token"] = "local_token_" + email
            st.session_state["user_email"] = email.lower()
            st.session_state["is_admin"] = email.lower() in admin_emails
            st.success("Login successful!")
            st.rerun()
            return True
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        st.error(f"Login failed: {str(e)}")
        return False



def main():
    # Streamlit page config
    st.set_page_config(
        page_title="TripPilot",
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
            submit = st.form_submit_button("Login")
            
            if submit:
                if email and password:
                    login_with_credentials(email, password)
                else:
                    st.error("Please enter both email and password")
        st.stop()

    # --- Authenticated Content ---
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("# TripPilot")
    with col2:
        if st.button("Log out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # If we reach here, user is authenticated:
    st.markdown("""
        Welcome to TripPilot!  
        Plan your perfect vacation with AI-powered recommendations.  
        Navigate to the TripPilot page to start planning your next adventure.
    """)

if __name__ == "__main__":
    main()