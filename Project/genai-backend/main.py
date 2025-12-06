import os
import json
import jwt
import logging
import datetime
from dotenv import load_dotenv
from fastapi import Query, HTTPException
import uvicorn
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse, JSONResponse #fast response compare to http
from starlette.middleware.cors import CORSMiddleware #filter header only request from this link should be processed
from pydantic import BaseModel
from typing import Optional, Annotated, Union, Any, AsyncGenerator, List, Dict #data validation
from fastapi import UploadFile, File, Form #to upload files
from utils.tools import update_qdrant_index
from utils.tools import list_documents_in_collection, delete_documents_by_filenames, ensure_collection_by_name
from utils.tools import StreamingOllamaAssistant, get_user_by_email, create_user, verify_user_credentials, update_user_password
import shutil #to move from one folder to another
from pathlib import Path

# --- initialize assistant & logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING", "")

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Authentication Configuration
client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")
access_password = os.getenv("ACCESS_PASSWORD", "password")

default_max_tokens = 4096
default_max_tokens_llama = 2048
default_temperature = 0.7
default_chatgpt_style = 0

chat_assistant = StreamingOllamaAssistant()


# --- Bootstrap: migrate admin.txt emails into MongoDB as admin users ---
def _migrate_admins_from_file():
    candidates = [
        Path(__file__).parent / "admin.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        email = line.strip()
                        if not email:
                            continue
                        try:
                            # use ACCESS_PASSWORD env as default for migrated admins
                            create_user(email, access_password, is_admin=True)
                            logger.info(f"Migrated admin user: {email}")
                        except Exception:
                            # user may already exist; that's fine
                            logger.debug(f"Admin user exists or could not be created: {email}")
        except Exception:
            continue


_migrate_admins_from_file()


# --- Authentication Functions ---
def authenticate(auth_token: Any) -> Optional[str]:
    """Authenticate user from JWT token and return email (personId)"""
    if not auth_token:
        return None
    try:
        bearer_token: str = auth_token.replace("Bearer ", "")
        output_payload: Dict[str, Any] = jwt.decode(bearer_token, client_secret, algorithms=["HS256"])
        
        if "personId" in output_payload:
            return str(output_payload["personId"])
        
        return None
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None
    
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    email: str
    message: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class UserRequest(BaseModel):  # type: ignore
    userInput: Optional[str] = None
    maxTokens: int = default_max_tokens_llama
    temperature: float = default_temperature
    #selectedModel: str = "llama3.2:latest"
    selectedModel: str = "gpt-4o-mini"
    chatgpt_style: int = default_chatgpt_style
    document_context: Optional[str] = None
    session_id: Optional[str] = None

UPLOAD_DIR = "knowledge"

# --- Authentication Endpoint ---
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint that validates credentials and returns JWT token"""
    try:
        # Validate email format
        if not request.email or "@" not in request.email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Verify credentials against MongoDB
        if not verify_user_credentials(request.email, request.password):
            # If user exists but wrong password or user doesn't exist, deny
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create JWT token with email as personId
        expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
        payload = {
            "personId": request.email.lower(),
            "email": request.email.lower(),
            "exp": expiration,
            "iat": datetime.datetime.utcnow()
        }
        
        token = jwt.encode(payload, client_secret, algorithm="HS256")
        
        logger.info(f"User {request.email} logged in successfully")
        
        return LoginResponse(
            access_token=token,
            email=request.email.lower(),
            message="Login successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.post("/auth/signup", response_model=LoginResponse)
async def signup(request: LoginRequest):
    """Endpoint for creating a new user. Returns a JWT on success."""
    try:
        if not request.email or "@" not in request.email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        # Create user in DB
        try:
            create_user(request.email, request.password, is_admin=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
        payload = {
            "personId": request.email.lower(),
            "email": request.email.lower(),
            "exp": expiration,
            "iat": datetime.datetime.utcnow()
        }
        token = jwt.encode(payload, client_secret, algorithm="HS256")
        return LoginResponse(access_token=token, email=request.email.lower(), message="Signup successful")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")


@app.post("/auth/change_password")
async def change_password(
    req: ChangePasswordRequest,
    Authorization: Annotated[Union[Any, None], Header()] = None,
):
    """Authenticated endpoint to change the current user's password.

    Requires Authorization: Bearer <token> header.
    """
    person_id = authenticate(Authorization)
    if not person_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Verify old password
    if not verify_user_credentials(person_id, req.old_password):
        raise HTTPException(status_code=401, detail="Old password is incorrect")

    try:
        updated = update_user_password(person_id, req.new_password)
        if not updated:
            raise Exception("Password update did not modify any records")
        return JSONResponse(status_code=200, content={"message": "Password changed successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error for {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/me")
def auth_me(Authorization: Annotated[Union[Any, None], Header()] = None):
    """Return basic profile info for the authenticated user."""
    person_id = authenticate(Authorization)
    if not person_id:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    user = get_user_by_email(person_id)
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    return JSONResponse(status_code=200, content={
        "email": user.get("email"),
        "is_admin": bool(user.get("is_admin", False)),
    })

#must be included all time:very important
# Info upload endpoint
default_info_collection = "My-DOCU"
@app.post("/info_upload")
async def info_upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(None),
):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks_added = await update_qdrant_index(file_path, collection_name=default_info_collection)
        logger.info(f"Info File uploaded for session_id: {session_id}")
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Info file uploaded and indexed successfully: {file.filename}",
                "chunks_added": chunks_added,
                "success": True,
            },
        )
    except Exception as e:
        logger.error(f"Info Upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to upload {file.filename}: {str(e)}", "session_id": session_id},
        )


# -----------------------------
# Admin: list & delete documents
# -----------------------------

def _is_admin(email: Optional[str]) -> bool:
    """Check if the given email belongs to an admin user stored in MongoDB."""
    if not email:
        return False
    user = get_user_by_email(email)
    if not user:
        return False
    return bool(user.get("is_admin", False))


class AdminDeleteRequest(BaseModel):
    filenames: List[str]


@app.get("/admin/collections/{collection}/documents")
def list_collection_documents(
    collection: str,
    Authorization: Annotated[Union[Any, None], Header()] = None,
):
    person_id = authenticate(Authorization)
    if not _is_admin(person_id):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    ensure_collection_by_name(collection)
    docs = list_documents_in_collection(collection)
    return JSONResponse(status_code=200, content={"collection": collection, "documents": docs})


@app.post("/admin/collections/{collection}/documents/delete")
def delete_collection_documents(
    collection: str,
    req: AdminDeleteRequest,
    Authorization: Annotated[Union[Any, None], Header()] = None,
):
    person_id = authenticate(Authorization)
    if not _is_admin(person_id):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    ensure_collection_by_name(collection)
    deleted_map = delete_documents_by_filenames(collection, req.filenames or [])
    return JSONResponse(status_code=200, content={"collection": collection, "deleted": deleted_map})
#very important
# Info chat endpoint
@app.post("/info_chat")
def info_process_chat(
    user_request: UserRequest,
    Authorization: Annotated[Union[Any, None], Header()] = None,
) -> Any:
    person_id = authenticate(Authorization)
    session_id = user_request.session_id
    if person_id:
        #selected_model = getattr(user_request, 'selectedModel', None) or "llama3.2:latest"
        selected_model = getattr(user_request, 'selectedModel', None) or "gpt-4o-mini"
        return StreamingResponse(chat_assistant.process_info_message(
            person_id=person_id,
            user_input=user_request.userInput,
            document_context=user_request.document_context,
            session_id=session_id,
            selected_model=selected_model,
            collection_name=default_info_collection
        ))
    else:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)



class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    feedback: str  # 'up' or 'down'

@app.post("/save_feedback")
def save_feedback(
    feedback_request: FeedbackRequest,
    Authorization: Annotated[Union[Any, None], Header()] = None,
):
    person_id = authenticate(Authorization)
    if person_id:
        history = chat_assistant.get_history(person_id=person_id, session_id=feedback_request.session_id)
        updated = False
        for msg in history:
            if msg.get("message_id") == feedback_request.message_id and (msg.get("role") == "assistant" or msg.get("sender") == "assistant"):
                msg.pop("feedback", None)
                if feedback_request.feedback == "up":
                    msg["thumbs_up"] = 1
                    msg["thumbs_down"] = 0
                elif feedback_request.feedback == "down":
                    msg["thumbs_up"] = 0
                    msg["thumbs_down"] = 1
                else:
                    msg["thumbs_up"] = None
                    msg["thumbs_down"] = None
                updated = True
                break
        if updated:
            chat_assistant.save_history(person_id=person_id, session_id=feedback_request.session_id, history=history)
            return StreamingResponse(content="Success", status_code=200)
        else:
            return StreamingResponse(content="Error: Message Not Found", status_code=401)
    else:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401) 

@app.get("/conversation_history")  # type: ignore
def get_conversation_history(
    Authorization: Annotated[Union[Any, None], Header()] = None,
    session_id: Optional[str] = Query(None)
) -> StreamingResponse:
    person_id = authenticate(Authorization)
    return StreamingResponse(retrieve_conversation(person_id, session_id=session_id))


async def retrieve_conversation(person_id: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    conversation_history = chat_assistant.get_history(person_id, session_id=session_id)
    yield json.dumps({"conversation_history": conversation_history}, default=str)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, timeout_keep_alive=60)
