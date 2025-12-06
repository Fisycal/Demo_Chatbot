```bash
cd genai-frontend

# Set Execution Policy on Windows for any 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

python -m venv frontvenv
source frontvenv/bin/activate # for mac
frontvenv\Scripts\activate # for windows
streamlit run Home.py
