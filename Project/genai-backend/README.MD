```bash
cd genai-backend

# create a virtual environment
python -m venv backvenv311
source backvenv311/bin/activate # for mac
backvenv311\Scripts\activate # for windows

# install dependencies
pip install -r requirements.txt

# run the app
python main.py or uvicorn main:app --reload

# stop the app
ctrl + c

