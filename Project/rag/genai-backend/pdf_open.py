import os

file_path = r"knowledge/visabulletin_December2025.pdf"

print("Exists:", os.path.exists(file_path))
print("Readable:", os.access(file_path, os.R_OK))

try:
    with open(file_path, "rb") as f:
        f.read(1)
    print("File opened successfully")
except Exception as e:
    print("Error opening file:", e)