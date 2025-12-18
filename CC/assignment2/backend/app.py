from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# Path to stored message file
DATA_PATH = "/data/message.txt"


def read_message():
    """
    TODO: 
    - If DATA_PATH exists, read and return the text inside
    - If it doesn't exist, return an empty string
    """
    if os.path.isfile(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            return content
    else:
        return ""


def write_message(msg: str):
    """
    TODO:
    - Open DATA_PATH
    - Write msg to the file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"{msg} (updated at {timestamp})"
    with open(DATA_PATH, "w+", encoding='utf-8') as f:
        f.write(msg)


@app.route("/api/message", methods=["GET"])
def get_message():
    """
    TODO:
    - Call read_message()
    - Return { "message": <stored message> } as JSON
    """
    message = read_message()
    return {"message": message}


@app.route("/api/message", methods=["POST"])
def update_message():
    """
    TODO:
    - Extract the field "message"
    - Call write_message() to save it
    - Return { "status": "ok" }
    """
    data = request.get_json()
    write_message(data["message"]) 
    return {"status": "ok"}


# v1 has no /api/health endpoint
# (Students add this in v2)
@app.route("/api/health", methods=["GET"])
def get_status():
    return {"status": "healthy"}
# v2 TODO:
# - Modify write_message() or update_message() to include a timestamp
#   Format: "<message> (updated at YYYY-MM-DD HH:MM:SS)"
#
# - Add new endpoint /api/health that returns:
#   { "status": "healthy" }


if __name__ == "__main__":
    # Do not change the host or port
    app.run(host="0.0.0.0", port=5001)
