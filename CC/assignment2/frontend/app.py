from flask import Flask, render_template, request, redirect
import requests

app = Flask(__name__)

# URL of the backend container inside Docker network
BACKEND_URL = "http://backend:5001"


@app.route("/", methods=["GET"])
def index():
    resp = requests.get(BACKEND_URL+"/api/message")
    data = resp.json()
    message = data['message']
    timestamp = ""
    if 'updated' in message:
        timestamp = message.split('updated at')[1].strip(')')
    return render_template('index.html', current_message=message, update_time=timestamp)


@app.route("/update", methods=["POST"])
def update():
    """
    TODO:
    - Get the value from the form field named "new_message"
    - Send a POST request to BACKEND_URL + "/api/message"
      with JSON body { "message": new_message }
    - Redirect back to "/"
    """
    new_message = request.form.get("new_message")
    payload = {"message": new_message}
    requests.post(f"{BACKEND_URL}/api/message", json=payload)
    return redirect('/')

# v2 TODO:
# - Change page title (in HTML)
# - Parse timestamp from backend message
# - Show "Last updated at: <timestamp>" in the template


if __name__ == "__main__":
    # Do not change the host or port
    app.run(host="0.0.0.0", port=5000)