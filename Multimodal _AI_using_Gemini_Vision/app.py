from flask import Flask, render_template, request
from google import genai
from PIL import Image
import os

app = Flask(__name__)

# ----------------------------
# Config
# ----------------------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# Gemini Client
# ----------------------------
client = genai.Client(api_key="AIzaSyDicGTsAh2uirpjz0Plw7egO9TJNIgcMiQ")

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    image_path = None

    if request.method == "POST":
        query = request.form.get("query")
        file = request.files.get("image")

        if file and query:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            image = Image.open(image_path)

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[image, query]
                )

                answer = response.text

            except Exception as e:
                answer = f"Error: {str(e)}"

    return render_template("index.html", answer=answer, image_path=image_path)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)