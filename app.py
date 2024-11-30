from flask import Flask, render_template, request, redirect, url_for
import os
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import openai
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

# Define threshold values for pH, salinity, and moisture
threshold_ph = 6.5
threshold_salinity = 2.0
threshold_moisture = 30.0

# Email configuration
email_config = {
    "smtp_server": "smtp.mailgun.org",
    "smtp_port": 587,
    "sender_email": "postmaster@sandboxec8a7f76f5bb469bb7f31e9c82af7adf.mailgun.org",
    "sender_password": "358b8226c2dc0b0a79d066df5d48185a-3e508ae1-cbf7c9de",
    "receiver_email": "ikatoch72@gmail.com",
}


def send_email(attribute, value):
    msg = MIMEText(
        f"The {attribute} value has crossed the threshold. Current value: {value}"
    )
    msg["Subject"] = f"{attribute} Threshold Alert"
    msg["From"] = email_config["sender_email"]
    msg["To"] = email_config["receiver_email"]

    try:
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        server.starttls()
        server.login(email_config["sender_email"], email_config["sender_password"])
        server.sendmail(
            email_config["sender_email"],
            email_config["receiver_email"],
            msg.as_string(),
        )
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")


# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the Hugging Face model
model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# OpenAI API key (replace with your actual API key)
openai.api_key = os.environ.get["OPENAI_API_KEY"]

# Disease class labels
class_labels = {
    0: "Apple Scab",
    1: "Apple with Black Rot",
    2: "Cedar Apple Rust",
    3: "Healthy Apple",
    4: "Healthy Blueberry Plant",
    5: "Cherry with Powdery Mildew",
    6: "Healthy Cherry Plant",
    7: "Corn (Maize) with Cercospora and Gray Leaf Spot",
    8: "Corn (Maize) with Common Rust",
    9: "Corn (Maize) with Northern Leaf Blight",
    10: "Healthy Corn (Maize) Plant",
    11: "Grape with Black Rot",
    12: "Grape with Esca (Black Measles)",
    13: "Grape with Isariopsis Leaf Spot",
    14: "Healthy Grape Plant",
    15: "Orange with Citrus Greening",
    16: "Peach with Bacterial Spot",
    17: "Healthy Peach Plant",
    18: "Bell Pepper with Bacterial Spot",
    19: "Healthy Bell Pepper Plant",
    20: "Potato with Early Blight",
    21: "Potato with Late Blight",
    22: "Healthy Potato Plant",
    23: "Healthy Raspberry Plant",
    24: "Healthy Soybean Plant",
    25: "Squash with Powdery Mildew",
    26: "Strawberry with Leaf Scorch",
    27: "Healthy Strawberry Plant",
    28: "Tomato with Bacterial Spot",
    29: "Tomato with Early Blight",
    30: "Tomato with Late Blight",
    31: "Tomato with Leaf Mold",
    32: "Tomato with Septoria Leaf Spot",
    33: "Tomato with Spider Mites or Two-spotted Spider Mite",
    34: "Tomato with Target Spot",
    35: "Tomato Yellow Leaf Curl Virus",
    36: "Tomato Mosaic Virus",
    37: "Healthy Tomato Plant",
}


def perform_inference(image_path):
    # Load and preprocess the image for inference
    image = Image.open(image_path)
    input_features = extractor(images=image, return_tensors="pt")
    outputs = model(**input_features)
    predicted_class = outputs.logits.argmax().item()
    disease_label = class_labels.get(predicted_class, "Unknown")
    return disease_label


def generate_recommendations(disease_label):
    # Call the OpenAI API to generate recommendations based on the disease label
    prompt = f"Recommendations for treating {disease_label} in crops."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,  # Adjust based on your needs
    )
    recommendations = response.choices[0].text
    return recommendations


# Define your login authentication logic
def authenticate(email, password):
    # Add your authentication logic here
    # For simplicity, we'll assume any email and password combination is valid
    return True


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/process_login", methods=["POST"])
def process_login():
    email = request.form.get("email")
    password = request.form.get("password")

    if authenticate(email, password):
        return redirect("/index.html")
    else:
        return "Login failed. Please try again."


@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/adjust", methods=["GET", "POST"])
def adjust_attributes():
    if request.method == "POST":
        ph = float(request.form.get("ph"))
        salinity = float(request.form.get("salinity"))
        moisture = float(request.form.get("moisture"))

        print(
            f"Adjusted pH: {ph}, Adjusted Salinity: {salinity}, Adjusted Moisture: {moisture}"
        )

        if ph > threshold_ph:
            send_email("pH", ph)
        if salinity > threshold_salinity:
            send_email("Salinity", salinity)
        if moisture < threshold_moisture:
            send_email("Moisture", moisture)

    return render_template("adjust.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)

        # Perform inference on the uploaded image and get the disease label
        disease_label = perform_inference(filename)

        # Generate recommendations based on the disease label
        recommendations = generate_recommendations(disease_label)

        return (
            f"Predicted disease: {disease_label}<br>Recommendations: {recommendations}"
        )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/services")
def services():
    return render_template("service.html")


if __name__ == "__main__":
    app.run(debug=True)
