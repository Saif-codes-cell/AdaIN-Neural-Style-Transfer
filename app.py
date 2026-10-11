import os
from pathlib import Path

import torch
from flask import Flask, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, HiddenField, SubmitField
from wtforms.validators import InputRequired, NumberRange

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CONTENT_DIR = BASE_DIR / "content_data"
STYLE_DIR = BASE_DIR / "style_data"
EXAMPLES_DIR = BASE_DIR / "examples"

VGG_PATH = BASE_DIR / "vgg_normalised.pth"
DECODER_PATH = BASE_DIR / "experiment" / "decoder_final.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "change-this-development-secret"
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
Bootstrap(app)


class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField(
        "Style Strength",
        default=1.0,
        validators=[
            InputRequired(),
            NumberRange(min=0.0, max=1.0,
                        message="Style strength must be between 0 and 1."),
        ],
    )
    submit = SubmitField("Transfer Style")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not VGG_PATH.is_file():
    raise FileNotFoundError(f"VGG weights not found: {VGG_PATH}")
if not DECODER_PATH.is_file():
    raise FileNotFoundError(f"Decoder checkpoint not found: {DECODER_PATH}")

encoder = VGGEncoder(str(VGG_PATH)).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
encoder.eval()
decoder.eval()

for parameter in encoder.parameters():
    parameter.requires_grad = False


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def save_uploaded_file(file_storage):
    if not file_storage or not file_storage.filename:
        return None
    if not allowed_file(file_storage.filename):
        raise ValueError("Please upload a PNG, JPG, or JPEG image.")

    filename = secure_filename(file_storage.filename)
    if not filename:
        raise ValueError("The uploaded filename is invalid.")

    stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
    candidate = filename
    counter = 1
    while (UPLOAD_DIR / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    file_storage.save(UPLOAD_DIR / candidate)
    return candidate


def style_transfer(content_image, style_image, alpha):
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    content_tensor = image_transform(content_image).unsqueeze(0).to(device)
    style_tensor = image_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)
        target_features = adaptive_instance_normalization(
            content_features, style_features
        )
        target_features = (
            alpha * target_features + (1.0 - alpha) * content_features
        )
        generated = decoder(target_features)

    print(
        "Generated tensor statistics:",
        f"min={generated.min().item():.4f}",
        f"max={generated.max().item():.4f}",
        f"mean={generated.mean().item():.4f}",
        f"std={generated.std().item():.4f}",
    )
    return generated


def save_generated_image(image_tensor, destination):
    image_tensor = image_tensor.detach().cpu().squeeze(0).clamp(0, 1)
    transforms.ToPILImage()(image_tensor).save(destination)


@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            new_content = save_uploaded_file(form.content.data)
            new_style = save_uploaded_file(form.style.data)

            content_filename = (
                new_content or secure_filename(form.content_path.data or "")
            )
            style_filename = (
                new_style or secure_filename(form.style_path.data or "")
            )

            if not content_filename or not (UPLOAD_DIR / content_filename).is_file():
                raise ValueError("Please upload a valid content image.")
            if not style_filename or not (UPLOAD_DIR / style_filename).is_file():
                raise ValueError("Please upload a valid style image.")

            alpha = float(form.alpha.data)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("Style strength must be between 0 and 1.")

            try:
                with Image.open(UPLOAD_DIR / content_filename) as image:
                    content_image = image.convert("RGB")
                with Image.open(UPLOAD_DIR / style_filename) as image:
                    style_image = image.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(
                    "One of the uploaded files is not a readable image."
                ) from exc

            generated = style_transfer(content_image, style_image, alpha)
            result_filename = f"stylized_{Path(content_filename).stem}.png"
            save_generated_image(generated, UPLOAD_DIR / result_filename)
            result_image = result_filename

        except Exception as exc:
            app.logger.exception("Style transfer failed")
            error = str(exc) or "Style transfer failed. Check the terminal."

    elif form.is_submitted():
        error = "Please check the form fields and upload both images."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(str(EXAMPLES_DIR), filename)


@app.route("/content-images/<path:filename>")
def send_content_image(filename):
    return send_from_directory(str(CONTENT_DIR), filename)


@app.route("/style-images/<path:filename>")
def send_style_image(filename):
    return send_from_directory(str(STYLE_DIR), filename)


@app.errorhandler(413)
def file_too_large(_error):
    return "Image too large. Please upload an image smaller than 10 MB.", 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
