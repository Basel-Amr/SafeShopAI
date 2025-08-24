import os
from django.shortcuts import render, redirect
from django.conf import settings
from .utils import load_model, predict_video
import torch

# Load once at startup
# 🔹 Global setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, "detection", "best_r3d18.pth")

# Load the model once when Django starts
MODEL = load_model(MODEL_PATH, num_classes=2, device=DEVICE)

# Optional: store mean/std if you want to normalize frames
MEAN, STD = None, None

def index(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]
        upload_dir = settings.MEDIA_ROOT / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save upload
        filename = video_file.name.replace(" ", "_")
        save_path = upload_dir / filename
        with open(save_path, "wb+") as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        return redirect("result", filename=filename)

    return render(request, "index.html", {})

def result(request, filename: str):
    file_path = settings.MEDIA_ROOT / "uploads" / filename
    pred_label, pred_prob, infer_sec = predict_video(
        MODEL, str(file_path), device=DEVICE, mean=MEAN, std=STD,
        frames_per_clip=16, frame_size=(112, 112),
        class_names=("No Theft", "Theft")
    )

    context = {
        "filename": filename,
        "video_url": f"{settings.MEDIA_URL}uploads/{filename}",
        "pred_label": pred_label,
        "pred_prob": f"{pred_prob*100:.2f}%",
        "infer_sec": f"{infer_sec:.3f}s",
        "badge_class": "danger" if pred_label == "Theft" else "success",
    }
    return render(request, "result.html", context)
