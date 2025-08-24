import os
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
import os
from rich.console import Console
from rich.panel import Panel

console = Console()

def build_pretrained_r3d18(num_classes=2, device=None, use_weights=True):
    """
    Build a pretrained R3D-18 model for video classification.
    """
    console.rule("[bold cyan]🚀 Building R3D-18 Model[/bold cyan]")

    if use_weights:
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        preprocess = weights.transforms()
        mean, std = preprocess.mean, preprocess.std
    else:
        model = r3d_18(weights=None)
        mean, std = None, None

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if device:
        model = model.to(device)

    console.print(Panel("[green]✅ R3D-18 Model Built Successfully[/green]", expand=False))
    return model, mean, std


def load_model(model_path, num_classes=2, device=None):
    """
    Load the trained R3D-18 model with custom weights.
    """
    model, mean, std = build_pretrained_r3d18(num_classes=num_classes, device=device, use_weights=False)

    if not os.path.exists(model_path):
        console.print(Panel(f"[bold red]❌ Model file not found:[/bold red] {model_path}", expand=False))
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    console.rule("[bold magenta]🎉 Model Ready[/bold magenta]")
    console.print(Panel(f"[bold green]✅ Successfully Loaded Model Weights[/bold green]\n📂 Path: {model_path}", expand=False))

    return model

def _sample_clip_from_video(video_path, frames_per_clip=16, frame_size=(112, 112)):
    """
    Sample a contiguous clip (T frames) from a video; returns array (T, H, W, C), RGB uint8.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    start_frame = 0 if total_frames <= frames_per_clip else random.randint(0, total_frames - frames_per_clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(frames_per_clip):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, frame_size)  # (W,H)
        frames.append(frame_rgb)

    cap.release()
    if len(frames) < frames_per_clip:
        return None

    return np.stack(frames, axis=0)  # (T, H, W, C)

def predict_video(model, video_path, device="cpu", mean=None, std=None, frames_per_clip=16, frame_size=(112, 112),
                  class_names=("No Theft", "Theft")):
    """
    Predict class + probability for a single uploaded video.
    Returns: (pred_class_idx, prob, inference_sec)
    """
    clip = _sample_clip_from_video(video_path, frames_per_clip=frames_per_clip, frame_size=frame_size)
    if clip is None:
        raise RuntimeError("Could not read enough frames from the video.")

    # Normalize to float, [0..1]
    clip = clip.astype(np.float32) / 255.0
    if mean is None: mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    if std  is None: std  = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    # Normalize per channel
    clip = (clip - mean) / std  # broadcasting over (T,H,W,C)

    # To torch: (1, C, T, H, W)
    clip_t = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
    clip_t = clip_t.to(device)

    with torch.no_grad():
        t0 = time.time()
        logits = model(clip_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        dt = time.time() - t0

    pred_idx = int(np.argmax(probs))
    pred_prob = float(probs[pred_idx])
    pred_label = class_names[pred_idx]
    return pred_label, pred_prob, dt
