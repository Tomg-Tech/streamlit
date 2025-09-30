# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:22:09 2025

@author: tomgr
"""

# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import cv2

# ---------------------------
# Minimal UNet + MC Dropout
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 32)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.up(x3)
        x = torch.cat([x1, x4], dim=1)
        x = self.upconv(x)
        return F.softmax(self.outc(x), dim=1)

@st.cache_resource
def load_model(device):
    model = UNetSmall(in_ch=1, n_classes=2)
    model.to(device)
    # If you have pretrained weights, load them here:
    # model.load_state_dict(torch.load("path_to_weights.pt", map_location=device))
    model.eval()
    return model

def preprocess_pil(img_pil, target_size=128):
    img = img_pil.convert("L")
    img = img.resize((target_size, target_size))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor

def mc_dropout_inference(model, x, n_iter=20, device="cpu"):
    model.train()  # enable dropout at inference
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            out = model(x.to(device))
            preds.append(out.cpu().numpy())
    preds = np.stack(preds)  # [n_iter, batch, classes, H, W]
    mean_pred = preds.mean(axis=0)
    uncertainty = preds.var(axis=0)
    return mean_pred, uncertainty

# ---------------------------
# Helper: contour extraction & saving
# ---------------------------
def extract_and_save_anomalies(uncertainty_map, input_image_np, threshold=0.2, out_dir="anomalies"):
    os.makedirs(out_dir, exist_ok=True)
    anomaly_mask = (uncertainty_map > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    saved = []
    for i, contour in enumerate(contours):
        mask = np.zeros_like(anomaly_mask, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask_path = os.path.join(out_dir, f"anomaly_mask_{i}.png")
        cv2.imwrite(mask_path, mask)
        x, y, w, h = cv2.boundingRect(contour)
        crop = (input_image_np[y:y+h, x:x+w] * 255).astype(np.uint8)
        crop_path = os.path.join(out_dir, f"anomaly_crop_{i}.png")
        cv2.imwrite(crop_path, crop)
        saved.append({"mask": mask_path, "crop": crop_path, "bbox": (x,y,w,h)})
    return saved

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Innovation — UNet Monitor", layout="wide")
st.title("AI Innovation — U-Net Self-Monitoring Demo")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Slides (edit & preview)")
    # default slides (you can change)
    default_slides = [
        {"title":"Opening: Who & Challenge", "body":"Intro. Problem: high costs in a critical process. Goal: use AI to deliver value."},
        {"title":"Problem", "body":"Slow process, repetitive tasks, high cost."},
        {"title":"Solution", "body":"Use AI for automation/prediction; integrate with workflow."},
        {"title":"Process", "body":"1) find pain 2) pick tool 3) implement 4) measure"},
        {"title":"Results", "body":"Cost savings: XX% — Time reduction: YY% — Better decisions."},
        {"title":"Conclusions", "body":"Start small, measure, scale."}
    ]
    # simple editor
    slides_text = st.text_area("Slides JSON (edit titles/bodies)", value=str(default_slides), height=220)
    preview = st.checkbox("Live preview", value=True)
    if st.button("Apply slides"):
        # naive parse: user can edit the Python list-of-dicts text. We keep simple.
        try:
            slides = eval(slides_text)  # NOTE: only for trusted local use
        except Exception as e:
            st.error(f"Failed to parse slides JSON: {e}")
            slides = default_slides
    else:
        slides = default_slides

    if preview:
        for s in slides:
            st.markdown(f"### {s['title']}")
            st.markdown(s['body'])
            st.markdown("---")

with col2:
    st.header("U-Net Uncertainty Monitor")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: {device}")

    model = load_model(device)

    uploaded = st.file_uploader("Upload image (grayscale or RGB)", type=["png","jpg","jpeg"])
    use_dummy = st.checkbox("Use dummy random image", value=False)
    n_iter = st.slider("MC Dropout iterations", min_value=5, max_value=100, value=30, step=5)
    threshold = st.slider("Uncertainty threshold", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

    if uploaded or use_dummy:
        if uploaded:
            img = Image.open(uploaded)
        else:
            # create a synthetic grayscale image
            img = Image.fromarray((np.random.rand(128,128)*255).astype(np.uint8))

        input_tensor = preprocess_pil(img, target_size=128)
        st.image(img, caption="Input image", use_column_width=True)

        if st.button("Run MC Dropout"):
            with st.spinner("Running inference..."):
                mean_pred, uncertainty = mc_dropout_inference(model, input_tensor, n_iter=n_iter, device=device)
                seg_map = np.argmax(mean_pred[0], axis=0)
                pred_uncertainty = np.max(uncertainty[0], axis=0)

                # Overlay visualization (alpha blend)
                input_np = np.array(img.convert("L").resize((128,128))).astype(np.float32)/255.0
                fig, axs = plt.subplots(1,3, figsize=(12,4))
                axs[0].imshow(input_np, cmap="gray")
                axs[0].set_title("Input")
                axs[0].axis("off")

                axs[1].imshow(seg_map, cmap="viridis")
                axs[1].set_title("Segmentation (argmax)")
                axs[1].axis("off")

                im = axs[2].imshow(input_np, cmap="gray")
                axs[2].imshow(pred_uncertainty, cmap="hot", alpha=0.5)
                axs[2].set_title("Overlay: Uncertainty")
                axs[2].axis("off")
                fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
                st.pyplot(fig)

                # Flag anomalies and save
                anomaly_mask = (pred_uncertainty > threshold).astype(np.uint8)
                contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                st.write(f"Found {len(contours)} anomaly regions (threshold {threshold})")

                if len(contours) > 0:
                    saved = extract_and_save_anomalies(anomaly_mask, input_np, threshold=threshold, out_dir="anomalies")
                    st.success(f"Saved {len(saved)} anomaly masks/crops to ./anomalies/")
                    # show thumbnails
                    thumbs = []
                    for s in saved:
                        crop = Image.open(s["crop"])
                        st.image(crop, width=160, caption=f"Crop {s['bbox']}")
                else:
                    st.info("No anomaly regions above threshold.")

    else:
        st.info("Upload an image or select dummy image to run the demo.")

st.markdown("---")
st.caption("For demo use only. Modify model loading to point at your real pretrained weights. Use eval-only on stable inputs in production.")
