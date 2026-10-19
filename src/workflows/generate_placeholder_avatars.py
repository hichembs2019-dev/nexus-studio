"""Generates the six original placeholder character avatars under
assets/placeholder_characters/ - re-run this (`python3 -m
src.workflows.generate_placeholder_avatars`) only if those files are
ever lost; the committed PNGs are the real source of truth checked into
the repo, not this script's output at generation time."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "placeholder_characters")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 512, 512
SKIN = (235, 198, 165)

def draw_face(gender, age_group):
    img = Image.new("RGB", (W, H), (230, 230, 235))
    d = ImageDraw.Draw(img)

    head_cx, head_cy, head_r = W // 2, 260, 150

    hair_color = {"young": (40, 30, 25), "middle": (70, 60, 55), "older": (210, 210, 210)}[age_group]
    cloth_color = (60, 90, 160) if gender == "male" else (180, 70, 110)

    # shoulders/body
    d.ellipse([head_cx - 220, 430, head_cx + 220, 700], fill=cloth_color)

    # neck
    d.rectangle([head_cx - 30, head_cy + 110, head_cx + 30, head_cy + 170], fill=SKIN)

    # head
    d.ellipse([head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r], fill=SKIN)

    # hair (varies by gender/age - simple cap shape)
    if gender == "male":
        d.pieslice([head_cx - head_r - 5, head_cy - head_r - 20, head_cx + head_r + 5, head_cy + 40],
                   180, 360, fill=hair_color)
        if age_group == "older":
            # balding - smaller hair patch on sides only
            d.pieslice([head_cx - head_r - 5, head_cy - head_r - 20, head_cx + head_r + 5, head_cy + 10],
                       200, 340, fill=SKIN)
    else:
        d.pieslice([head_cx - head_r - 30, head_cy - head_r - 30, head_cx + head_r + 30, head_cy + 220],
                   170, 370, fill=hair_color)
        d.ellipse([head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r], fill=SKIN)

    # eyes
    eye_y = head_cy - 10
    d.ellipse([head_cx - 70, eye_y - 15, head_cx - 40, eye_y + 15], fill=(255, 255, 255))
    d.ellipse([head_cx + 40, eye_y - 15, head_cx + 70, eye_y + 15], fill=(255, 255, 255))
    d.ellipse([head_cx - 60, eye_y - 10, head_cx - 45, eye_y + 10], fill=(50, 35, 25))
    d.ellipse([head_cx + 45, eye_y - 10, head_cx + 60, eye_y + 10], fill=(50, 35, 25))

    # eyebrows
    brow_color = hair_color if age_group != "older" else (160, 160, 160)
    d.line([head_cx - 75, eye_y - 30, head_cx - 35, eye_y - 35], fill=brow_color, width=6)
    d.line([head_cx + 35, eye_y - 35, head_cx + 75, eye_y - 30], fill=brow_color, width=6)

    # nose
    d.line([head_cx, head_cy - 5, head_cx - 8, head_cy + 40], fill=(200, 160, 130), width=4)
    d.line([head_cx - 8, head_cy + 40, head_cx + 15, head_cy + 45], fill=(200, 160, 130), width=4)

    # mouth (neutral closed, clear region for the video model to animate)
    d.arc([head_cx - 45, head_cy + 55, head_cx + 45, head_cy + 100], 10, 170, fill=(150, 70, 70), width=6)

    # wrinkles for older
    if age_group == "older":
        for dx in (-50, 50):
            d.line([head_cx + dx - 10, head_cy - 40, head_cx + dx + 10, head_cy - 30], fill=(190, 160, 130), width=2)

    return img

ages = ["young", "middle", "older"]
genders = ["male", "female"]
for g in genders:
    for a in ages:
        img = draw_face(g, a)
        path = os.path.join(OUT_DIR, f"{g}_{a}.png")
        img.save(path)
        print("wrote", path)
