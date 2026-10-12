#!/usr/bin/env python3
"""
Animated 10-second event flyer renderer  (9:16 vertical, 1080x1920, 30 fps, mp4)

Design: bright gradient / pop.  All artwork is generated procedurally with
Pillow + numpy (no image assets), and every piece of copy comes from
`content.json`, so you can swap in your own event details without touching the
animation code.

    python render_flyer.py                     # -> out/event-flyer.mp4 (10 s)
    python render_flyer.py --preview           # key frames only, as PNGs
    python render_flyer.py --ss 3 --crf 16     # higher quality / slower

Requires: pillow, numpy  +  an `ffmpeg` binary (see setup.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"
OUT_DIR = ROOT                    # keeps deliverables out of excluded dir names

# ------------------------------------------------------------------------ geometry
W, H = 1080, 1920          # output size (9:16)
FPS = 30
DURATION = 10.0
FRAMES = int(round(FPS * DURATION))          # 300

S = 2                      # supersample factor (CLI)
W2, H2 = W * S, H * S
CX = W // 2                # horizontal centre (design space)

FONT_FILES = {
    "black": "Poppins-Black.ttf",
    "ebold": "Poppins-ExtraBold.ttf",
    "bold": "Poppins-Bold.ttf",
    "semi": "Poppins-SemiBold.ttf",
    "med": "Poppins-Medium.ttf",
    "ital": "Poppins-Italic.ttf",
    "italb": "Poppins-ExtraBoldItalic.ttf",
}
_fonts: dict = {}


def F(name: str, size: float) -> ImageFont.FreeTypeFont:
    """Font at *design* pixel size (automatically multiplied by the SS factor)."""
    key = (name, round(size, 1))
    f = _fonts.get(key)
    if f is None:
        f = ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), max(6, int(round(size * S))))
        _fonts[key] = f
    return f


# ---------------------------------------------------------------------- animation
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def seg01(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else (1.0 if t >= start else 0.0)


def out_cubic(x):
    return 1 - (1 - x) ** 3


def in_out_cubic(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def in_cubic(x):
    return x ** 3


def snap_out(x):
    """fast start, hard landing (pow 5 eases in)"""
    return 1 - (1 - x) ** 5


def spring(x):
    """settle with a ~6% overshoot - used for every 'pop'"""
    x = clamp(x)
    if x >= 1.0:
        return 1.0
    return 1 - math.exp(-6.5 * x) * math.cos(2 * math.pi * 1.15 * x)


def sheen(t):
    """eased 0..1 progress of the light sweep"""
    return seg01(t, 6.7, 2.3)


# -------------------------------------------------------------------------- colour
def rgba(t):
    if len(t) == 3:
        return (int(t[0]), int(t[1]), int(t[2]), 255)
    return (int(t[0]), int(t[1]), int(t[2]), int(t[3]))


def mix(c1, c2, k):
    k = clamp(k)
    return tuple(int(round(a + (b - a) * k)) for a, b in zip(c1, c2))


def recolor(layer: Image.Image, color) -> Image.Image:
    """Same alpha as *layer*, flat *color* fill."""
    r, g, b, a = rgba(color)
    out = Image.new("RGBA", layer.size, (r, g, b, 255))
    if a >= 255:
        out.putalpha(layer.getchannel("A"))
    else:
        out.putalpha(layer.getchannel("A").point(lambda v: v * a // 255))
    return out


def transform(layer: Image.Image, scale=1.0, angle=0.0) -> Image.Image:
    if angle:
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
    if abs(scale - 1.0) > 1e-3:
        nw = max(1, int(round(layer.width * scale)))
        nh = max(1, int(round(layer.height * scale)))
        layer = layer.resize((nw, nh), Image.LANCZOS)
    return layer


def paste(base: Image.Image, layer: Image.Image, center):
    """Alpha-composite *layer* centred on *center* (clipped to base)."""
    x = int(round(center[0] - layer.width / 2))
    y = int(round(center[1] - layer.height / 2))
    w, h = layer.size
    if x >= base.width or y >= base.height or x + w <= 0 or y + h <= 0:
        return
    base.alpha_composite(layer, (x, y))


def place_many(base, layer, center, scale=1.0, angle=0.0,
               soft_shadow=None, hard_shadow=None, extrude=None):
    """
    Draw one sticker / text layer with optional depth effects.
      soft_shadow = (dx, dy, rgba, blur)
      hard_shadow = (dx, dy, rgba)          # single flat offset copy
      extrude     = (dx, dy, rgba, steps)   # stacked flat copies = 3D block
    Coordinates are in design space and scaled by S internally.
    """
    lay = transform(layer, scale, angle)
    cx, cy = center

    if soft_shadow:
        dx, dy, col, blur = soft_shadow
        sh = recolor(lay, col).filter(ImageFilter.GaussianBlur(blur * S))
        paste(base, sh, (cx + dx * S, cy + dy * S))
    if extrude:
        dx, dy, col, steps = extrude
        ex = recolor(lay, col)
        for i in range(steps, 0, -1):
            k = i / steps
            paste(base, ex, (cx + dx * k * S, cy + dy * k * S))
    if hard_shadow:
        dx, dy, col = hard_shadow
        paste(base, recolor(lay, col), (cx + dx * S, cy + dy * S))

    paste(base, lay, (cx, cy))


# ------------------------------------------------------------------ text utilities
def text_width(text, f, tracking=0.0):
    return sum(f.getlength(ch) for ch in text) + tracking * (len(text) - 1 if len(text) > 1 else 0)


def text_layer(text, f, fill, tracking=0.0, stroke=0, stroke_fill=(0, 0, 0, 255),
               pad=0.0, anchor_v="baseline"):
    """
    Rasterise *text* (+letterspacing +outline) into a tight RGBA layer.
    The glyph baseline sits at `pad + above` inside the layer.
    """
    w = text_width(text, f, tracking)
    bb = f.getbbox(text or "H", anchor="ls")          # relative to baseline origin
    above = max(-bb[1], 1)
    below = max(bb[3], 0)
    pad = pad + stroke + 2
    lw = int(math.ceil(w + 2 * pad))
    lh = int(math.ceil(above + below + 2 * pad))
    img = Image.new("RGBA", (max(lw, 2), max(lh, 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad
    baseline = pad + above
    for ch in text:
        d.text((x, baseline), ch, font=f, fill=fill, anchor="ls",
               stroke_width=int(round(stroke)), stroke_fill=stroke_fill)
        x += f.getlength(ch) + tracking
    return img


def make_pill(text, f, tracking=0.0, pad_x=40, pad_y=20, radius=None,
              fill=(255, 255, 255, 255), grad=None, outline=(0, 0, 0, 255),
              outline_w=0, text_fill=(0, 0, 0, 255), stroke=0,
              stroke_fill=(0, 0, 0, 255)):
    """Rounded 'pill' badge with centred text; gradient fill optional."""
    tw = text_width(text, f, tracking)
    bb = f.getbbox(text or "H", anchor="ls")
    above = max(-bb[1], 1)
    below = max(bb[3], 0)
    h = above + below + 2 * pad_y
    w = tw + 2 * pad_x
    r = min(radius if radius is not None else h / 2, h / 2)

    shape = Image.new("L", (int(math.ceil(w)), int(math.ceil(h))), 0)
    ImageDraw.Draw(shape).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)

    img = Image.new("RGBA", shape.size, (0, 0, 0, 0))
    if grad is not None:
        c1, c2 = grad
        g = np.linspace(0, 1, shape.width, dtype=np.float32)[None, :, None]
        ramp = np.array(c1[:3], np.float32)[None, None, :] * (1 - g) + \
               np.array(c2[:3], np.float32)[None, None, :] * g
        ramp = np.repeat(ramp, shape.height, axis=0)
        grad_img = Image.fromarray(np.clip(ramp, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        grad_img.putalpha(shape)
        img.alpha_composite(grad_img)
    else:
        solid = Image.new("RGBA", shape.size, rgba(fill))
        solid.putalpha(shape.point(lambda v: v * rgba(fill)[3] // 255))
        img.alpha_composite(solid)

    d = ImageDraw.Draw(img)
    if outline_w > 0:
        d.rounded_rectangle([outline_w / 2, outline_w / 2, w - 1 - outline_w / 2, h - 1 - outline_w / 2],
                            radius=r, outline=rgba(outline), width=int(round(outline_w)))

    x = pad_x
    baseline = pad_y + above
    for ch in text:
        d.text((x, baseline), ch, font=f, fill=rgba(text_fill), anchor="ls",
               stroke_width=int(round(stroke)), stroke_fill=rgba(stroke_fill))
        x += f.getlength(ch) + tracking
    return img


def auto_fit(words, fname, text, tracking, max_w, size_from=260, size_min=70):
    """Largest Poppins size at which *text* fits inside *max_w* design pixels."""
    size = size_from
    while size > size_min:
        if text_width(text, F(fname, size), tracking * S) <= max_w * S:
            break
        size -= 2
    return size


# --------------------------------------------------------------------- decorations
def star_layer(outer, inner=None, color=(255, 209, 74, 255), outline=(30, 8, 40, 255),
               ow=0, points=5, spin=-90):
    inner = inner if inner is not None else outer * 0.46
    pad = ow + 2
    size = int(outer * 2 + pad * 2)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = size / 2
    pts = []
    for i in range(points * 2):
        r = outer if i % 2 == 0 else inner
        a = math.radians(spin + i * (360 / (points * 2)))
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    d.polygon(pts, fill=rgba(color))
    if ow > 0:
        d.line(pts + [pts[0]], fill=rgba(outline), width=int(round(ow)), joint="curve")
    return img


def ring_layer(r, width, color):
    pad = width + 2
    size = int(r * 2 + pad * 2)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([pad, pad, pad + r * 2, pad + r * 2], outline=rgba(color), width=int(round(width)))
    return img


def dot_layer(r, color):
    pad = 2
    size = int(r * 2 + pad * 2)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([pad, pad, pad + r * 2, pad + r * 2], fill=rgba(color))
    return img


def plus_layer(size, width, color):
    pad = 2
    s = int(size) + pad * 2
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s / 2
    w = width / 2
    d.rounded_rectangle([c - size / 2, c - w, c + size / 2, c + w], radius=w, fill=rgba(color))
    d.rounded_rectangle([c - w, c - size / 2, c + w, c + size / 2], radius=w, fill=rgba(color))
    return img


def sparkle_layer(size, color=(255, 255, 255, 255), glow=0.16):
    """4-point star with a soft bloom."""
    pad = 4
    s = int(size) + pad * 2
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = s / 2
    w = size * 0.055
    pts = [(c, pad), (c + w, c - w), (s - pad, c), (c + w, c + w),
           (c, s - pad), (c - w, c + w), (pad, c), (c - w, c - w)]
    d.polygon(pts, fill=rgba(color))
    bloom = img.filter(ImageFilter.GaussianBlur(size * glow))
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.alpha_composite(bloom)
    out.alpha_composite(img)
    return out


def dots_grid(cols, rows, spacing, r, color, falloff=(0.15, 0.15), drift=(0, 0)):
    """Halftone dot field; dot alpha fades out towards the lower-right."""
    pad = int(r) + 2
    w = int(cols * spacing + pad * 2)
    h = int(rows * spacing + pad * 2)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base_a = rgba(color)[3]
    d = ImageDraw.Draw(img)
    for j in range(rows):
        for i in range(cols):
            fx = i / max(cols - 1, 1)
            fy = j / max(rows - 1, 1)
            dist = math.hypot(fx - falloff[0], fy - falloff[1])
            a = int(base_a * clamp(1 - dist * 1.15) ** 2)
            if a <= 3:
                continue
            cx = pad + i * spacing + drift[0]
            cy = pad + j * spacing + drift[1]
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgba(color)[:3] + (a,))
    return img


# ----------------------------------------------------------------------- background
SMALL = (270, 480)                     # low-res canvas for the gradient (upscaled)


def background(t, pal):
    """Soft tri-stop gradient + screen-blended coloured lights, upscaled."""
    sw, sh = SMALL
    u = np.linspace(0, 1, sw, dtype=np.float32)[None, :]
    v = np.linspace(0, 1, sh, dtype=np.float32)[:, None]
    uu = np.repeat(u, sh, axis=0)
    vv = np.repeat(v, sw, axis=1)

    ang = math.radians(20 + 16 * math.sin(t * 0.23))
    p = uu * math.cos(ang) + vv * math.sin(ang)
    p = (p - p.min()) / max(p.max() - p.min(), 1e-6)

    top = np.array(pal["bg_top"], np.float32)
    mid = np.array(pal["bg_mid"], np.float32)
    bot = np.array(pal["bg_bot"], np.float32)
    a = np.clip(p / 0.5, 0, 1)[..., None]
    b = np.clip((p - 0.5) / 0.5, 0, 1)[..., None]
    a = a * a * (3 - 2 * a)
    b = b * b * (3 - 2 * b)
    col = np.where(p[..., None] < 0.5, top * (1 - a) + mid * a, mid * (1 - b) + bot * b)

    aspect = sw / sh
    lights = [
        # (cx, cy, radius, colour, strength)
        (0.20 + 0.09 * math.sin(t * 0.45), 0.10 + 0.05 * math.cos(t * 0.37), 0.55, pal["accent"], 0.60),
        (0.88 - 0.07 * math.cos(t * 0.33), 0.26 + 0.05 * math.sin(t * 0.52), 0.45, pal["accent2"], 0.42),
        (0.50 + 0.12 * math.sin(t * 0.29), 0.34, 0.62, pal["bg_top"], 0.35),
        (0.06, 0.72 + 0.05 * math.sin(t * 0.41), 0.50, pal["bg_bot"], 0.30),
        (0.95, 0.88, 0.55, pal["accent"], 0.26),
        (0.5, 0.05 + 0.03 * math.sin(t * 0.61), 0.40, (255, 120, 220), 0.22),
    ]
    for cx, cy, r, c, k in lights:
        d = np.sqrt(((uu - cx) * aspect) ** 2 + (vv - cy) ** 2)
        g = np.clip(1 - d / r, 0, 1) ** 2
        bcol = np.array(c, np.float32)[None, None, :] * (g * k)[..., None]
        col = 255 - (255 - col) * (255 - bcol) / 255.0

    small = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB")
    return small.resize((W2, H2), Image.BICUBIC).convert("RGBA")


def vignette_mask():
    x = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    r = np.sqrt(x ** 2 + (y * 0.98) ** 2) / 1.30
    return np.clip(1 - 0.34 * r ** 2.2, 0, 1)[..., None]


VIGNETTE = vignette_mask()


def sheen_mask(k):
    """Diagonal soft light band sweeping across the frame (post stage)."""
    cw, ch = W // 4, H // 4
    x = np.linspace(0, 1, cw, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, ch, dtype=np.float32)[:, None]
    s = (x * 0.86 + y * 0.5) / 1.36                    # ~30 deg sweep axis
    band = np.exp(-((s - (k * 1.5 - 0.25)) / 0.085) ** 2)
    edge = math.sin(math.pi * clamp(k)) ** 0.6          # fade in/out
    m = np.repeat((band * edge * 0.42)[..., None], 3, axis=2)
    m = np.clip(m, 0, 1)
    img = Image.fromarray((m * 255).astype(np.uint8), "RGB").resize((W, H), Image.BILINEAR)
    return np.asarray(img, np.float32) / 255.0


# -------------------------------------------------------------------------- sparkles
rng = np.random.default_rng(7)

SPARKLES = [dict(x=float(rng.uniform(70, W - 70)), y=float(rng.uniform(90, H - 90)),
                 s=float(rng.uniform(26, 74)), ph=float(rng.uniform(0, 6.28)),
                 sp=float(rng.uniform(1.1, 2.1)), rise=float(rng.uniform(6, 26)),
                 born=float(rng.uniform(0.9, 6.6)))
            for _ in range(20)]


def draw_sparkles(world, t):
    for sp in SPARKLES:
        a = seg01(t, sp["born"], 0.6)
        if a <= 0.01:
            continue
        tw = 0.5 + 0.5 * math.sin((t * sp["sp"] + sp["ph"]) * 2 * math.pi)
        alpha = a * (0.22 + 0.78 * tw ** 2)
        if alpha < 0.03:
            continue
        size = sp["s"] * (0.72 + 0.42 * tw)
        lay = sparkle_layer(size * S, (255, 255, 255, int(255 * alpha)))
        y = sp["y"] - sp["rise"] * (t - sp["born"])
        paste(world, lay, (sp["x"] * S, y * S))


# -------------------------------------------------------------------------- confetti
def make_confetti():
    parts = []
    bursts = [(4.30, 540, 980, 54, 0.60), (6.05, 540, 1240, 34, 0.30)]
    for bt, bx, by, n, upbias in bursts:
        for i in range(n):
            ang = rng.uniform(-math.pi, math.pi)
            # bias the burst sideways/upwards so it does not rain over the CTA
            ang = ang * 0.72 + (-math.pi / 2 if rng.random() < upbias else 0) * 0.5
            sp = rng.uniform(420, 1500)
            parts.append(dict(
                t0=bt + rng.uniform(0, 0.16),
                x0=bx + rng.uniform(-40, 40), y0=by + rng.uniform(-30, 30),
                vx=math.cos(ang) * sp, vy=math.sin(ang) * sp,
                drag=rng.uniform(1.5, 2.6), g=rng.uniform(700, 1150),
                life=rng.uniform(2.4, 3.9),
                w=rng.uniform(16, 40), h=rng.uniform(10, 26),
                rot=float(rng.uniform(0, 360)), spin=float(rng.uniform(-6, 6)),
                col=int(rng.integers(0, 4)), round_=rng.random() < 0.35))
    return parts


CONFETTI = make_confetti()


def draw_confetti(world, pal, t):
    cols = [pal["accent"], (255, 255, 255), pal["accent2"], (255, 140, 210)]
    for p in CONFETTI:
        age = t - p["t0"]
        if age < -0.001 or age > p["life"]:
            continue
        if age < 0:
            continue
        tau = (1 - math.exp(-p["drag"] * age)) / p["drag"]
        x = p["x0"] + p["vx"] * tau
        y = p["y0"] + p["vy"] * tau + 0.5 * p["g"] * age * age * 0.6
        if y > H + 120 or x < -160 or x > W + 160:
            continue
        fade_in = clamp(age / 0.10)
        fade_out = clamp((p["life"] - age) / 0.55)
        al = int(232 * min(fade_in, fade_out))
        if al < 4:
            continue
        scale = 0.62 + 0.38 * clamp(age / 0.18)
        w = p["w"] * scale
        h = p["h"] * scale
        pad = 4
        piece = Image.new("RGBA", (int(w + pad * 2), int(h + pad * 2)), (0, 0, 0, 0))
        d = ImageDraw.Draw(piece)
        col = cols[p["col"] % len(cols)] + (al,)
        if p["round_"]:
            d.ellipse([pad, pad, pad + w, pad + h], fill=col)
        else:
            d.rounded_rectangle([pad, pad, pad + w, pad + h], radius=min(4 * S, h / 2), fill=col)
        piece = transform(piece, 1.0, p["rot"] + p["spin"] * age * 57.3)
        paste(world, piece, (x * S, y * S))


# --------------------------------------------------------------------- layout grid
BADGE_Y = 186
LINE1_Y, LINE2_Y = 676, 876
TAGLINE_Y = 1016
CARD_Y = 1166
VENUE_Y, CITY_Y = 1292, 1366
DOORS_Y = 1440
CTA_Y = 1570
MARQUEE_Y = 1710
FOOTER_Y = 1815



MARQUEE_SPEED = 168.0            # ribbon scroll speed, design px / second


# ----------------------------------------------------------------------------- title
def draw_title(world, cfg, t, pal):
    lines = cfg["title"][:2]
    ys = [LINE1_Y, LINE2_Y]
    for li, line in enumerate(lines):
        size = auto_fit(None, "black", line, 0, 790, size_from=248, size_min=76)
        f = F("black", size)
        widths = [f.getlength(ch) for ch in line]
        total = sum(widths)
        cx0 = CX * S - total / 2
        for ci, ch in enumerate(line):
            start = 1.10 + li * 0.16 + ci * 0.085
            q = seg01(t, start, 0.75)
            x = cx0 + sum(widths[:ci])
            if q <= 0.003:
                continue
            p = spring(q)
            lay = text_layer(ch, f, (255, 255, 255, 255), stroke=11, stroke_fill=pal["ink"])
            drop = (1 - p) * -80
            ang = (1 - p) * (8 if ci % 2 else -8)
            scale = 0.3 + 0.7 * p
            place_many(world, lay, (x + widths[ci] / 2, (ys[li] + drop) * S),
                       scale=scale, angle=ang,
                       soft_shadow=(18, 26, (10, 0, 26, 90), 16),
                       extrude=(12, 16, pal["ink"] + (255,), 6))


# --------------------------------------------------------------------------- marquee
def marquee_tile(cfg, pal):
    """One repeating unit of the scrolling ribbon: text, star, text, star ..."""
    f = F("ebold", 42)
    items = cfg.get("marquee") or [cfg["hashtag"], "FREE ENTRY", "ONE NIGHT ONLY"]
    star = star_layer(13 * S, color=pal["accent"] + (255,), outline=pal["ink"] + (255,), ow=3 * S)
    seq = []
    GAP, SGAP = 34 * S, 26 * S
    for it in items:
        seq.append((text_layer(it, f, pal["accent"] + (255,), tracking=3 * S), GAP))
        seq.append((star, SGAP))
    total = int(sum(l.width + g for l, g in seq))
    h = 104 * S
    tile = Image.new("RGBA", (total, h), (0, 0, 0, 0))
    x = 0
    for lay, gap in seq:
        paste(tile, lay, (x + lay.width / 2, h / 2))
        x += lay.width + gap
    return tile


def draw_marquee(world, cfg, pal, t):
    t0, wipe = 4.55, 0.6
    if t < t0:
        return
    tile = marquee_tile(cfg, pal)
    tw = tile.width
    strip_w = (W + 500) * S
    strip = Image.new("RGBA", (strip_w, 104 * S), pal["ink"] + (242,))
    off = -((t - t0) * MARQUEE_SPEED * S) % tw
    x = off - tw
    while x < strip_w:
        paste(strip, tile, (x + tw / 2, 52 * S))
        x += tw
    d = ImageDraw.Draw(strip)
    for y in (9 * S, 95 * S):                       # thin accent rules top & bottom
        d.rectangle([0, y - 2 * S, strip_w, y + 2 * S], fill=pal["accent"] + (255,))

    strip = transform(strip, 1.0, -4.0)

    k = seg01(t, t0, wipe)                          # soft wipe-in from the left
    if k < 0.999:
        a = np.asarray(strip.getchannel("A"), np.float32) / 255.0
        xx = np.linspace(0, 1, strip.width, dtype=np.float32)[None, :]
        mask = np.clip((k * 1.7 - xx) / 0.22, 0, 1)
        mask = mask * mask * (3 - 2 * mask)
        strip.putalpha(Image.fromarray((a * mask * 255).astype(np.uint8), "L"))

    y = MARQUEE_Y * S + (1 - out_cubic(seg01(t, t0, 0.7))) * 34 * S
    paste(world, strip, (CX * S, y))


# --------------------------------------------------------------------------- decor
DECOR = [
    # (kind, x, y, size, colour key, born, angle, drift_x, drift_y)
    ("star", 912, 236, 130, "accent", 0.72, -14, 10, -14),
    ("ring", 978, 652, 78, "white", 1.05, 0, -12, 18),
    ("plus", 152, 366, 54, "accent2", 0.95, 0, 12, -16),
    ("dot", 110, 654, 34, "accent", 1.20, 0, -8, 22),
    ("ring", 1052, 1188, 100, "accent", 1.35, 0, 14, 10),
    ("plus", 62, 1204, 38, "white", 1.55, 0, -10, 14),
    ("ring", 120, 830, 92, "accent", 1.70, 0, 10, -14),
    ("dot", 1006, 1320, 26, "accent2", 1.85, 0, -12, 16),
    ("star", 128, 1006, 74, "white", 2.00, 18, 12, 12),
    ("plus", 966, 1420, 42, "accent2", 4.30, 0, -10, 14),
]


def draw_decor(world, cfg, t, pal):
    colmap = {"accent": pal["accent"] + (255,), "accent2": pal["accent2"] + (255,),
              "white": (255, 255, 255, 235), "ink": pal["ink"] + (255,)}
    for kind, x0, y0, size, ck, born, ang, dx, dy in DECOR:
        p = seg01(t, born, 0.55)
        if p <= 0.004:
            continue
        q = spring(p)
        s = 0.2 + 0.8 * q
        x = x0 + dx * math.sin((t - born) * 0.5)
        y = y0 + dy * math.sin((t - born) * 0.66)
        col = colmap[ck]
        if kind == "star":
            lay = star_layer(size * S / 2, color=pal["accent"] + (255,),
                             outline=pal["ink"] + (255,), ow=7 * S)
        elif kind == "ring":
            lay = ring_layer(size * S / 2, 9 * S, col)
        elif kind == "dot":
            lay = dot_layer(size * S / 2, col)
        else:
            lay = plus_layer(size * S, 11 * S, col)
        place_many(world, lay, (x * S, y * S), scale=s, angle=ang + 40 * (1 - q),
                   soft_shadow=(10, 14, (10, 0, 26, 70), 10) if kind != "plus" else None)


# ----------------------------------------------------------------------------- HUD
def draw_hud(world, cfg, t, pal):
    """badge -> tagline -> date sticker -> venue -> doors -> CTA -> ribbon -> footer"""
    ink = pal["ink"] + (255,)
    white = (255, 255, 255, 255)
    accent = pal["accent"] + (255,)

    # --- badge -------------------------------------------------------------
    p = seg01(t, 0.30, 0.75)
    if p > 0.004:
        q = spring(p)
        lay = make_pill(cfg["badge"], F("semi", 37), tracking=6.5 * S, pad_x=46 * S, pad_y=22 * S,
                        fill=white, outline=ink, outline_w=7 * S, text_fill=ink)
        place_many(world, lay, (CX * S + (1 - q) * -60 * S, BADGE_Y * S), scale=0.55 + 0.45 * q,
                   angle=(1 - q) * 12, soft_shadow=(12, 16, (10, 0, 26, 80), 12))

    # --- pop flash behind the title ---------------------------------------
    fp = seg01(t, 1.06, 0.42)
    if 0 < fp < 1:
        r = (150 + 900 * out_cubic(fp)) * S
        flash = Image.new("RGBA", (int(r * 2), int(r * 2)), (0, 0, 0, 0))
        ImageDraw.Draw(flash).ellipse([0, 0, r * 2, r * 2],
                                      fill=(255, 255, 255, int(140 * (1 - fp) ** 2)))
        flash = flash.filter(ImageFilter.GaussianBlur(r * 0.22))
        paste(world, flash, (CX * S, (LINE1_Y + LINE2_Y) / 2 * S))

    draw_title(world, cfg, t, pal)

    # --- tagline (rule - text - rule) --------------------------------------
    p = seg01(t, 2.92, 0.55)
    if p > 0.004 and cfg.get("tagline"):
        q = spring(p)
        f = F("semi", 38)
        tl = text_layer(cfg["tagline"], f, accent, tracking=6 * S,
                        stroke=6 * S, stroke_fill=ink)
        rw, gap, h = 70 * S, 30 * S, 84 * S
        w = rw * 2 + gap * 2 + tl.width
        lay = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
        paste(lay, tl, (w / 2, h / 2))
        d = ImageDraw.Draw(lay)
        for x0 in (0, w - rw):
            d.rounded_rectangle([x0, h / 2 - 4 * S, x0 + rw, h / 2 + 4 * S],
                                radius=4 * S, fill=accent)
        place_many(world, lay, (CX * S, TAGLINE_Y * S + (1 - q) * 22 * S),
                   scale=0.7 + 0.3 * q)

    # --- date sticker ------------------------------------------------------
    p = seg01(t, 3.16, 0.44)
    if p > 0.004:
        q = snap_out(p)
        card = make_pill(cfg["date"], F("black", 92), tracking=3 * S, pad_x=54 * S, pad_y=30 * S,
                         radius=30 * S, fill=ink, outline=accent, outline_w=7 * S,
                         text_fill=accent)
        cw, ch = card.width / S, card.height / S
        place_many(world, card, (CX * S, CARD_Y * S + (1 - q) * -34 * S),
                   scale=1.95 - 0.95 * q, angle=-1.6,
                   soft_shadow=(16, 22, (10, 0, 26, 110), 18))
        if q > 0.95:                                     # sticky-tape corners
            tape = Image.new("RGBA", (int(122 * S), int(44 * S)), (0, 0, 0, 0))
            ImageDraw.Draw(tape).rectangle([0, 0, 122 * S, 44 * S], fill=(255, 238, 178, 175))
            for sgn in (-1, 1):
                place_many(world, tape, (CX * S + sgn * cw * 0.40 * S, (CARD_Y - ch * 0.34) * S),
                           angle=sgn * -16)

    # --- venue + city ------------------------------------------------------
    for text, fname, size, y, start in ((cfg["venue"], "ebold", 52, VENUE_Y, 3.48),
                                        (cfg["city"], "bold", 38, CITY_Y, 3.60)):
        p = seg01(t, start, 0.5)
        if p <= 0.004:
            continue
        q = spring(p)
        lay = text_layer(text, F(fname, size), white, tracking=3 * S,
                         stroke=6 * S, stroke_fill=ink)
        place_many(world, lay, (CX * S, y * S + (1 - q) * 26 * S), scale=0.6 + 0.4 * q,
                   soft_shadow=(9, 12, (10, 0, 26, 70), 10))

    # --- doors pill --------------------------------------------------------
    p = seg01(t, 3.82, 0.5)
    if p > 0.004:
        q = spring(p)
        lay = make_pill(cfg["doors"], F("bold", 30), tracking=5 * S, pad_x=34 * S, pad_y=17 * S,
                        radius=22 * S, fill=(255, 255, 255, 55),
                        outline=(255, 255, 255, 235), outline_w=4 * S, text_fill=white)
        place_many(world, lay, (CX * S, DOORS_Y * S + (1 - q) * 20 * S), scale=0.6 + 0.4 * q)

    draw_marquee(world, cfg, pal, t)

    # --- CTA button (keeps breathing after it lands) ------------------------
    p = seg01(t, 5.20, 0.86)
    if p > 0.004:
        q = spring(p)
        breath = 1.0 + 0.024 * math.sin(2 * math.pi * 0.5 * (t - 5.9)) * seg01(t, 5.9, 0.8)
        lay = make_pill(cfg["cta"], F("ebold", 36), tracking=2.5 * S, pad_x=42 * S, pad_y=26 * S,
                        radius=999, grad=(pal["accent"], (255, 122, 60)),
                        outline=ink, outline_w=7 * S, text_fill=ink)
        place_many(world, lay, (CX * S, CTA_Y * S + (1 - q) * 40 * S),
                   scale=(0.55 + 0.45 * q) * breath,
                   soft_shadow=(13, 17, (10, 0, 26, 95), 13))

    # --- footer ------------------------------------------------------------
    p = seg01(t, 5.62, 0.7)
    if p > 0.004:
        q = spring(p)
        lay = text_layer(cfg["footer"], F("med", 22), (255, 255, 255, 190), tracking=7 * S)
        place_many(world, lay, (CX * S, FOOTER_Y * S), scale=0.8 + 0.2 * q)

    # --- halftone corner ---------------------------------------------------
    p = seg01(t, 4.72, 0.9)
    if p > 0.02:
        q = spring(p)
        dots = dots_grid(7, 6, 34 * S, 7 * S, (255, 255, 255, 120), falloff=(0.0, 0.0))
        place_many(world, dots, (250 * S, 396 * S), scale=0.8 + 0.2 * q,
                   angle=12 + 6 * math.sin(t * 0.6))


# --------------------------------------------------------------------------- render
def camera_scale(t):
    return 1.0 + 0.035 * in_out_cubic(seg01(t, 6.4, 3.6)) + 0.006 * (1 + math.sin(2 * math.pi * 0.42 * t)) / 2


def post_process(img: Image.Image, t) -> Image.Image:
    """bloom + sheen + vignette + film grain, all in numpy"""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    arr = np.asarray(rgb, np.float32)

    small = rgb.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(7))
    bloom = np.asarray(small.resize((W, H), Image.BILINEAR), np.float32)
    arr = 255.0 - (255.0 - arr) * (255.0 - bloom * 0.34) / 255.0

    k = sheen(t)
    if 0.0 < k < 1.0:
        arr += sheen_mask(k) * 92.0

    arr *= VIGNETTE
    arr += np.random.normal(0.0, 3.4, arr.shape[:2])[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def render_frame(idx: int, cfg, pal) -> Image.Image:
    t = idx / FPS
    world = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    draw_decor(world, cfg, t, pal)
    draw_hud(world, cfg, t, pal)
    draw_confetti(world, pal, t)
    draw_sparkles(world, t)

    frame = background(t, pal)
    frame.alpha_composite(world)

    # light pop flash right after the title lands
    fa = seg01(t, 1.34, 0.16)
    if fa < 1.0:
        a = int(70 * (1 - fa) ** 2)
        if a > 2:
            flash = Image.new("RGBA", (W2, H2), (255, 255, 255, a))
            frame.alpha_composite(flash)

    frame = frame.reduce(S)                              # 2x -> 1x, box filter AA

    k = camera_scale(t)                                  # slow push-in
    if abs(k - 1.0) > 1e-4:
        nw, nh = int(round(W * k)), int(round(H * k))
        frame = frame.resize((nw, nh), Image.LANCZOS)
        ox, oy = (nw - W) // 2, (nh - H) // 2
        frame = frame.crop((ox, oy, ox + W, oy + H))

    return post_process(frame, t)


# ----------------------------------------------------------------------------- main
def find_ffmpeg() -> str | None:
    here = Path(__file__).resolve().parent
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg"),
              here / ".tools/ffmpeg",
              "/tmp/tools/node_modules/@ffmpeg-installer/linux-x64/ffmpeg"):
        if c and Path(c).exists():
            return str(c)
    return None


_worker_cfg = None
_worker_pal = None


def _init_worker(cfg, pal, ss):
    global _worker_cfg, _worker_pal, S, W2, H2
    _worker_cfg, _worker_pal = cfg, pal


def _worker_render(idx):
    return np.asarray(render_frame(idx, _worker_cfg, _worker_pal), np.uint8).tobytes()


def main():
    ap = argparse.ArgumentParser(description="Render the animated 10s event flyer.")
    ap.add_argument("--content", default=str(ROOT / "content.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "event-flyer.mp4"))
    ap.add_argument("--ss", type=int, default=2, help="supersample factor (1-3)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    ap.add_argument("--preview", nargs="*", type=float,
                    help="render only these timestamps (s) as PNGs into --png-dir")
    ap.add_argument("--png-dir", default=str(OUT_DIR / "stills"))
    args = ap.parse_args()

    global S, W2, H2
    S = max(1, min(3, args.ss))
    W2, H2 = W * S, H * S

    cfg = json.loads(Path(args.content).read_text())
    pal = {k: tuple(v) for k, v in cfg["palette"].items()}
    for k in ("bg_top", "bg_mid", "bg_bot", "glow", "accent", "accent2", "ink", "paper"):
        pal.setdefault(k, (255, 0, 255))

    if args.preview is not None:
        times = args.preview or [0.6, 1.6, 2.6, 3.5, 4.4, 5.2, 6.2, 7.4, 8.6, 9.7]
        outdir = Path(args.png_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for tt in times:
            img = render_frame(int(round(tt * FPS)), cfg, pal)
            p = outdir / f"frame-{tt:04.1f}s.jpg"
            img.save(p, quality=88, optimize=True)
            print(f"  {p.name}  {time.time() - t0:5.2f}s")
        print(f"preview written to {outdir}")
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        sys.exit("ffmpeg not found - run ./setup.sh or set $FFMPEG")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
           "-movflags", "+faststart", str(out)]

    print(f"rendering {FRAMES} frames @ {W}x{H} ({DURATION:g}s, {FPS}fps), ss={S}, jobs={args.jobs}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        if args.jobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(cfg, pal, S)) as pool:
                for i, buf in enumerate(pool.imap(_worker_render, range(FRAMES), chunksize=2)):
                    proc.stdin.write(buf)
                    if i % 30 == 0 or i == FRAMES - 1:
                        el = time.time() - t0
                        print(f"  frame {i + 1}/{FRAMES}  {el:5.1f}s  "
                              f"eta {el / (i + 1) * (FRAMES - i - 1):5.1f}s", flush=True)
        else:
            for i in range(FRAMES):
                proc.stdin.write(_worker_render(i))
                if i % 30 == 0:
                    print(f"  frame {i + 1}/{FRAMES}  {time.time() - t0:5.1f}s", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"ffmpeg exited with {rc}")

    poster = out.with_name("event-flyer-poster.png")
    render_frame(int(9.0 * FPS), cfg, pal).save(poster)
    size = out.stat().st_size / 1e6
    print(f"done in {time.time() - t0:.1f}s -> {out}  ({size:.1f} MB)\nposter -> {poster}")


if __name__ == "__main__":
    main()
