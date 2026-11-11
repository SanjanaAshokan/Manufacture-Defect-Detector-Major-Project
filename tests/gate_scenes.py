"""
Synthetic scenes for regression-testing the "is this a transistor?" geometry checks.

Real transistor photos cannot be shipped with the repo, so this draws simple top-down
transistors (TO-92 / TO-220 / TO-3P style: body, optional metal tab, thin leads) on many
backgrounds - plain white, grey gradient, wood, cluttered desk, green PCB - at any rotation,
optionally with a stock-photo style footer bar and tiled watermark. It also draws things that
are NOT transistors (blank sheet, bare rectangle, cable, hand, fork, face, clutter).

These are approximations: they prove the geometry logic copes with the situations that
broke on real photos; they do not replace trying real photos from your own parts bin.
"""
import math

import cv2
import numpy as np

W, H = 480, 360
BODY_COLOURS = [(22, 22, 24), (96, 74, 64), (64, 64, 70)]
METAL = (185, 185, 190)


def _background(kind, rng):
    if kind == "white":
        return np.full((H, W, 3), 255, np.uint8)
    if kind == "grey":
        g = np.linspace(215, 245, H, dtype=np.float32)[:, None, None] * np.ones((1, W, 3), np.float32)
        return g.astype(np.uint8)
    if kind == "pcb":
        img = np.zeros((H, W, 3), np.float32)
        img[:] = (40, 110, 60)
        img += rng.normal(0, 6, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)
    if kind == "wood":
        x = np.arange(W)[None, :, None].astype(np.float32)
        y = np.arange(H)[:, None, None].astype(np.float32)
        grain = 18 * np.sin(x / 7.0 + 4 * np.sin(y / 40.0)) + rng.normal(0, 7, (H, W, 1))
        base = np.array([150, 105, 65], np.float32)[None, None, :]
        return np.clip(base + grain, 0, 255).astype(np.uint8)
    if kind == "clutter":
        img = np.full((H, W, 3), 170, np.uint8)
        for _ in range(26):
            c = tuple(int(v) for v in rng.integers(30, 230, 3))
            cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
            if rng.random() < 0.5:
                cv2.circle(img, (cx, cy), int(rng.integers(15, 60)), c, -1)
            else:
                cv2.rectangle(img, (cx, cy), (cx + int(rng.integers(20, 120)), cy + int(rng.integers(20, 90))), c, -1)
        return cv2.GaussianBlur(img, (0, 0), 2.0)
    raise ValueError(kind)


def _transistor_layer(kind, body_rgb, scale):
    """Draw an upright part (leads pointing DOWN) on its own transparent layer."""
    layer = np.zeros((H, W, 4), np.uint8)
    cx, top = W // 2, int(95 * (1.0 if scale >= 1 else 1.1))

    def rect(x0, y0, x1, y1, rgb):
        cv2.rectangle(layer, (int(x0), int(y0)), (int(x1), int(y1)), (*rgb, 255), -1)

    bw = {"TO92": 56, "TO220": 74, "TO3P": 92}[kind] * scale
    bh = {"TO92": 56, "TO220": 80, "TO3P": 96}[kind] * scale
    if kind == "TO92":
        cv2.ellipse(layer, (cx, int(top + bh * 0.5)), (int(bw / 2), int(bh / 2)), 0, 180, 360, (*body_rgb, 255), -1)
        rect(cx - bw / 2, top + bh * 0.5, cx + bw / 2, top + bh, body_rgb)
    else:
        tab_h = int(0.38 * bh)
        rect(cx - bw / 2, top - tab_h, cx + bw / 2, top, METAL)
        cv2.circle(layer, (cx, int(top - tab_h * 0.5)), max(3, int(tab_h * 0.22)), (0, 0, 0, 0), -1)
        rect(cx - bw / 2, top, cx + bw / 2, top + bh, body_rgb)
    lead_w = max(3, int(5 * scale))
    lead_len = int(95 * scale)
    gap = bw * 0.33
    for i in (-1, 0, 1):
        x = cx + i * gap
        rect(x - lead_w / 2, top + bh, x + lead_w / 2, top + bh + lead_len, METAL)
    return layer


def _rotate_layer(layer, angle):
    m = cv2.getRotationMatrix2D((W / 2, H / 2 + 20), angle, 1.0)
    return cv2.warpAffine(layer, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _composite(bg, layer, shadow=True):
    out = bg.astype(np.float32)
    a = layer[..., 3:4].astype(np.float32) / 255.0
    if shadow:
        sh = cv2.GaussianBlur(a[..., 0], (0, 0), 6)
        sh = np.roll(np.roll(sh, 8, axis=0), 8, axis=1)[..., None]
        out = out * (1 - 0.25 * sh)
    out = out * (1 - a) + layer[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def add_stock_footer(img):
    img = img.copy()
    bar = int(0.085 * H)
    img[H - bar:] = (38, 38, 42)
    cv2.putText(img, "stockphoto", (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, "ID 1234567890", (W - 150, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def add_watermark(img):
    over = img.copy()
    for y in range(20, H, 70):
        for x in range(10, W, 90):
            cv2.putText(over, "stock", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    return cv2.addWeighted(over, 0.25, img, 0.75, 0)


def positive(rng, background=None, package=None, angle=None, footer=None, watermark=None):
    background = background or str(rng.choice(["white", "grey", "pcb", "wood", "clutter"]))
    package = package or str(rng.choice(["TO92", "TO220", "TO3P"]))
    angle = float(rng.choice([0, 90, 180, 270, 35, -50, 150])) if angle is None else angle
    scale = float(rng.uniform(0.85, 1.25))
    body = BODY_COLOURS[int(rng.integers(0, len(BODY_COLOURS)))]
    img = _composite(_background(background, rng), _rotate_layer(_transistor_layer(package, body, scale), angle))
    if watermark if watermark is not None else rng.random() < 0.3:
        img = add_watermark(img)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    if footer if footer is not None else (background in ("white", "grey") and rng.random() < 0.4):
        img = add_stock_footer(img)
    return img, {"background": background, "package": package, "angle": angle}


def negative(rng, kind=None):
    kind = kind or str(rng.choice(["blank", "rect", "disc", "cable", "hand", "fork", "face", "clutter", "text"]))
    bg = _background(str(rng.choice(["white", "grey", "wood", "clutter"])), rng)
    img = bg.copy()
    cx, cy = W // 2, H // 2
    if kind == "blank":
        img = np.full((H, W, 3), 128, np.uint8)
    elif kind == "rect":
        cv2.rectangle(img, (cx - 70, cy - 55), (cx + 70, cy + 55), (40, 40, 45), -1)
    elif kind == "disc":
        cv2.circle(img, (cx, cy), 80, (60, 60, 200), -1)
    elif kind == "cable":
        pts = np.array([[20, cy + 40], [150, cy - 60], [300, cy + 60], [460, cy - 20]], np.int32)
        cv2.polylines(img, [pts], False, (30, 30, 30), 7, cv2.LINE_AA)
    elif kind == "hand":
        skin = (190, 150, 125)
        cv2.ellipse(img, (cx, cy + 40), (70, 80), 0, 0, 360, skin, -1)
        for i, ang in enumerate([-60, -30, 0, 30, 60]):
            ex = cx + int(math.sin(math.radians(ang)) * 150)
            ey = cy + 40 - int(math.cos(math.radians(ang)) * 150)
            cv2.line(img, (cx + (i - 2) * 22, cy + 10), (ex, ey), skin, 22, cv2.LINE_AA)
    elif kind == "fork":
        cv2.rectangle(img, (cx - 8, cy), (cx + 8, cy + 140), (170, 170, 175), -1)
        cv2.ellipse(img, (cx, cy), (45, 25), 0, 0, 180, (170, 170, 175), -1)
        for i in (-30, -10, 10, 30):
            cv2.rectangle(img, (cx + i - 4, cy - 85), (cx + i + 4, cy - 5), (170, 170, 175), -1)
    elif kind == "face":
        cv2.ellipse(img, (cx, cy), (85, 110), 0, 0, 360, (200, 160, 135), -1)
        cv2.circle(img, (cx - 30, cy - 20), 9, (30, 30, 30), -1)
        cv2.circle(img, (cx + 30, cy - 20), 9, (30, 30, 30), -1)
        cv2.ellipse(img, (cx, cy + 45), (28, 12), 0, 0, 180, (120, 50, 50), 3)
    elif kind == "text":
        for r in range(8):
            cv2.putText(img, "lorem ipsum dolor sit amet", (15, 40 + r * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    elif kind == "clutter":
        pass
    return cv2.GaussianBlur(img, (0, 0), 0.8), {"kind": kind}
