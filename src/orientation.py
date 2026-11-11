"""
Orientation Normalisation — make every model rotation-tolerant
===============================================================
The classifiers (EfficientNet-B0, ConvNeXt-Tiny, ResNet-50) were trained on MVTec AD
Transistor images, where every transistor is photographed *upright*: the black
TO-92 body at the top and the three metal leads pointing DOWN. A transistor lying
sideways (or upside-down) is far outside that distribution, so the networks score it
as "defective" almost regardless of its real condition (it looks like the dataset's
"misplaced" class).

Instead of touching the trained weights, this module canonicalises the input:

    1. find the transistor body,
    2. find the leads that stick out of it,
    3. work out which way the leads point,
    4. rotate the image so the leads point down (a lossless multiple of 90 degrees
       for sideways / upside-down parts, an exact angle for parts photographed at a
       clearly diagonal tilt).

Two detectors cooperate: a fast one tuned to the dark-body-on-copper look of the
training photos, and a background-agnostic one (plain white / grey / bench photos).

Upright images are returned untouched (same object, no resampling), so existing
behaviour and accuracy for upright transistors are unchanged. If the geometry is
ambiguous (no body found, no leads visible, conflicting evidence) the image is also
left untouched rather than guessing.

Only numpy / OpenCV / Pillow are used (already in requirements.txt).
"""

from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

# Counter-clockwise turn (degrees, as PIL's ROTATE_90/180/270 transposes) that brings
# leads pointing in <direction> round to "down":
#   leads on the LEFT  -> 90 deg counter-clockwise
#   leads on the RIGHT -> 270 deg counter-clockwise (= 90 deg clockwise)
#   leads pointing UP  -> 180 deg
_ROTATION_FOR_DIRECTION = {"down": 0, "left": 90, "up": 180, "right": 270}
_PIL_TRANSPOSE = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}

_ANALYSIS_SIZE = 256          # long side used for the (cheap) geometric analysis
_MIN_BODY_FRACTION = 0.015    # body must cover at least 1.5% of the frame
_MAX_BODY_FRACTION = 0.45     # ... and at most 45% (otherwise it's not a single part)
_MAX_BODY_SPAN = 0.85         # body may not stretch across >85% of the frame in either axis
_MIN_BODY_EXTENT = 0.55       # body must fill >=55% of its bounding box (compact, blocky shape)
_MIN_LEAD_FRACTION = 0.025    # lead pixels needed, as a fraction of the body area
_MIN_DOMINANCE = 2.0          # winning side must beat the runner-up by this factor
_MIN_CONFIDENCE = 0.60        # confidence needed before we dare to rotate
_MIN_BOARD_SATURATION = 30    # plain (grey / white) backgrounds can't be told apart from silver leads
_TILT_TOLERANCE = 20.0        # degrees: models were trained with +/-15 deg jitter, so smaller tilts are left alone
_GENERIC_MIN_RESULTANT = 0.75 # how one-sided the lead pixels must be around the body (0..1)
_GENERIC_MIN_LEAD_RATIO = 0.04
_GENERIC_MAX_BG_SPREAD = 7.0   # only trust the contrast detector on fairly plain backgrounds

_COMPASS = ["down", "down-left", "left", "up-left", "up", "up-right", "right", "down-right"]


def _compass(alpha_deg: float) -> str:
    """alpha: 0 = leads point down, -90 = left, +90 = right, +/-180 = up."""
    idx = int(round((-alpha_deg % 360.0) / 45.0)) % 8
    return _COMPASS[idx]


def _none(reason: str, **extra) -> Dict:
    return {"direction": None, "confidence": 0.0, "reason": reason, **extra}


def _analysis_view(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = _ANALYSIS_SIZE / float(max(h, w))
    if scale >= 1.0:
        return rgb
    return cv2.resize(
        rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _detect_dark_body(small: np.ndarray) -> Dict:
    """Detector 1: dark plastic body + silver leads on a coloured (copper) board."""
    height, width = small.shape[:2]

    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    # ---- 1) dark plastic body --------------------------------------------------
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_thr = min(float(otsu), 0.5 * float(np.median(gray)))
    dark = (gray < dark_thr).astype(np.uint8)
    k = max(3, int(0.06 * max(height, width)) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n_labels < 2:
        return {"direction": None, "confidence": 0.0, "reason": "no dark body found"}

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, area = (int(v) for v in stats[idx])
    body_fraction = area / float(height * width)
    if not (_MIN_BODY_FRACTION <= body_fraction <= _MAX_BODY_FRACTION):
        return {"direction": None, "confidence": 0.0, "reason": "body size implausible"}
    if bw > _MAX_BODY_SPAN * width or bh > _MAX_BODY_SPAN * height or area < _MIN_BODY_EXTENT * bw * bh:
        return {"direction": None, "confidence": 0.0, "reason": "dark region is not a compact part"}
    body = (labels == idx).astype(np.uint8)
    body_box = [round(bx / width, 4), round(by / height, 4),
                round((bx + bw) / width, 4), round((by + bh) / height, 4)]

    # ---- 2) bright, low-saturation metal leads (copper board is saturated) -----
    # Thresholds are relative to the scene so they survive dim / bright lighting.
    # Reference statistics come from the board only (not the black body), so a close-up
    # that is mostly body does not skew them.
    saturation, value = hsv[..., 1], hsv[..., 2]
    near_body = cv2.dilate(body, np.ones((9, 9), np.uint8))
    board = near_body == 0
    if int(board.sum()) < 0.05 * height * width:
        return {"direction": None, "confidence": 0.0, "reason": "not enough background to analyse"}
    if float(np.median(saturation[board])) < _MIN_BOARD_SATURATION:
        # Grey / white background: silver leads look just like it, so this test is blind.
        return _none("plain background - lead test not applicable", body_box=body_box)
    sat_limit = 0.6 * float(np.median(saturation[board]))
    val_limit = 0.7 * float(np.median(value[board]))
    metal = ((saturation < sat_limit) & (value > val_limit)).astype(np.uint8)
    metal &= 1 - near_body
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # ---- 3) count metal pixels in a band hugging each side of the body ---------
    band = max(8, int(0.8 * min(bw, bh)))
    side_boxes = {
        "down": (bx, by + bh, bx + bw, by + bh + band),
        "up": (bx, by - band, bx + bw, by),
        "right": (bx + bw, by, bx + bw + band, by + bh),
        "left": (bx - band, by, bx, by + bh),
    }
    counts = {}
    for side, (x0, y0, x1, y1) in side_boxes.items():
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        counts[side] = int(metal[y0:y1, x0:x1].sum()) if x1 > x0 and y1 > y0 else 0

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    (best_side, best), (_, runner_up) = ranked[0], ranked[1]
    if best < _MIN_LEAD_FRACTION * area:
        return {"direction": None, "confidence": 0.0, "reason": "no leads visible",
                "counts": counts, "body_box": body_box, "lead_evidence": False}

    dominance = best / float(max(runner_up, 1))
    if dominance < _MIN_DOMINANCE:
        # Keep positive body + metal-lead evidence even when diagonal leads do not
        # fit one of the four cardinal direction bands. OOD validation uses this as
        # object-domain evidence; orientation still correctly remains unchanged.
        return {"direction": None, "confidence": 0.0, "reason": "leads direction ambiguous",
                "counts": counts, "body_box": body_box, "lead_evidence": True}

    # Confidence blends how dominant the winning side is with how much lead there is.
    confidence = 0.6 * min(dominance / 6.0, 1.0) + 0.4 * min(best / (0.10 * area), 1.0)
    x0, y0, x1, y1 = side_boxes[best_side]
    lead_box = [round(max(0, x0) / width, 4), round(max(0, y0) / height, 4),
                round(min(width, x1) / width, 4), round(min(height, y1) / height, 4)]
    return {"direction": best_side, "confidence": round(float(confidence), 3),
            "reason": f"leads point {best_side}", "counts": counts,
            "body_box": body_box, "lead_box": lead_box, "lead_evidence": True}


_VOTE_FAINT_MIN = 10.0        # LAB distance: pale / silver leads on white are far fainter than the body
_VOTE_FAR_FRACTION = 0.4      # a lead pixel must protrude >= 40% of the body half-thickness from the body
_VOTE_MIN_PIXELS = 8
_VOTE_MIN_CONE = 0.75         # share of protruding mass within +/-60 deg of the winning direction
_VOTE_MAX_OPPOSITE = 0.25     # ... and at most this share within +/-60 deg of the opposite direction


def _protruding_lead_vote(dist: np.ndarray, spread: float, body: np.ndarray,
                          thickest: float, centre: np.ndarray) -> Dict:
    """
    Direction vote that survives metal tabs, bent leads, shadows and pale leads.

    Only pixels that genuinely PROTRUDE from the body take part (shadows, corner residue
    and tab outlines hug the body and are ignored), found with a much lower contrast
    threshold than the body so that pale / silver leads on a white backdrop are seen.
    Each pixel is weighted by its distance from the body, so long leads outweigh stubs.
    The verdict is trusted only when the protruding mass is clearly one-sided (a cone of
    +/-60 deg around the winning direction holds most of it, the opposite cone almost
    none); otherwise the caller keeps the image exactly as uploaded.
    """
    faint = cv2.morphologyEx((dist > max(3.0 * spread + 6.0, _VOTE_FAINT_MIN)).astype(np.uint8),
                             cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, comp, _, _ = cv2.connectedComponentsWithStats(faint)
    ids = np.unique(comp[body > 0])
    attached = np.isin(comp, ids[ids > 0]).astype(np.uint8)
    away = cv2.distanceTransform((1 - body).astype(np.uint8), cv2.DIST_L2, 5)
    far = attached & (away >= max(4.0, _VOTE_FAR_FRACTION * thickest)).astype(np.uint8)
    fy, fx = np.nonzero(far)
    if len(fx) < _VOTE_MIN_PIXELS:
        return {"ok": False, "mask": far}
    vec = np.stack([fx - centre[0], fy - centre[1]], axis=1)
    unit = vec / (np.linalg.norm(vec, axis=1)[:, None] + 1e-9)
    weight = away[fy, fx]
    mean = (unit * weight[:, None]).sum(0) / weight.sum()
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return {"ok": False, "mask": far}
    cos = unit @ (mean / norm)
    limit = np.cos(np.radians(60.0))
    cone = float(weight[cos >= limit].sum() / weight.sum())
    opposite = float(weight[cos <= -limit].sum() / weight.sum())
    ok = cone >= _VOTE_MIN_CONE and opposite <= _VOTE_MAX_OPPOSITE
    # pixels on the winning side only: a tab / outline on the opposite side stays out of the
    # lead box that is drawn for the user
    side = np.zeros_like(far)
    side[fy[cos >= limit], fx[cos >= limit]] = 1
    return {"ok": ok, "mask": far, "side": side, "mean": mean, "resultant": norm,
            "cone": cone, "opposite": opposite}


def _fit_smooth_background(lab: np.ndarray, b: int):
    """
    Fit a quadratic surface (per LAB channel) to the picture's border strips. Returns
    (background_field, spread) where spread is the median residual of the border pixels, or
    (None, inf) when it cannot be fitted.
    """
    height, width = lab.shape[:2]
    mask = np.zeros((height, width), bool)
    mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = True, True, True, True
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None, float("inf")

    def design(x, y):
        x = x / float(max(width - 1, 1)) - 0.5
        y = y / float(max(height - 1, 1)) - 0.5
        return np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=1)

    a = design(xs.astype(np.float32), ys.astype(np.float32))
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    full = design(grid_x.ravel().astype(np.float32), grid_y.ravel().astype(np.float32))
    field = np.empty((height, width, 3), np.float32)
    resid = []
    for c in range(3):
        coef, *_ = np.linalg.lstsq(a, lab[ys, xs, c], rcond=None)
        field[..., c] = (full @ coef).reshape(height, width)
        resid.append(np.abs(a @ coef - lab[ys, xs, c]))
    spread = float(np.median(np.sqrt(sum(r * r for r in resid))))
    return field, spread


def _detect_generic(small: np.ndarray) -> Dict:
    """
    Detector 2: background-agnostic. Everything that differs from the photo's border
    colour is "object"; the thick blob is the body and the thin parts attached to it
    are the leads. The lead direction is the mean direction of the lead pixels as seen
    from the body centre, so it also works for diagonal tilts.
    """
    height, width = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(2, int(0.04 * min(height, width)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    spread = float(np.median(np.linalg.norm(border - bg, axis=1)))
    bg_field = None
    if spread > _GENERIC_MAX_BG_SPREAD:
        # Product photos often sit on a soft gradient / vignette, which a single border colour
        # reads as "busy". Model the backdrop as a smooth surface fitted to the border pixels;
        # genuinely busy scenes (wood, clutter) still do not fit and are given up on as before.
        bg_field, spread = _fit_smooth_background(lab, b)
        if bg_field is None or spread > _GENERIC_MAX_BG_SPREAD:
            return _none("background too busy for the contrast detector")
    dist = np.linalg.norm(lab - (bg if bg_field is None else bg_field), axis=2)
    dmax = max(float(dist.max()), 1.0)
    d8 = np.clip(dist * 255.0 / dmax, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_hi = max(otsu * dmax / 255.0, 3 * spread + 8, 15)
    thr_lo = min(max(3 * spread + 8, 15, 0.15 * dmax), thr_hi)

    # body = strong-contrast blob with thin parts opened away
    strong = cv2.morphologyEx((dist > thr_hi).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong)
    if n < 2:
        return _none("no object found")
    blob = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    thickest = float(dt.max())
    if thickest < 4:
        return _none("object too thin to be a package")
    core = (dt >= 0.55 * thickest).astype(np.uint8)
    r = max(2, int(round(0.6 * thickest)))
    body = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) & blob
    body_fraction = body.sum() / float(height * width)
    if not (0.012 <= body_fraction <= 0.6):
        return _none("body size implausible")
    x, y, bw, bh = cv2.boundingRect(body)
    if (x <= 1) + (y <= 1) + (x + bw >= width - 1) + (y + bh >= height - 1) >= 3:
        return _none("body fills the frame")
    body_box = [round(x / width, 4), round(y / height, 4),
                round((x + bw) / width, 4), round((y + bh) / height, 4)]

    # leads = weaker-contrast pixels connected to the body but outside it
    weak = cv2.morphologyEx((dist > thr_lo).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels2, _, _ = cv2.connectedComponentsWithStats(weak)
    ids = np.unique(labels2[body > 0])
    attached = np.isin(labels2, ids[ids > 0]).astype(np.uint8)
    leads = attached & (1 - cv2.dilate(body, np.ones((5, 5), np.uint8)))
    leads = cv2.morphologyEx(leads, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ly, lx = np.nonzero(leads)
    by, bx = np.nonzero(body)
    ratio = len(lx) / float(max(len(bx), 1))
    centre = np.array([bx.mean(), by.mean()])
    plain_ok = ratio >= _GENERIC_MIN_LEAD_RATIO
    lead_mask = leads
    vote = None
    if plain_ok:
        vec = np.stack([lx - centre[0], ly - centre[1]], axis=1)
        norms = np.linalg.norm(vec, axis=1)
        unit = vec[norms > 1e-6] / norms[norms > 1e-6, None]
        mean_dir = unit.mean(axis=0)
        resultant = float(np.linalg.norm(mean_dir))
    if (not plain_ok) or resultant < _GENERIC_MIN_RESULTANT:
        # The plain vote found no leads or is split (tab opposite the leads, bent / fanned-out
        # leads, pale leads ...). Retry with the protruding-pixel vote. Inputs that passed
        # the plain vote never reach this branch, so their behaviour is unchanged.
        vote = _protruding_lead_vote(dist, spread, body, thickest, centre)
        if vote["ok"]:
            mean_dir, resultant = vote["mean"], vote["resultant"]
            lead_mask = vote["side"]
            ratio = max(ratio, np.count_nonzero(vote["mask"]) / float(max(len(bx), 1)))
        elif not plain_ok:
            return _none("no leads visible", body_box=body_box, lead_evidence=False)
        else:
            ambiguous = _none("leads direction ambiguous", body_box=body_box, lead_evidence=True)
            far = vote["mask"] if np.count_nonzero(vote["mask"]) >= _VOTE_MIN_PIXELS else leads
            lbx, lby, lbw, lbh = cv2.boundingRect(far)
            if lbw > 0 and lbh > 0:
                ambiguous["lead_box"] = [round(lbx / width, 4), round(lby / height, 4),
                                         round((lbx + lbw) / width, 4), round((lby + lbh) / height, 4)]
            return ambiguous

    alpha = float(np.degrees(np.arctan2(mean_dir[0], mean_dir[1])))   # 0 = down, -90 = left
    lx0, ly0, lw, lh = cv2.boundingRect(lead_mask)
    lead_box = [round(lx0 / width, 4), round(ly0 / height, 4),
                round((lx0 + lw) / width, 4), round((ly0 + lh) / height, 4)]
    if vote is not None and vote["ok"]:
        # one-sidedness of the protruding mass: cone share 0.5 -> 0, 0.9+ -> 1
        confidence = 0.7 * min(1.0, (vote["cone"] - 0.5) / 0.4) + 0.3 * min(1.0, ratio / 0.3)
    else:
        confidence = 0.7 * min(1.0, (resultant - 0.5) / 0.4) + 0.3 * min(1.0, ratio / 0.3)
    edge = np.concatenate([small[:b].reshape(-1, 3), small[-b:].reshape(-1, 3),
                           small[:, :b].reshape(-1, 3), small[:, -b:].reshape(-1, 3)])
    bg_rgb = tuple(int(v) for v in np.median(edge, axis=0))
    word = _compass(alpha)
    return {"direction": word, "angle_deg": round(alpha, 1), "confidence": round(float(confidence), 3),
            "reason": f"leads point {word}", "body_box": body_box, "lead_box": lead_box,
            "bg_rgb": bg_rgb, "method": "contrast", "lead_evidence": True,
            **({"assisted": ["gradient background"]} if bg_field is not None else {})}



# --------------------------------------------------------------------------------------
# Robustness helpers for real-world photos (stock-photo footers, letterbox bars, captions)
# --------------------------------------------------------------------------------------
_BAND_MIN_STEP = 60.0        # mean-luminance difference (0-255) between an edge band and the picture
_BAND_MIN_JUMP = 0.7         # ... and the row-to-row jump at the band's inner edge, as a share of that difference
_BAND_MIN_FRACTION = 0.01    # a band must be at least 1% of that image dimension ...
_BAND_MAX_FRACTION = 0.30    # ... and at most 30% (anything bigger is the picture itself)


def _edge_band_length(profile: np.ndarray) -> int:
    """
    `profile` is the mean luminance of every row (or column) of the picture. Returns how many
    rows/columns at the START of the profile form a uniform band that is much darker or
    brighter than the picture's middle (0 when there is none).
    """
    n = len(profile)
    if n < 40:
        return 0
    inner = float(np.median(profile[int(0.3 * n):int(0.7 * n) + 1]))
    edge = float(np.median(profile[:2]))
    step = abs(edge - inner)
    if step < _BAND_MIN_STEP:
        return 0
    tol = 0.5 * step
    limit = int(_BAND_MAX_FRACTION * n)
    length = 0
    while length < limit and abs(float(profile[length]) - edge) < tol:
        length += 1
    if not (length >= max(3, int(_BAND_MIN_FRACTION * n)) and length < limit):
        return 0
    # A real bar ends abruptly. A part entering the frame changes the row average gradually,
    # which must not be mistaken for a bar around a white margin.
    jump = abs(float(np.mean(profile[length:length + 3])) - float(np.mean(profile[length - 3:length])))
    return length if jump >= _BAND_MIN_JUMP * step else 0


def content_box(rgb: np.ndarray):
    """
    Find the picture inside uniform bars along its edges (stock-photo footer strips, letterbox
    bars, dark captions). Returns (x0, y0, x1, y1) of the content, or None when the image
    has no such bars - in which case nothing downstream changes at all.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    rows, cols = gray.mean(axis=1), gray.mean(axis=0)
    top = _edge_band_length(rows)
    bottom = _edge_band_length(rows[::-1])
    left = _edge_band_length(cols)
    right = _edge_band_length(cols[::-1])
    if not (top or bottom or left or right):
        return None
    pad = 2   # drop the anti-aliased seam between bar and picture
    return (left + (pad if left else 0), top + (pad if top else 0),
            w - right - (pad if right else 0), h - bottom - (pad if bottom else 0))


def _scale_boxes(found: Dict, x0: int, y0: int, cw: int, ch: int, w: int, h: int) -> Dict:
    """Map normalised boxes measured inside a sub-image back to the whole picture."""
    for key in ("body_box", "lead_box"):
        box = found.get(key)
        if box:
            found[key] = [round((x0 + box[0] * cw) / w, 4), round((y0 + box[1] * ch) / h, 4),
                          round((x0 + box[2] * cw) / w, 4), round((y0 + box[3] * ch) / h, 4)]
    return found


def detect_lead_direction(image: Image.Image) -> Dict:
    """
    Estimate which way the transistor's leads point.

    Returns a dict:
        direction   : compass word ("down", "left", "down-left", ...) or None (undetermined)
        confidence  : 0..1 (None-direction results carry 0.0)
        reason      : short human-readable explanation
        angle_deg   : (contrast detector only) 0 = down, -90 = left, +90 = right, +/-180 = up
        body_box / lead_box : normalised [x0, y0, x1, y1] in the ORIGINAL image
    """
    rgb = np.array(image.convert("RGB"))
    box = content_box(rgb)
    if box is not None:
        x0, y0, x1, y1 = box
        if (x1 - x0) >= 32 and (y1 - y0) >= 32:
            # Bars along the edge (e.g. a dark stock-photo footer) otherwise get mistaken for
            # the transistor body, or make the background look "busy". Analyse the picture
            # inside the bars and map the boxes back; fall back to the whole image only if
            # that finds nothing.
            found = _detect_on_rgb(rgb[y0:y1, x0:x1])
            if found.get("direction") is not None or found.get("body_box") is not None:
                found["content_box"] = [round(x0 / rgb.shape[1], 4), round(y0 / rgb.shape[0], 4),
                                        round(x1 / rgb.shape[1], 4), round(y1 / rgb.shape[0], 4)]
                found["assisted"] = list(found.get("assisted", [])) + ["border bars trimmed"]
                return _scale_boxes(found, x0, y0, x1 - x0, y1 - y0, rgb.shape[1], rgb.shape[0])
    return _detect_on_rgb(rgb)


def _detect_on_rgb(rgb: np.ndarray) -> Dict:
    small = _analysis_view(rgb)
    height, width = small.shape[:2]
    if min(height, width) < 32:
        return _none("image too small")
    first = _detect_dark_body(small)
    if first["direction"] is not None:
        first["method"] = "dark-body"
        return first
    second = _detect_generic(small)
    if second["direction"] is not None:
        return second
    # undetermined by both: keep the first detector's body box if it has one
    if first.get("body_box") is None and second.get("body_box") is not None:
        first["body_box"] = second["body_box"]
    # Lead evidence found by the contrast detector must survive even when the dark-body
    # detector contributed the body box itself (e.g. TO-220 / TO-3P parts on a plain white
    # backdrop: the dark-body detector finds the body but its lead test is blind there,
    # while the contrast detector sees body + leads yet cannot name one direction).
    # Direction stays None, so no rotation is ever applied because of this.
    if second.get("lead_evidence") and first.get("body_box") is not None:
        first["lead_evidence"] = True
        if second.get("assisted"):
            first["assisted"] = second["assisted"]
        if first.get("lead_box") is None and second.get("lead_box") is not None:
            first["lead_box"] = second["lead_box"]
    return first

def auto_orient(image: Image.Image, min_confidence: float = _MIN_CONFIDENCE) -> Tuple[Image.Image, Dict]:
    """
    Rotate `image` so the transistor leads point down (the orientation every model was
    trained on).

    Returns (image, info). `info` always has:
        applied        : bool  - True when the image was rotated
        rotation_deg   : float - degrees the picture was turned counter-clockwise (0 if not)
        leads_pointing : str|None - detected lead direction in the ORIGINAL image
        confidence     : float
        reason         : str
    Upright / undetermined images are returned as the very same object, unchanged.
    """
    try:
        found = detect_lead_direction(image)
    except Exception as exc:  # never let a helper break inference
        return image, {"applied": False, "rotation_deg": 0, "leads_pointing": None,
                       "confidence": 0.0, "reason": f"orientation check skipped ({exc})",
                       "body_box": None, "lead_box": None}

    direction = found["direction"]
    info = {
        "applied": False,
        "rotation_deg": 0,
        "leads_pointing": direction,
        "confidence": found["confidence"],
        "reason": found["reason"],
        "body_box": found.get("body_box"),   # normalised [x0, y0, x1, y1] in the ORIGINAL image
        "lead_box": found.get("lead_box"),
        "lead_evidence": bool(found.get("lead_evidence", found.get("lead_box") is not None)),
        # which relaxations (trimmed bars, gradient backdrop) were needed to find the geometry;
        # the "is this a transistor?" gate asks for a little extra similarity in that case
        "assisted": list(found.get("assisted", [])),
    }
    if direction is None or found["confidence"] < min_confidence:
        if direction not in (None, "down"):
            info["reason"] = "orientation unclear - left as uploaded"
        return image, info

    # Counter-clockwise turn that brings the leads round to "down".
    if "angle_deg" in found:
        turn = (-found["angle_deg"]) % 360.0
    else:
        turn = float(_ROTATION_FOR_DIRECTION[direction])
    nearest = int(round(turn / 90.0)) % 4
    residual = ((turn - 90.0 * nearest + 180.0) % 360.0) - 180.0

    if abs(residual) <= _TILT_TOLERANCE:
        # sideways / upside-down (or already upright): lossless quarter-turn
        if nearest == 0:
            return image, info
        applied = 90 * nearest
        rotated = image.convert("RGB").transpose(_PIL_TRANSPOSE[applied])
        how = f"rotated {applied} deg (counter-clockwise) to upright"
    else:
        # clearly diagonal: turn by the exact angle, padding with the background colour
        applied = round(turn, 1)
        fill = found.get("bg_rgb") or (0, 0, 0)
        rotated = image.convert("RGB").rotate(applied, resample=Image.BICUBIC, expand=True, fillcolor=fill)
        how = f"tilted; rotated {applied} deg (counter-clockwise) to upright"
    info.update(applied=True, rotation_deg=applied, reason=f"leads pointed {direction}; {how}")
    return rotated, info


def centre_crop(image: Image.Image, fraction: float):
    """Central `fraction` of the picture and its pixel offset (x0, y0)."""
    w, h = image.size
    cw, ch = max(1, int(round(w * fraction))), max(1, int(round(h * fraction)))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return image.crop((x0, y0, x0 + cw, y0 + ch)), x0, y0


def find_transistor_geometry(image: Image.Image, views=(0.6,)) -> Dict:
    """
    Rescue search for the live camera, used when the whole-frame geometry check found nothing.
    The part is meant to be held in the guide box in the middle, so re-run the check on a
    tighter central crop: that discards the busy desk / hand / room around it.

    Returns {"ok": bool, "method": str|None, "body_box": [...]|None, "lead_box": [...]|None}
    with boxes normalised to the WHOLE picture. Nothing here ever rotates the image.
    """
    W, H = image.size
    for fraction in views:
        crop, x0, y0 = centre_crop(image, fraction)
        cw, ch = crop.size
        if min(cw, ch) < 48:
            continue
        found = detect_lead_direction(crop)
        if found.get("body_box") is not None and (found.get("lead_box") is not None or found.get("lead_evidence")):
            found = _scale_boxes(dict(found), x0, y0, cw, ch, W, H)
            return {"ok": True, "method": f"centre {int(fraction * 100)}% crop",
                    "body_box": found["body_box"], "lead_box": found.get("lead_box")}
    return {"ok": False, "method": None, "body_box": None, "lead_box": None}
