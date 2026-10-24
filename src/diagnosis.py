"""
Suspected-defect diagnosis (heuristic)
======================================
The classifiers are binary (good / defective), so they cannot name a defect type on
their own. This module turns the Grad-CAM heat-map into something an inspector can
act on: a bounding box around the hottest region plus a *suspected* category based on
where that region sits relative to the transistor body and leads.

It is an explanation aid, NOT a trained multi-class classifier, and every result is
flagged ``"heuristic": True`` so the UI can say so.
"""

from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from src.orientation import detect_lead_direction

_HOT_LEVEL = 0.6   # heat-map level that counts as "the model is looking here"

TYPE_LABELS = {
    "lead_defect": "Lead defect (bent or cut lead)",
    "damaged_case": "Case / body damage",
    "misplaced": "Placement issue (misplaced or missing part)",
    "unlocalised": "Defect not localised",
}


def diagnose(model_view: Image.Image, heatmap: Optional[np.ndarray]) -> Optional[Dict]:
    """
    Args:
        model_view: the image exactly as the model saw it (upright, 224 crop).
        heatmap:    Grad-CAM heat-map [H, W] in [0, 1] for the same view.
    Returns:
        {"type", "label", "box": [x0, y0, x1, y1] normalised 0..1 or None, "heuristic": True,
         "note"} or None when no heat-map is available.
    """
    if heatmap is None:
        return None
    h, w = heatmap.shape[:2]
    hot = (heatmap >= _HOT_LEVEL).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(hot)
    if n < 2:
        return {"type": "unlocalised", "label": TYPE_LABELS["unlocalised"], "box": None,
                "heuristic": True, "note": "No concentrated region of attention."}

    # Hottest connected region = the one with the most accumulated heat.
    best = max(range(1, n), key=lambda i: float(heatmap[labels == i].sum()))
    x, y, bw, bh, _ = (int(v) for v in stats[best])
    box = [round(x / w, 4), round(y / h, 4), round((x + bw) / w, 4), round((y + bh) / h, 4)]
    cx, cy = cents[best][0] / w, cents[best][1] / h

    body = detect_lead_direction(model_view).get("body_box")
    if body is None:
        kind = "misplaced"
    else:
        bx0, by0, bx1, by1 = body
        body_w = bx1 - bx0
        below = cy > by1 - 0.02
        within_x = (bx0 - 0.5 * body_w) <= cx <= (bx1 + 0.5 * body_w)
        if bx0 <= cx <= bx1 and by0 <= cy <= by1 - 0.02:
            kind = "damaged_case"
        elif below and within_x:
            kind = "lead_defect"
        else:
            kind = "misplaced"
    return {"type": kind, "label": TYPE_LABELS[kind], "box": box, "heuristic": True,
            "note": "Suspected from where the model's attention falls; not a trained defect-type classifier."}
