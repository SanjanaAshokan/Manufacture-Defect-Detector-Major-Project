"""
Regression tests for the geometry half of the "is this a transistor?" gate.

Run from the project root:   python tests/test_gate_geometry.py      (or: pytest tests)
Needs only numpy / OpenCV / Pillow (no torch, no model weights, no dataset).

The scenes are synthetic (see gate_scenes.py): they reproduce the situations that broke on
real photos - stock-photo footer bars, soft grey backdrops, sideways parts - and make sure
the clean cases (white / PCB backgrounds) are completely unaffected.
"""
import ast
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_scenes as gs                      # noqa: E402
from src.orientation import auto_orient       # noqa: E402


def geometry_ok(info):
    return bool(info and info.get("body_box") is not None
                and (info.get("lead_box") is not None or info.get("lead_evidence")))


def accepted(img):
    return geometry_ok(auto_orient(Image.fromarray(img))[1])


def test_footer_bar_does_not_hide_the_transistor():
    rng = np.random.default_rng(1)
    for package in ("TO92", "TO220", "TO3P"):
        for angle in (0, 90, 180, 270, 35):
            img, _ = gs.positive(rng, background="white", package=package, angle=angle, footer=True, watermark=True)
            assert accepted(img), f"{package} at {angle} deg with a stock footer was not recognised"


def test_soft_grey_backdrop_is_recognised():
    rng = np.random.default_rng(2)
    hits = sum(accepted(gs.positive(rng, background="grey")[0]) for _ in range(40))
    assert hits >= 36, f"only {hits}/40 transistors on a grey gradient were recognised"


def test_clean_backgrounds_are_untouched():
    """Plain white / PCB photos must not need (or report) any relaxation."""
    rng = np.random.default_rng(3)
    for bg in ("white", "pcb"):
        for _ in range(15):
            img, _ = gs.positive(rng, background=bg, footer=False, watermark=False)
            info = auto_orient(Image.fromarray(img))[1]
            assert geometry_ok(info), f"{bg}: transistor not recognised"
            assert info["assisted"] == [], f"{bg}: clean image was flagged as relaxed: {info['assisted']}"


def test_blank_and_text_images_are_not_transistors():
    rng = np.random.default_rng(4)
    for kind in ("blank", "text"):
        for _ in range(8):
            img, _ = gs.negative(rng, kind=kind)
            assert not accepted(img), f"'{kind}' image was accepted as a transistor"


def test_overall_recall_on_mixed_scenes():
    rng = np.random.default_rng(7)
    n = 120
    hits = sum(accepted(gs.positive(rng)[0]) for _ in range(n))
    assert hits / n >= 0.80, f"recall {hits / n:.0%} on mixed synthetic scenes"


# --- resolve_geometry (web/app.py) - extracted so the test does not need torch -------------
def _load_resolve_geometry(floor=0.20, live_found=None, crop_similarity=0.0):
    tree = ast.parse((ROOT / "web" / "app.py").read_text(encoding="utf-8"))
    wanted = {"has_transistor_geometry", "assisted_min_similarity", "resolve_geometry"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    ns = {
        "os": __import__("os"), "Image": Image, "CONFIG": {"web": {"ood_assisted_min_similarity": floor}},
        "find_transistor_geometry": lambda image: live_found or {"ok": False},
        "centre_crop": lambda image, f: (image, 0, 0),
        "_ood_similarity": lambda image: crop_similarity,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "app_extract", "exec"), ns)
    return ns["resolve_geometry"]


GEOM = {"body_box": [0.4, 0.4, 0.6, 0.6], "lead_box": [0.4, 0.6, 0.6, 0.9]}
IMG = Image.new("RGB", (100, 100))


def test_standard_geometry_passes_regardless_of_similarity():
    resolve = _load_resolve_geometry()
    assert resolve(IMG, dict(GEOM, assisted=[]), 0.05, False)      # exactly as before


def test_relaxed_geometry_needs_a_similarity_floor():
    resolve = _load_resolve_geometry(floor=0.20)
    relaxed = dict(GEOM, assisted=["border bars trimmed"])
    assert resolve(IMG, relaxed, 0.32, False)
    assert not resolve(IMG, relaxed, 0.10, False)


def test_live_centre_crop_rescue_only_in_live_mode():
    found = {"ok": True, "method": "centre 60% crop", "body_box": [0.4, 0.4, 0.6, 0.6], "lead_box": None}
    live = _load_resolve_geometry(live_found=found, crop_similarity=0.30)
    assert live(IMG, {}, 0.12, True)          # crop similarity clears the floor
    assert not live(IMG, {}, 0.12, False)     # uploads never use the rescue
    weak = _load_resolve_geometry(live_found=found, crop_similarity=0.05)
    assert not weak(IMG, {}, 0.05, True)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failures else 0)
