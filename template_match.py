"""OpenCV template matching for debris vs no-debris images."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


DATA = Path(r"C:\Qinwu\Qinwu\Xu publication\template_match_cnn\data")

YES_DIR = DATA / "yes"
NO_DIR = DATA / "no"

TEMPLATE_PATH = DATA / "template1.png"

TEMPLATE_NAMES = [
    # "template2.png",
    # "template3.png",
    "template4.png",
    # "template4b.png",
    # "template5.png",
    # "template6a.png",
    # "template6b.png",
    # "template6c.png",
]

THRESHOLDS = np.round(np.arange(0.10, 0.90, 0.05), 2)

SEED = 42
TRAIN_FRAC = 0.70

# ------------------------------------------------------------
# NEW: preprocessing configuration
# ------------------------------------------------------------

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(path)

    return img


# ============================================================
# NEW: find highest-resolution candidate image
# ============================================================

def find_target_resolution(paths: list[Path]) -> tuple[int, int]:
    """
    Find the candidate image with the largest number of pixels.

    Returns:
        (width, height)
    """

    max_pixels = -1
    target_size = None
    target_path = None

    for path in paths:

        img = load_gray(path)

        h, w = img.shape[:2]
        pixels = h * w

        if pixels > max_pixels:
            max_pixels = pixels
            target_size = (w, h)
            target_path = path

    if target_size is None:
        raise RuntimeError("No candidate images found.")

    print(
        f"\nHighest-resolution candidate:\n"
        f"  {target_path}\n"
        f"  resolution = {target_size[0]} x {target_size[1]}"
    )

    return target_size


# ============================================================
# NEW: preprocessing
# ============================================================

def preprocess_image(
    img: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Preprocess image.

    Candidate images:
        resize to target_size + Gaussian blur.

    Templates:
        target_size=None -> Gaussian blur only.
    """

    if target_size is not None:

        h, w = img.shape[:2]
        target_w, target_h = target_size

        if (w, h) != target_size:

            # Enlargement
            if target_w > w or target_h > h:
                interpolation = cv2.INTER_CUBIC

            # Reduction
            else:
                interpolation = cv2.INTER_AREA

            img = cv2.resize(
                img,
                target_size,
                interpolation=interpolation,
            )

    # Gaussian blur
    img = cv2.GaussianBlur(
        img,
        GAUSSIAN_KERNEL,
        GAUSSIAN_SIGMA,
    )

    return img


def match_score(
    image: np.ndarray,
    template: np.ndarray,
) -> float:

    ih, iw = image.shape[:2]
    th, tw = template.shape[:2]

    tmpl = template

    if ih < th or iw < tw:

        scale = min(
            ih / th,
            iw / tw,
        )

        if scale <= 0:
            return 0.0

        tw2 = max(
            1,
            min(iw, int(tw * scale)),
        )

        th2 = max(
            1,
            min(ih, int(th * scale)),
        )

        tmpl = cv2.resize(
            template,
            (tw2, th2),
            interpolation=cv2.INTER_AREA,
        )

    res = cv2.matchTemplate(
        image,
        tmpl,
        cv2.TM_CCOEFF_NORMED,
    )

    return float(res.max())


def stratified_split(
    pos: list,
    neg: list,
    train_frac: float,
    seed: int,
):

    rng = np.random.default_rng(seed)

    # copy so original lists are not modified
    pos = pos.copy()
    neg = neg.copy()

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_tr_p = int(len(pos) * train_frac)
    n_tr_n = int(len(neg) * train_frac)

    train = (
        [(p, 1) for p in pos[:n_tr_p]]
        + [(p, 0) for p in neg[:n_tr_n]]
    )

    test = (
        [(p, 1) for p in pos[n_tr_p:]]
        + [(p, 0) for p in neg[n_tr_n:]]
    )

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


# ============================================================
# MODIFIED:
# Save resized + Gaussian-blurred copies instead of simply
# copying the originals.
# ============================================================

def write_split(
    train,
    test,
    train_name: str,
    test_name: str,
    target_size: tuple[int, int],
) -> None:

    # Re-create output folders
    for parent in (YES_DIR, NO_DIR):

        for name in (train_name, test_name):

            dest = parent / name

            if dest.exists():
                shutil.rmtree(dest)

            dest.mkdir(
                parents=True,
                exist_ok=True,
            )

    # Process and save images
    for split_name, pairs in (
        (train_name, train),
        (test_name, test),
    ):

        for path, lab in pairs:

            dest_dir = (
                YES_DIR
                if lab == 1
                else NO_DIR
            ) / split_name

            # Load original
            img = load_gray(path)

            # NEW:
            # resize to uniform resolution
            # + Gaussian blur
            img = preprocess_image(
                img,
                target_size,
            )

            dest_path = dest_dir / path.name

            ok = cv2.imwrite(
                str(dest_path),
                img,
            )

            if not ok:
                raise IOError(
                    f"Could not save {dest_path}"
                )

    print(
        f"\nwrote processed images:\n"
        f"{YES_DIR / train_name}\n"
        f"{YES_DIR / test_name}\n"
        f"{NO_DIR / train_name}\n"
        f"{NO_DIR / test_name}"
    )


def scores_for(
    pairs,
    template,
) -> tuple[np.ndarray, np.ndarray]:

    y = np.array(
        [lab for _, lab in pairs],
        dtype=int,
    )

    s = np.array(
        [
            match_score(
                load_gray(p),
                template,
            )
            for p, _ in pairs
        ]
    )

    return s, y


def metrics(
    pred: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:

    tp = int(
        np.sum(
            (pred == 1) & (y == 1)
        )
    )

    fp = int(
        np.sum(
            (pred == 1) & (y == 0)
        )
    )

    tn = int(
        np.sum(
            (pred == 0) & (y == 0)
        )
    )

    fn = int(
        np.sum(
            (pred == 0) & (y == 1)
        )
    )

    prec = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    rec = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    f1 = (
        2 * prec * rec / (prec + rec)
        if prec + rec
        else 0.0
    )

    return prec, rec, fpr, f1


def table(
    title: str,
    scores: np.ndarray,
    y: np.ndarray,    
) -> int:

    print(f"\n{title}")

    print(
        f"{'threshold':>10}  "
        f"{'precision':>10}  "
        f"{'recall':>10}  "
        f"{'FPR':>10}  "
        f"{'F1':>10}"
    )

    best_i = 0
    best_f1 = -1.0

    for i, t in enumerate(THRESHOLDS):

        pred = (
            scores >= t
        ).astype(int)

        p, r, fpr, f1 = metrics(
            pred,
            y,
        )

        mark = ""

        if f1 > best_f1:
            best_f1 = f1
            best_i = i
            mark = "  <-- best F1"

        print(
            f"{t:10.2f}  "
            f"{p:10.3f}  "
            f"{r:10.3f}  "
            f"{fpr:10.3f}  "
            f"{f1:10.3f}"
            f"{mark}"
        )

    return best_i


def eval_template(
    name: str,
    train_set,
    test_set,
    frac
):

    path = DATA / name

    if not path.exists():
        print(f"\nskip missing {path}")
        return

    # Load template
    template = load_gray(path)

    # NEW:
    # Gaussian blur template, but DO NOT resize it to
    # full candidate-image resolution.
    template = preprocess_image(
        template,
        target_size=None,
    )

    print(
        f"\n======== "
        f"{name} {template.shape} "
        f"========"
    )
    variants = generate_scale_rotate_templates(
        template,
        scales=(1.0, 0.75, 0.5, 0.25),
        angles=(0, 45, 90),
    )

    for transformed_template, scale, angle in variants:

        print(
            f"\nScale={scale:.2f}, "
            f"Rotation={angle:.0f} deg, "
            f"shape={transformed_template.shape}"
        )
        tr_s, tr_y = scores_for(
            train_set,
            template,
        )

        te_s, te_y = scores_for(
            test_set,
            template,
        )

        best = table(
            f"Train - {name}",
            tr_s,
            tr_y,
        )

        table(
            f"Test - {name}",
            te_s,
            te_y,
        )

        t = float(
            THRESHOLDS[best]
        )

        p, r, fpr, f1 = metrics(
            (te_s >= t).astype(int),
            te_y,
        )

        print(
            f"The train-test split ratio: {frac}"
            f"Chosen from train: "
            f"threshold={t:.2f}  "
            f"test precision={p:.3f}  "
            f"recall={r:.3f}  "
            f"FPR={fpr:.3f}  "
            f"F1={f1:.3f}"
        )

def generate_scale_rotate_templates(
    template: np.ndarray,
    scales=(1.0, 0.5, 0.25),
    angles=(0, 45, 90),
) -> list[tuple[np.ndarray, float, float]]:
    """
    Generate multi-scale and rotated versions of one template.

    Parameters
    ----------
    template:
        Input grayscale template.

    scales:
        Scale factors, e.g. (1.0, 0.5, 0.25).

    angles:
        Rotation angles in degrees, e.g. (0, 45, 90).

    Returns
    -------
    variants:
        List of:
            (transformed_template, scale, angle)

    Example:
        scales = 3
        angles = 3
        -> 9 transformed templates
    """

    variants = []

    h, w = template.shape[:2]

    for scale in scales:

        # --------------------------------------------------
        # 1. SCALE
        # --------------------------------------------------

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if scale == 1.0:
            scaled = template.copy()
        else:
            interpolation = (
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_CUBIC
            )

            scaled = cv2.resize(
                template,
                (new_w, new_h),
                interpolation=interpolation,
            )

        # --------------------------------------------------
        # 2. ROTATE each scaled template
        # --------------------------------------------------

        sh, sw = scaled.shape[:2]

        center = (
            sw / 2.0,
            sh / 2.0,
        )

        for angle in angles:

            if angle == 0:
                rotated = scaled.copy()

            else:

                M = cv2.getRotationMatrix2D(
                    center,
                    angle,
                    1.0,
                )

                # ------------------------------------------
                # Expand canvas so rotated template
                # is NOT cropped.
                # ------------------------------------------

                cos = abs(M[0, 0])
                sin = abs(M[0, 1])

                bound_w = int(
                    np.ceil(
                        sh * sin +
                        sw * cos
                    )
                )

                bound_h = int(
                    np.ceil(
                        sh * cos +
                        sw * sin
                    )
                )

                # Adjust translation for expanded canvas
                M[0, 2] += (
                    bound_w / 2.0 -
                    center[0]
                )

                M[1, 2] += (
                    bound_h / 2.0 -
                    center[1]
                )

                rotated = cv2.warpAffine(
                    scaled,
                    M,
                    (bound_w, bound_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

            variants.append(
                (
                    rotated,
                    scale,
                    angle,
                )
            )

    return variants

def main():

    # --------------------------------------------------------
    # Original candidate images
    # --------------------------------------------------------

    pos = list_images(YES_DIR)
    neg = list_images(NO_DIR)

    if not pos or not neg:
        raise SystemExit(
            f"Need images in {YES_DIR} and {NO_DIR}"
        )

    all_images = pos + neg

    # --------------------------------------------------------
    # NEW:
    # Determine highest candidate-image resolution
    # --------------------------------------------------------

    target_size = find_target_resolution(
        all_images
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    # for TRAIN_FRAC in [0.1*(i+1) for i in range(7)]:
    for TRAIN_FRAC in [0]:        
        train, test = stratified_split(
        pos,
        neg,
        TRAIN_FRAC,
        SEED,
        )

        print(
            f"\n{int(TRAIN_FRAC * 100)}/"
            f"{int((1 - TRAIN_FRAC) * 100)} "
            f"train={len(train)} "
            f"test={len(test)}"
        )

        # --------------------------------------------------------
        # NEW:
        # Save processed train/test copies
        # --------------------------------------------------------

        write_split(
            train,
            test,
            "train",
            "test",
            target_size,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # Evaluation should use the NEW processed copies.
        # --------------------------------------------------------

        train_processed = [
            (
                (YES_DIR if lab == 1 else NO_DIR)
                / "train"
                / path.name,
                lab,
            )
            for path, lab in train
        ]

        test_processed = [
            (
                (YES_DIR if lab == 1 else NO_DIR)
                / "test"
                / path.name,
                lab,
            )
            for path, lab in test
        ]

        # --------------------------------------------------------
        # Evaluate templates
        # --------------------------------------------------------

        for name in TEMPLATE_NAMES:

            eval_template(
                name,
                train_processed,
                test_processed,
                TRAIN_FRAC
            )


if __name__ == "__main__":
    main()