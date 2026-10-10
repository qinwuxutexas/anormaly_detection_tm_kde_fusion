"""Statistical sliding-window debris detection.

Uses EXACTLY the same preprocessing/data split method as the
template-matching code:

1. Read original images from data/yes and data/no
2. Stratified train/test split
3. Find highest-resolution candidate image
4. Resize every candidate to that resolution
5. Gaussian blur
6. Save processed copies into:
       data/yes/train
       data/yes/test
       data/no/train
       data/no/test
7. Run statistical anomaly detection on those SAVED images
8. Tune statistical hyperparameters + threshold on TRAIN F1
9. Evaluate the selected configuration on TEST
"""

from __future__ import annotations

import shutil
from itertools import product
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DATA = Path(
    r"C:\Qinwu\Qinwu\Xu publication\template_match_cnn\data"
)

YES_DIR = DATA / "yes"
NO_DIR = DATA / "no"

SEED = 42
TRAIN_FRAC = 0.70


# ------------------------------------------------------------
# EXACT SAME preprocessing configuration
# ------------------------------------------------------------

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0


# ============================================================
# Statistical hyperparameter search
# ============================================================

WINDOW_WIDTH_FRACS = [
    0.05,
    0.10,
    0.15,
    0.20,
]

WINDOW_HEIGHT_FRACS = [
    0.25,
    0.50,
    0.75,
    1.00,
]

STRIDE_FRACS = [
    0.25,
    0.50,
]

QUANTILE_PAIRS = [
    (0.05, 0.95),
    (0.10, 0.90),
    (0.20, 0.80),
]

AGGREGATIONS = [
    "max",
    "top_k_mean",
]

TOP_K = 5


# ============================================================
# EXACT SAME image loading code
# ============================================================

def list_images(folder: Path) -> list[Path]:

    exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in exts
    )


def load_gray(path: Path) -> np.ndarray:

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise FileNotFoundError(path)

    return img


# ============================================================
# EXACT SAME: find highest-resolution candidate image
# ============================================================

def find_target_resolution(
    paths: list[Path],
) -> tuple[int, int]:

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
        raise RuntimeError(
            "No candidate images found."
        )

    print(
        f"\nHighest-resolution candidate:\n"
        f"  {target_path}\n"
        f"  resolution = "
        f"{target_size[0]} x {target_size[1]}"
    )

    return target_size


# ============================================================
# EXACT SAME preprocessing
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


# ============================================================
# EXACT SAME stratified split
# ============================================================

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

    n_tr_p = int(
        len(pos) * train_frac
    )

    n_tr_n = int(
        len(neg) * train_frac
    )

    train = (
        [(p, 1) for p in pos[:n_tr_p]]
        +
        [(p, 0) for p in neg[:n_tr_n]]
    )

    test = (
        [(p, 1) for p in pos[n_tr_p:]]
        +
        [(p, 0) for p in neg[n_tr_n:]]
    )

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


# ============================================================
# EXACT SAME write_split
# Save resized + Gaussian-blurred copies
# ============================================================

def write_split(
    train,
    test,
    train_name: str,
    test_name: str,
    target_size: tuple[int, int],
) -> None:

    # Re-create output folders
    for parent in (
        YES_DIR,
        NO_DIR,
    ):

        for name in (
            train_name,
            test_name,
        ):

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

            # resize to uniform resolution
            # + Gaussian blur
            img = preprocess_image(
                img,
                target_size,
            )

            dest_path = (
                dest_dir / path.name
            )

            ok = cv2.imwrite(
                str(dest_path),
                img,
            )

            if not ok:
                raise IOError(
                    f"Could not save "
                    f"{dest_path}"
                )

    print(
        f"\nwrote processed images:\n"
        f"{YES_DIR / train_name}\n"
        f"{YES_DIR / test_name}\n"
        f"{NO_DIR / train_name}\n"
        f"{NO_DIR / test_name}"
    )


# ============================================================
# Reload SAVED processed train/test images
# ============================================================

def load_processed_split(
    split_name: str,
):

    yes_folder = (
        YES_DIR / split_name
    )

    no_folder = (
        NO_DIR / split_name
    )

    pos = list_images(
        yes_folder
    )

    neg = list_images(
        no_folder
    )

    pairs = (
        [(p, 1) for p in pos]
        +
        [(p, 0) for p in neg]
    )

    print(
        f"\nProcessed {split_name}:"
        f"\n  debris     = {len(pos)}"
        f"\n  no debris  = {len(neg)}"
        f"\n  total      = {len(pairs)}"
    )

    if not pairs:

        raise RuntimeError(
            f"No processed images found in "
            f"{yes_folder} and {no_folder}"
        )

    return pairs


# ============================================================
# Statistical distribution / empirical CDF
# ============================================================

def empirical_cdf(
    patch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    """
    Compute empirical cumulative distribution
    of grayscale intensity.
    """

    hist = cv2.calcHist(
        [patch],
        [0],
        None,
        [256],
        [0, 256],
    ).ravel()

    total = hist.sum()

    if total <= 0:

        return (
            np.arange(256),
            np.zeros(
                256,
                dtype=np.float32,
            ),
        )

    pdf = (
        hist / total
    )

    cdf = np.cumsum(
        pdf
    )

    intensity = np.arange(
        256
    )

    return (
        intensity,
        cdf,
    )


# ============================================================
# Local statistical tail/spread score
# ============================================================

def patch_tail_score(
    patch: np.ndarray,
    q_low: float,
    q_high: float,
) -> float:

    """
    Statistical anomaly score for one local patch.

    Find the intensity interval containing the selected
    cumulative probability range:

        I(q_high) - I(q_low)

    Normalize to [0,1].

    Broader intensity distribution / longer tail
    -> higher anomaly score.
    """

    intensity, cdf = (
        empirical_cdf(
            patch
        )
    )

    low_idx = np.searchsorted(
        cdf,
        q_low,
    )

    high_idx = np.searchsorted(
        cdf,
        q_high,
    )

    low_idx = min(
        low_idx,
        len(intensity) - 1,
    )

    high_idx = min(
        high_idx,
        len(intensity) - 1,
    )

    low_intensity = (
        intensity[low_idx]
    )

    high_intensity = (
        intensity[high_idx]
    )

    spread = (
        high_intensity
        - low_intensity
    )

    score = (
        spread / 255.0
    )

    return float(score)


# ============================================================
# Sliding-window statistical detector
# ============================================================

def statistical_score(
    image: np.ndarray,
    window_width_frac: float,
    window_height_frac: float,
    stride_frac: float,
    q_low: float,
    q_high: float,
    aggregation: str,
    top_k: int = 5,
) -> float:

    """
    Slide a vertical/local kernel through the image.

    Each local window receives a statistical tail score.

    Image-level score:
        max
    or
        mean of top-K local scores.
    """

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Window size
    # --------------------------------------------------------

    win_w = max(
        2,
        int(
            round(
                w
                * window_width_frac
            )
        ),
    )

    win_h = max(
        2,
        int(
            round(
                h
                * window_height_frac
            )
        ),
    )

    win_w = min(
        win_w,
        w,
    )

    win_h = min(
        win_h,
        h,
    )

    # --------------------------------------------------------
    # Stride
    # --------------------------------------------------------

    stride_x = max(
        1,
        int(
            round(
                win_w
                * stride_frac
            )
        ),
    )

    stride_y = max(
        1,
        int(
            round(
                win_h
                * stride_frac
            )
        ),
    )

    local_scores = []

    # --------------------------------------------------------
    # Sliding window
    # --------------------------------------------------------

    for y in range(
        0,
        h - win_h + 1,
        stride_y,
    ):

        for x in range(
            0,
            w - win_w + 1,
            stride_x,
        ):

            patch = image[
                y:y + win_h,
                x:x + win_w,
            ]

            score = patch_tail_score(
                patch,
                q_low,
                q_high,
            )

            local_scores.append(
                score
            )

    if len(local_scores) == 0:
        return 0.0

    local_scores = np.asarray(
        local_scores,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    if aggregation == "max":

        return float(
            np.max(
                local_scores
            )
        )

    elif aggregation == "top_k_mean":

        k = min(
            top_k,
            len(local_scores),
        )

        top = np.partition(
            local_scores,
            -k,
        )[-k:]

        return float(
            np.mean(top)
        )

    else:

        raise ValueError(
            f"Unknown aggregation: "
            f"{aggregation}"
        )


# ============================================================
# Calculate scores for all images
# ============================================================

def scores_for(
    pairs,
    window_width_frac: float,
    window_height_frac: float,
    stride_frac: float,
    q_low: float,
    q_high: float,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:

    scores = []
    labels = []

    for path, label in pairs:

        image = load_gray(
            path
        )

        score = statistical_score(
            image=image,
            window_width_frac=window_width_frac,
            window_height_frac=window_height_frac,
            stride_frac=stride_frac,
            q_low=q_low,
            q_high=q_high,
            aggregation=aggregation,
            top_k=TOP_K,
        )

        scores.append(
            score
        )

        labels.append(
            label
        )

    return (
        np.asarray(
            scores,
            dtype=np.float32,
        ),
        np.asarray(
            labels,
            dtype=int,
        ),
    )


# ============================================================
# Metrics
# ============================================================

def metrics(
    pred: np.ndarray,
    y: np.ndarray,
) -> dict:

    tp = int(
        np.sum(
            (pred == 1)
            & (y == 1)
        )
    )

    fp = int(
        np.sum(
            (pred == 1)
            & (y == 0)
        )
    )

    tn = int(
        np.sum(
            (pred == 0)
            & (y == 0)
        )
    )

    fn = int(
        np.sum(
            (pred == 0)
            & (y == 1)
        )
    )

    accuracy = (
        (tp + tn) / len(y)
        if len(y)
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
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
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ============================================================
# Find best threshold on TRAIN
# ============================================================

def find_best_threshold(
    scores: np.ndarray,
    y: np.ndarray,
):

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    if len(scores) == 0:

        raise ValueError(
            "scores is empty"
        )

    if len(scores) != len(y):

        raise ValueError(
            f"scores/y length mismatch: "
            f"{len(scores)} vs {len(y)}"
        )

    if not np.all(
        np.isfinite(scores)
    ):

        valid = np.isfinite(
            scores
        )

        scores = scores[
            valid
        ]

        y = y[
            valid
        ]

    if len(scores) == 0:

        raise ValueError(
            "All scores are NaN/Inf"
        )

    # --------------------------------------------------------
    # Candidate thresholds
    # --------------------------------------------------------

    unique_scores = np.unique(
        scores
    )

    if len(unique_scores) == 1:

        thresholds = (
            unique_scores.copy()
        )

    else:

        midpoints = (
            unique_scores[:-1]
            + unique_scores[1:]
        ) / 2.0

        thresholds = np.concatenate(
            [
                unique_scores,
                midpoints,
            ]
        )

    thresholds = np.unique(
        thresholds
    )

    # --------------------------------------------------------
    # Initialize using first threshold
    # --------------------------------------------------------

    best_threshold = float(
        thresholds[0]
    )

    pred = (
        scores
        >= best_threshold
    ).astype(int)

    best_metrics = metrics(
        pred,
        y,
    )

    best_f1 = (
        best_metrics["f1"]
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    for threshold in thresholds[1:]:

        pred = (
            scores
            >= threshold
        ).astype(int)

        m = metrics(
            pred,
            y,
        )

        if m["f1"] > best_f1:

            best_f1 = (
                m["f1"]
            )

            best_threshold = float(
                threshold
            )

            best_metrics = m

        elif (
            np.isclose(
                m["f1"],
                best_f1,
            )
            and
            m["fpr"]
            < best_metrics["fpr"]
        ):

            best_threshold = float(
                threshold
            )

            best_metrics = m

    return (
        best_threshold,
        best_metrics,
    )


# ============================================================
# Hyperparameter search on TRAIN
# ============================================================

def tune_hyperparameters(
    train_pairs,
):

    best = None
    best_f1 = -1.0

    combinations = list(
        product(
            WINDOW_WIDTH_FRACS,
            WINDOW_HEIGHT_FRACS,
            STRIDE_FRACS,
            QUANTILE_PAIRS,
            AGGREGATIONS,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "STATISTICAL HYPERPARAMETER SEARCH"
    )

    print(
        "========================================"
    )

    print(
        f"Total configurations: "
        f"{len(combinations)}"
    )

    for i, config in enumerate(
        combinations,
        start=1,
    ):

        (
            width_frac,
            height_frac,
            stride_frac,
            quantiles,
            aggregation,
        ) = config

        q_low, q_high = (
            quantiles
        )

        scores, y = scores_for(
            train_pairs,
            width_frac,
            height_frac,
            stride_frac,
            q_low,
            q_high,
            aggregation,
        )

        threshold, m = (
            find_best_threshold(
                scores,
                y,
            )
        )

        print(
            f"[{i:3d}/{len(combinations)}] "
            f"W={width_frac:.2f} "
            f"H={height_frac:.2f} "
            f"stride={stride_frac:.2f} "
            f"q=({q_low:.2f},{q_high:.2f}) "
            f"agg={aggregation:10s} "
            f"thr={threshold:.4f} "
            f"P={m['precision']:.3f} "
            f"R={m['recall']:.3f} "
            f"FPR={m['fpr']:.3f} "
            f"F1={m['f1']:.3f}"
        )

        # ----------------------------------------------------
        # Select highest TRAIN F1
        # Tie: lower FPR
        # ----------------------------------------------------

        better = False

        if m["f1"] > best_f1:

            better = True

        elif (
            np.isclose(
                m["f1"],
                best_f1,
            )
            and
            best is not None
            and
            m["fpr"]
            < best["metrics"]["fpr"]
        ):

            better = True

        if better:

            best_f1 = (
                m["f1"]
            )

            best = {
                "width_frac":
                    width_frac,

                "height_frac":
                    height_frac,

                "stride_frac":
                    stride_frac,

                "q_low":
                    q_low,

                "q_high":
                    q_high,

                "aggregation":
                    aggregation,

                "threshold":
                    threshold,

                "metrics":
                    m,
            }

            print(
                "    <-- NEW BEST"
            )

    if best is None:

        raise RuntimeError(
            "No valid statistical "
            "configuration found."
        )

    return best


# ============================================================
# Evaluate TEST using frozen TRAIN-selected configuration
# ============================================================

def evaluate_test(
    test_pairs,
    best,
):

    scores, y = scores_for(
        test_pairs,
        best["width_frac"],
        best["height_frac"],
        best["stride_frac"],
        best["q_low"],
        best["q_high"],
        best["aggregation"],
    )

    threshold = (
        best["threshold"]
    )

    pred = (
        scores >= threshold
    ).astype(int)

    m = metrics(
        pred,
        y,
    )

    return (
        scores,
        y,
        m,
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # STEP 1:
    # Load ORIGINAL images exactly like template-match code
    # ========================================================

    pos = list_images(
        YES_DIR
    )

    neg = list_images(
        NO_DIR
    )

    if not pos or not neg:

        raise SystemExit(
            f"Need images in "
            f"{YES_DIR} and {NO_DIR}"
        )

    print(
        f"\nOriginal dataset:"
        f"\n  debris     = {len(pos)}"
        f"\n  no debris  = {len(neg)}"
        f"\n  total      = {len(pos) + len(neg)}"
    )

    # ========================================================
    # STEP 2:
    # EXACT SAME highest-resolution calculation
    # ========================================================

    all_images = (
        pos + neg
    )

    target_size = (
        find_target_resolution(
            all_images
        )
    )

    # ========================================================
    # STEP 3:
    # EXACT SAME stratified split
    # ========================================================

    train, test = (
        stratified_split(
            pos,
            neg,
            TRAIN_FRAC,
            SEED,
        )
    )

    print(
        f"\n"
        f"{int(TRAIN_FRAC * 100)}/"
        f"{int((1 - TRAIN_FRAC) * 100)} "
        f"split:"
        f"\n  train = {len(train)}"
        f"\n  test  = {len(test)}"
    )

    # ========================================================
    # STEP 4:
    # EXACT SAME write_split:
    #
    # resize -> Gaussian blur -> SAVE
    # ========================================================

    write_split(
        train,
        test,
        "train",
        "test",
        target_size,
    )

    # ========================================================
    # STEP 5:
    # Reload the SAVED processed images.
    #
    # Statistical detector therefore NEVER operates on the
    # original images.
    # ========================================================

    train_processed = (
        load_processed_split(
            "train"
        )
    )

    test_processed = (
        load_processed_split(
            "test"
        )
    )

    # ========================================================
    # STEP 6:
    # Tune statistical method on processed TRAIN
    # ========================================================

    best = tune_hyperparameters(
        train_processed
    )

    print(
        "\n\n"
        "========================================"
    )

    print(
        "BEST TRAIN CONFIGURATION"
    )

    print(
        "========================================"
    )

    print(
        f"Window width fraction  : "
        f"{best['width_frac']}"
    )

    print(
        f"Window height fraction : "
        f"{best['height_frac']}"
    )

    print(
        f"Stride fraction        : "
        f"{best['stride_frac']}"
    )

    print(
        f"CDF quantiles          : "
        f"{best['q_low']}, "
        f"{best['q_high']}"
    )

    print(
        f"Aggregation            : "
        f"{best['aggregation']}"
    )

    print(
        f"Threshold              : "
        f"{best['threshold']:.4f}"
    )

    m = best["metrics"]

    print(
        "\nTRAIN RESULTS:"
    )

    print(
        f"Accuracy  = "
        f"{m['accuracy']:.4f}"
    )

    print(
        f"Precision = "
        f"{m['precision']:.4f}"
    )

    print(
        f"Recall    = "
        f"{m['recall']:.4f}"
    )

    print(
        f"FPR       = "
        f"{m['fpr']:.4f}"
    )

    print(
        f"F1        = "
        f"{m['f1']:.4f}"
    )

    print(
        f"TP={m['tp']}  "
        f"FP={m['fp']}  "
        f"TN={m['tn']}  "
        f"FN={m['fn']}"
    )

    # ========================================================
    # STEP 7:
    # TEST once with frozen configuration + threshold
    # ========================================================

    (
        test_scores,
        test_y,
        test_metrics,
    ) = evaluate_test(
        test_processed,
        best,
    )

    print(
        "\n\n"
        "========================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Threshold fixed from train: "
        f"{best['threshold']:.4f}"
    )

    print(
        f"Accuracy  = "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision = "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall    = "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"FPR       = "
        f"{test_metrics['fpr']:.4f}"
    )

    print(
        f"F1        = "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        f"TP={test_metrics['tp']}  "
        f"FP={test_metrics['fp']}  "
        f"TN={test_metrics['tn']}  "
        f"FN={test_metrics['fn']}"
    )

    # ========================================================
    # Individual TEST scores
    # ========================================================

    print(
        "\nIndividual TEST "
        "statistical scores:"
    )

    test_pred = (
        test_scores
        >= best["threshold"]
    ).astype(int)

    for (
        (path, label),
        score,
        pred,
    ) in zip(
        test_processed,
        test_scores,
        test_pred,
    ):

        print(
            f"{path.name:35s} "
            f"label={label} "
            f"score={score:.4f} "
            f"pred={pred}"
        )


if __name__ == "__main__":
    main()