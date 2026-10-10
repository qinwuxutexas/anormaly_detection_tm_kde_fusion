"""Paper-style statistical CF debris detector."""

from __future__ import annotations

import csv
import shutil
from itertools import product
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

DATA = Path(
    r"C:\Qinwu\Qinwu\Xu publication\template_match_cnn\data"
)

YES_DIR = DATA / "yes"
NO_DIR = DATA / "no"


# ============================================================
# DATA SPLIT
# ============================================================

SEED = 42
TRAIN_FRAC = 0.70


# ============================================================
# PREPROCESSING
# SAME AS YOUR WORKING TEMPLATE-MATCH CODE
# ============================================================

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0


# ============================================================
# PAPER STATISTICAL SEARCH
# ============================================================

WINDOW_WIDTH_FRACS = [
    0.05,
    # 0.10,
    # 0.15,
    # 0.20,
]

WINDOW_HEIGHT_FRACS = [
    0.25,
    # 0.50,
    # 0.75,
    # 1.00,
]

STRIDE_FRACS = [
    # 0.25,
    0.50,
]

# 0.10 is the paper setting
CF_THRESHOLDS = [
    0.02,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
]

PAPER_C = 2.55

AGGREGATIONS = [
    "max_anomaly",
    "top_k_anomaly_mean",
]

TOP_K = 5


# ============================================================
# IMAGE FUNCTIONS
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
# SAME TARGET RESOLUTION METHOD
# ============================================================

def find_target_resolution(
    paths: list[Path],
) -> tuple[int, int]:

    """
    Find candidate image with largest number of pixels.

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

            target_size = (
                w,
                h,
            )

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
# SAME PREPROCESSING METHOD
# ============================================================

def preprocess_image(
    img: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:

    """
    Candidate image:
        resize + Gaussian blur.
    """

    if target_size is not None:

        h, w = img.shape[:2]

        target_w, target_h = (
            target_size
        )

        if (w, h) != target_size:

            # Enlargement
            if (
                target_w > w
                or target_h > h
            ):

                interpolation = (
                    cv2.INTER_CUBIC
                )

            # Reduction
            else:

                interpolation = (
                    cv2.INTER_AREA
                )

            img = cv2.resize(
                img,
                target_size,
                interpolation=interpolation,
            )

    img = cv2.GaussianBlur(
        img,
        GAUSSIAN_KERNEL,
        GAUSSIAN_SIGMA,
    )

    return img


# ============================================================
# SAME STRATIFIED SPLIT
# ============================================================

def stratified_split(
    pos: list,
    neg: list,
    train_frac: float,
    seed: int,
):

    rng = np.random.default_rng(
        seed
    )

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
        [
            (p, 1)
            for p in pos[:n_tr_p]
        ]
        +
        [
            (p, 0)
            for p in neg[:n_tr_n]
        ]
    )

    test = (
        [
            (p, 1)
            for p in pos[n_tr_p:]
        ]
        +
        [
            (p, 0)
            for p in neg[n_tr_n:]
        ]
    )

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


# ============================================================
# SAME WRITE_SPLIT
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

                shutil.rmtree(
                    dest
                )

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
            img = load_gray(
                path
            )

            # Resize + Gaussian blur
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
# LOAD SAVED PROCESSED DATA
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

    if len(pairs) == 0:

        raise RuntimeError(
            f"No processed images "
            f"found for {split_name}."
        )

    return pairs


# ============================================================
# HISTOGRAM / CUMULATIVE FREQUENCY
# ============================================================

def cumulative_frequency(
    patch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    """
    Calculate grayscale histogram and normalized
    cumulative frequency.
    """

    hist = cv2.calcHist(
        [patch],
        [0],
        None,
        [256],
        [0, 256],
    ).ravel()

    total = float(
        hist.sum()
    )

    if total <= 0:

        return (
            np.arange(
                256,
                dtype=np.float64,
            ),
            np.zeros(
                256,
                dtype=np.float64,
            ),
        )

    pdf = (
        hist / total
    )

    cf = np.cumsum(
        pdf
    )

    intensity = np.arange(
        256,
        dtype=np.float64,
    )

    return (
        intensity,
        cf,
    )


# ============================================================
# PAPER STATISTICAL SCORE
#
# s = Delta_CF * C / Delta_I
#
# S_stat = 1 / (1 + exp(s))
# ============================================================

def paper_patch_score(
    patch: np.ndarray,
    cf_threshold: float,
    C: float = PAPER_C,
) -> tuple[float, float]:

    intensity, cf = (
        cumulative_frequency(
            patch
        )
    )

    # --------------------------------------------------------
    # First intensity where cumulative probability > 0
    # --------------------------------------------------------

    positive = np.flatnonzero(
        cf > 0.0
    )

    if len(positive) == 0:

        return (
            0.0,
            0.5,
        )

    start_idx = int(
        positive[0]
    )

    start_I = float(
        intensity[start_idx]
    )

    # --------------------------------------------------------
    # Intensity where CF reaches selected threshold
    # --------------------------------------------------------

    upper_idx = int(
        np.searchsorted(
            cf,
            cf_threshold,
            side="left",
        )
    )

    upper_idx = min(
        upper_idx,
        len(intensity) - 1,
    )

    upper_I = float(
        intensity[upper_idx]
    )

    # --------------------------------------------------------
    # Paper statistical index
    # --------------------------------------------------------

    delta_I = (
        upper_I
        - start_I
    )

    delta_CF = (
        cf_threshold
    )

    # Avoid division by zero
    if delta_I <= 0:

        s = float("inf")
        S_stat = 0.0

        return (
            s,
            S_stat,
        )

    s = (
        delta_CF
        * C
        / delta_I
    )

    # --------------------------------------------------------
    # Statistical anomaly score
    # --------------------------------------------------------

    S_stat = (
        1.0
        /
        (
            1.0
            + np.exp(
                min(
                    s,
                    700.0,
                )
            )
        )
    )

    return (
        float(s),
        float(S_stat),
    )


# ============================================================
# SLIDING WINDOW STATISTICAL SCORE
# ============================================================

def statistical_score(
    image: np.ndarray,
    window_width_frac: float,
    window_height_frac: float,
    stride_frac: float,
    cf_threshold: float,
    aggregation: str,
) -> float:

    h, w = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Window dimensions
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
    # Slide windows
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

            _, S_stat = (
                paper_patch_score(
                    patch,
                    cf_threshold,
                    PAPER_C,
                )
            )

            local_scores.append(
                S_stat
            )

    if len(local_scores) == 0:

        return 0.0

    local_scores = np.asarray(
        local_scores,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Image-level aggregation
    # --------------------------------------------------------

    if aggregation == "max_anomaly":

        return float(
            np.max(
                local_scores
            )
        )

    elif (
        aggregation
        == "top_k_anomaly_mean"
    ):

        k = min(
            TOP_K,
            len(local_scores),
        )

        top = np.partition(
            local_scores,
            -k,
        )[-k:]

        return float(
            np.mean(
                top
            )
        )

    else:

        raise ValueError(
            f"Unknown aggregation: "
            f"{aggregation}"
        )


# ============================================================
# SCORE ALL IMAGES
# ============================================================

def scores_for(
    pairs,
    width_frac: float,
    height_frac: float,
    stride_frac: float,
    cf_threshold: float,
    aggregation: str,
):

    scores = []
    labels = []

    for path, label in pairs:

        image = load_gray(
            path
        )

        score = statistical_score(
            image=image,
            window_width_frac=width_frac,
            window_height_frac=height_frac,
            stride_frac=stride_frac,
            cf_threshold=cf_threshold,
            aggregation=aggregation,
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
            dtype=np.float64,
        ),
        np.asarray(
            labels,
            dtype=int,
        ),
    )


# ============================================================
# METRICS
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
# FULL MEANINGFUL d THRESHOLD RANGE
# ============================================================

def candidate_thresholds(
    scores: np.ndarray,
) -> np.ndarray:

    """
    Search every threshold that can actually change
    classification.

    Uses:
        observed scores
        +
        midpoints between adjacent scores.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    scores = scores[
        np.isfinite(scores)
    ]

    if len(scores) == 0:

        raise ValueError(
            "No finite scores available."
        )

    unique_scores = np.unique(
        scores
    )

    if len(unique_scores) == 1:

        return (
            unique_scores.copy()
        )

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

    return np.unique(
        thresholds
    )


# ============================================================
# OPTIMIZE d ON TRAIN
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
            f"scores/y mismatch: "
            f"{len(scores)} vs {len(y)}"
        )

    # --------------------------------------------------------
    # Remove non-finite values
    # --------------------------------------------------------

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
            "All scores are NaN/Inf."
        )

    thresholds = (
        candidate_thresholds(
            scores
        )
    )

    best = None

    # --------------------------------------------------------
    # Test both score directions
    # --------------------------------------------------------

    for direction in (
        "high_S",
        "low_d",
    ):

        for threshold in thresholds:

            if direction == "high_S":

                pred = (
                    scores
                    >= threshold
                ).astype(int)

            else:

                pred = (
                    scores
                    <= threshold
                ).astype(int)

            m = metrics(
                pred,
                y,
            )

            candidate = {
                "threshold":
                    float(threshold),

                "direction":
                    direction,

                "metrics":
                    m,
            }

            if best is None:

                best = candidate

            elif (
                m["f1"]
                > best["metrics"]["f1"]
            ):

                best = candidate

            elif (
                np.isclose(
                    m["f1"],
                    best["metrics"]["f1"],
                )
                and
                m["fpr"]
                < best["metrics"]["fpr"]
            ):

                best = candidate

    return best


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def tune_hyperparameters(
    train_pairs,
):

    combinations = list(
        product(
            WINDOW_WIDTH_FRACS,
            WINDOW_HEIGHT_FRACS,
            STRIDE_FRACS,
            CF_THRESHOLDS,
            AGGREGATIONS,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "PAPER CF STATISTICAL SEARCH"
    )

    print(
        "========================================"
    )

    print(
        f"Total configurations: "
        f"{len(combinations)}"
    )

    best = None

    all_results = []

    # ========================================================
    # Search
    # ========================================================

    for i, config in enumerate(
        combinations,
        start=1,
    ):

        (
            width_frac,
            height_frac,
            stride_frac,
            cf_threshold,
            aggregation,
        ) = config

        print(
            f"\nStarting "
            f"[{i}/{len(combinations)}] "
            f"W={width_frac:.2f} "
            f"H={height_frac:.2f} "
            f"stride={stride_frac:.2f} "
            f"CF={cf_threshold:.3f} "
            f"agg={aggregation}",
            flush=True,
        )

        # ----------------------------------------------------
        # TRAIN scores
        # ----------------------------------------------------

        scores, y = scores_for(
            train_pairs,
            width_frac,
            height_frac,
            stride_frac,
            cf_threshold,
            aggregation,
        )

        # ----------------------------------------------------
        # Optimize decision threshold d
        # ----------------------------------------------------

        result = find_best_threshold(
            scores,
            y,
        )

        m = result[
            "metrics"
        ]

        # ----------------------------------------------------
        # Store ALL results
        # ----------------------------------------------------

        row = {
            "rank": 0,

            "width_frac":
                width_frac,

            "height_frac":
                height_frac,

            "stride_frac":
                stride_frac,

            "cf_threshold":
                cf_threshold,

            "aggregation":
                aggregation,

            "direction":
                result["direction"],

            "d_threshold":
                result["threshold"],

            "accuracy":
                m["accuracy"],

            "precision":
                m["precision"],

            "recall":
                m["recall"],

            "fpr":
                m["fpr"],

            "f1":
                m["f1"],

            "tp":
                m["tp"],

            "fp":
                m["fp"],

            "tn":
                m["tn"],

            "fn":
                m["fn"],
        }

        all_results.append(
            row
        )

        # ----------------------------------------------------
        # Console result
        # ----------------------------------------------------

        print(
            f"[{i:3d}/{len(combinations)}] "
            f"W={width_frac:.2f} "
            f"H={height_frac:.2f} "
            f"stride={stride_frac:.2f} "
            f"CF={cf_threshold:.3f} "
            f"agg={aggregation:18s} "
            f"dir={result['direction']:6s} "
            f"d={result['threshold']:.6f} "
            f"P={m['precision']:.3f} "
            f"R={m['recall']:.3f} "
            f"FPR={m['fpr']:.3f} "
            f"F1={m['f1']:.3f}",
            flush=True,
        )

        # ----------------------------------------------------
        # Overall best
        # ----------------------------------------------------

        candidate = {
            "width_frac":
                width_frac,

            "height_frac":
                height_frac,

            "stride_frac":
                stride_frac,

            "cf_threshold":
                cf_threshold,

            "aggregation":
                aggregation,

            "threshold":
                result["threshold"],

            "direction":
                result["direction"],

            "metrics":
                m,
        }

        better = False

        if best is None:

            better = True

        elif (
            m["f1"]
            > best["metrics"]["f1"]
        ):

            better = True

        elif (
            np.isclose(
                m["f1"],
                best["metrics"]["f1"],
            )
            and
            m["fpr"]
            < best["metrics"]["fpr"]
        ):

            better = True

        if better:

            best = candidate

            print(
                "    <-- NEW BEST",
                flush=True,
            )

    # ========================================================
    # SORT ALL 448 RESULTS
    #
    # 1. highest F1
    # 2. lowest FPR
    # 3. highest recall
    # ========================================================

    all_results.sort(
        key=lambda x: (
            -x["f1"],
            x["fpr"],
            -x["recall"],
        )
    )

    for rank, row in enumerate(
        all_results,
        start=1,
    ):

        row["rank"] = rank

    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    fieldnames = [
        "rank",
        "width_frac",
        "height_frac",
        "stride_frac",
        "cf_threshold",
        "aggregation",
        "direction",
        "d_threshold",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    all_results_path = (
        DATA
        / "statistical_paper_all_results.csv"
    )

    with open(
        all_results_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    # ========================================================
    # TOP 10
    # ========================================================

    top10 = (
        all_results[:10]
    )

    top10_path = (
        DATA
        / "statistical_paper_top10.csv"
    )

    with open(
        top10_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            top10
        )

    # ========================================================
    # PRINT TOP 10
    # ========================================================

    print(
        "\n\n========================================"
    )

    print(
        "TOP 10 TRAIN CONFIGURATIONS"
    )

    print(
        "========================================"
    )

    for row in top10:

        print(
            f"#{row['rank']:2d} "
            f"W={row['width_frac']:.2f} "
            f"H={row['height_frac']:.2f} "
            f"stride={row['stride_frac']:.2f} "
            f"CF={row['cf_threshold']:.3f} "
            f"agg={row['aggregation']:18s} "
            f"dir={row['direction']:6s} "
            f"d={row['d_threshold']:.6f} "
            f"P={row['precision']:.3f} "
            f"R={row['recall']:.3f} "
            f"FPR={row['fpr']:.3f} "
            f"F1={row['f1']:.3f}"
        )

    print(
        f"\nSaved ALL results:\n"
        f"{all_results_path}"
    )

    print(
        f"\nSaved TOP 10:\n"
        f"{top10_path}"
    )

    if best is None:

        raise RuntimeError(
            "No valid configuration found."
        )

    return best


# ============================================================
# TEST EVALUATION
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
        best["cf_threshold"],
        best["aggregation"],
    )

    threshold = (
        best["threshold"]
    )

    if (
        best["direction"]
        == "high_S"
    ):

        pred = (
            scores
            >= threshold
        ).astype(int)

    else:

        pred = (
            scores
            <= threshold
        ).astype(int)

    m = metrics(
        pred,
        y,
    )

    return (
        scores,
        y,
        pred,
        m,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nSTATISTICAL KDE/CF DEBRIS SCRIPT STARTED",
        flush=True,
    )

    # ========================================================
    # 1. ORIGINAL IMAGES
    # ========================================================

    pos = list_images(
        YES_DIR
    )

    neg = list_images(
        NO_DIR
    )

    if not pos or not neg:

        raise SystemExit(
            f"Need original images in:\n"
            f"{YES_DIR}\n"
            f"{NO_DIR}"
        )

    print(
        f"\nOriginal dataset:"
        f"\n  debris     = {len(pos)}"
        f"\n  no debris  = {len(neg)}"
        f"\n  total      = "
        f"{len(pos) + len(neg)}",
        flush=True,
    )

    # ========================================================
    # 2. TARGET RESOLUTION
    # ========================================================

    target_size = (
        find_target_resolution(
            pos + neg
        )
    )

    # ========================================================
    # 3. 70/30 STRATIFIED SPLIT
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
        f"\n70/30 split:"
        f"\n  train = {len(train)}"
        f"\n  test  = {len(test)}",
        flush=True,
    )

    # ========================================================
    # 4. RESIZE + BLUR + SAVE
    # ========================================================

    write_split(
        train,
        test,
        "train",
        "test",
        target_size,
    )

    # ========================================================
    # 5. LOAD SAVED PROCESSED DATA
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
    # 6. TRAIN HYPERPARAMETER SEARCH
    # ========================================================

    best = tune_hyperparameters(
        train_processed
    )

    m = best[
        "metrics"
    ]

    # ========================================================
    # BEST TRAIN RESULT
    # ========================================================

    print(
        "\n\n========================================"
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
        f"CF threshold           : "
        f"{best['cf_threshold']}"
    )

    print(
        f"Paper C                : "
        f"{PAPER_C}"
    )

    print(
        f"Aggregation            : "
        f"{best['aggregation']}"
    )

    print(
        f"Direction              : "
        f"{best['direction']}"
    )

    print(
        f"d threshold            : "
        f"{best['threshold']:.6f}"
    )

    print(
        "\nTRAIN METRICS:"
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
        f"TP={m['tp']} "
        f"FP={m['fp']} "
        f"TN={m['tn']} "
        f"FN={m['fn']}"
    )

    # ========================================================
    # 7. FINAL TEST
    # ========================================================

    (
        test_scores,
        test_y,
        test_pred,
        test_metrics,
    ) = evaluate_test(
        test_processed,
        best,
    )

    print(
        "\n\n========================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Frozen CF threshold = "
        f"{best['cf_threshold']}"
    )

    print(
        f"Frozen d threshold  = "
        f"{best['threshold']:.6f}"
    )

    print(
        f"Direction           = "
        f"{best['direction']}"
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
        f"TP={test_metrics['tp']} "
        f"FP={test_metrics['fp']} "
        f"TN={test_metrics['tn']} "
        f"FN={test_metrics['fn']}"
    )

    # ========================================================
    # TEST SCORE DISTRIBUTIONS
    # ========================================================

    positive_scores = (
        test_scores[
            test_y == 1
        ]
    )

    negative_scores = (
        test_scores[
            test_y == 0
        ]
    )

    print(
        "\nTEST SCORE DISTRIBUTION:"
    )

    if len(positive_scores):

        print(
            f"Debris: "
            f"mean={positive_scores.mean():.6f} "
            f"median={np.median(positive_scores):.6f} "
            f"min={positive_scores.min():.6f} "
            f"max={positive_scores.max():.6f}"
        )

    if len(negative_scores):

        print(
            f"No debris: "
            f"mean={negative_scores.mean():.6f} "
            f"median={np.median(negative_scores):.6f} "
            f"min={negative_scores.min():.6f} "
            f"max={negative_scores.max():.6f}"
        )

    print(
        "\nDONE.",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()