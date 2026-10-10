"""Paper KDE/CF statistical debris detector.

Fixed window geometry:
    width  = 5% image width
    height = 25% image height
    stride = 50% of window size

Only CF percentile and decision threshold are evaluated.

Preprocessing is kept identical to the working template-matching code:
    - 70/30 stratified split
    - highest-resolution candidate determines target size
    - INTER_CUBIC enlargement / INTER_AREA reduction
    - Gaussian blur (5,5)
    - processed images saved to yes/train, yes/test,
      no/train, no/test
"""

from __future__ import annotations

import csv
import shutil
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
# SPLIT
# ============================================================

SEED = 42
TRAIN_FRAC = 0.70


# ============================================================
# PREPROCESSING -- SAME AS TEMPLATE CODE
# ============================================================

GAUSSIAN_KERNEL = (5, 5)
GAUSSIAN_SIGMA = 0


# ============================================================
# FIXED WINDOW -- YOUR SELECTED WINDOW
# ============================================================

WINDOW_WIDTH_FRAC = 0.05
WINDOW_HEIGHT_FRAC = 0.25
STRIDE_FRAC = 0.50


# ============================================================
# STATISTICAL PARAMETERS
# ============================================================

# Paper value 0.10 is included.
CF_THRESHOLDS = [
    0.02,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
]

# Paper normalization constant.
PAPER_C = 2.55


# ------------------------------------------------------------
# KDE bandwidth in grayscale intensity units.
#
# Instead of treating the raw histogram as KDE, smooth the
# intensity histogram with a Gaussian kernel.
#
# Test several reasonable bandwidths because h is explicitly
# a KDE parameter in the paper but no single h is specified
# in the implementation section.
# ------------------------------------------------------------

KDE_BANDWIDTHS = [
    1.0,
    2.0,
    4.0,
    8.0,
]


# ============================================================
# IMAGE IO
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
# SAME TARGET-RESOLUTION CODE
# ============================================================

def find_target_resolution(
    paths: list[Path],
) -> tuple[int, int]:

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
# SAME PREPROCESSING CODE
# ============================================================

def preprocess_image(
    img: np.ndarray,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:

    if target_size is not None:

        h, w = img.shape[:2]

        target_w, target_h = target_size

        if (w, h) != target_size:

            # Enlargement
            if target_w > w or target_h > h:

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

    # Gaussian blur
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

    rng = np.random.default_rng(seed)

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
# SAME WRITE_SPLIT
# ============================================================

def write_split(
    train,
    test,
    train_name: str,
    test_name: str,
    target_size: tuple[int, int],
) -> None:

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

            img = load_gray(path)

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
# LOAD PROCESSED SPLIT
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
            f"No processed images "
            f"found for {split_name}."
        )

    return pairs


# ============================================================
# KDE
# ============================================================

def kde_pdf_cdf(
    patch: np.ndarray,
    bandwidth: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    """
    KDE approximation on discrete intensity grid [0,255].

    1. Calculate normalized intensity histogram.
    2. Gaussian smooth histogram using bandwidth h.
    3. Normalize smoothed PDF.
    4. Integrate to obtain CDF.

    This follows the paper's practical note that KDE may be
    approximated on a discretized intensity grid.
    """

    hist = np.bincount(
        patch.ravel(),
        minlength=256,
    ).astype(np.float64)

    total = hist.sum()

    if total <= 0:

        intensity = np.arange(
            256,
            dtype=np.float64,
        )

        return (
            intensity,
            np.zeros(256),
            np.zeros(256),
        )

    hist /= total

    # --------------------------------------------------------
    # Gaussian KDE approximation
    #
    # OpenCV sigma is expressed in intensity-bin units.
    # kernel size is selected automatically from sigma.
    # --------------------------------------------------------

    pdf = cv2.GaussianBlur(
        hist.reshape(-1, 1),
        ksize=(1, 0),
        sigmaX=0,
        sigmaY=bandwidth,
        borderType=cv2.BORDER_REPLICATE,
    ).ravel()

    pdf_sum = pdf.sum()

    if pdf_sum <= 0:

        pdf = hist.copy()

    else:

        pdf /= pdf_sum

    cdf = np.cumsum(pdf)

    # Numerical normalization
    if cdf[-1] > 0:

        cdf /= cdf[-1]

    intensity = np.arange(
        256,
        dtype=np.float64,
    )

    return (
        intensity,
        pdf,
        cdf,
    )


# ============================================================
# PAPER s AND S_stat
# ============================================================

def paper_patch_score(
    patch: np.ndarray,
    cf_threshold: float,
    bandwidth: float,
) -> tuple[float, float]:

    """
    Paper:

        s = Delta_F * C / Delta_I

        S_stat = 1 / (1 + exp(s))

    Lower s -> flatter/broader distribution according
    to Eq. 3 / Eq. 9 interpretation.

    Return both values so we do NOT silently choose the
    paper's internally inconsistent d direction.
    """

    intensity, pdf, cdf = (
        kde_pdf_cdf(
            patch,
            bandwidth,
        )
    )

    # --------------------------------------------------------
    # Start of KDE/CDF support.
    #
    # KDE technically has nonzero support everywhere, so
    # "first CDF > 0" is not useful.
    #
    # Use a tiny probability floor to identify the effective
    # beginning of the distribution.
    # --------------------------------------------------------

    support_floor = 1e-4

    support = np.flatnonzero(
        cdf >= support_floor
    )

    if len(support) == 0:

        return (
            np.inf,
            0.0,
        )

    start_idx = int(
        support[0]
    )

    start_I = float(
        intensity[start_idx]
    )

    start_F = float(
        cdf[start_idx]
    )

    # --------------------------------------------------------
    # CF percentile point
    # --------------------------------------------------------

    upper_idx = int(
        np.searchsorted(
            cdf,
            cf_threshold,
            side="left",
        )
    )

    upper_idx = min(
        max(
            upper_idx,
            start_idx,
        ),
        255,
    )

    upper_I = float(
        intensity[upper_idx]
    )

    upper_F = float(
        cdf[upper_idx]
    )

    # --------------------------------------------------------
    # Actual measured Delta F and Delta I
    # --------------------------------------------------------

    delta_F = (
        upper_F
        - start_F
    )

    delta_I = (
        upper_I
        - start_I
    )

    if (
        delta_I <= 0
        or delta_F <= 0
    ):

        return (
            np.inf,
            0.0,
        )

    s = (
        delta_F
        * PAPER_C
        / delta_I
    )

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
# IMAGE SCORE
# ============================================================

def statistical_score(
    image: np.ndarray,
    cf_threshold: float,
    bandwidth: float,
) -> tuple[float, float]:

    """
    Fixed geometry:

        W = 0.05
        H = 0.25
        stride = 0.50

    Returns:

        image_s_min

            Minimum local s.
            This corresponds to the paper's Eq. 9 statement
            that lower s implies stronger anomaly tendency.

        image_S_max

            Maximum local S_stat.
            This is equivalent in ranking because S_stat
            decreases monotonically with s.
    """

    h, w = image.shape[:2]

    win_w = max(
        2,
        int(
            round(
                w
                * WINDOW_WIDTH_FRAC
            )
        ),
    )

    win_h = max(
        2,
        int(
            round(
                h
                * WINDOW_HEIGHT_FRAC
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

    stride_x = max(
        1,
        int(
            round(
                win_w
                * STRIDE_FRAC
            )
        ),
    )

    stride_y = max(
        1,
        int(
            round(
                win_h
                * STRIDE_FRAC
            )
        ),
    )

    local_s = []
    local_S = []

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

            s, S_stat = (
                paper_patch_score(
                    patch,
                    cf_threshold,
                    bandwidth,
                )
            )

            if np.isfinite(s):

                local_s.append(
                    s
                )

                local_S.append(
                    S_stat
                )

    if not local_s:

        return (
            np.inf,
            0.0,
        )

    return (
        float(
            np.min(local_s)
        ),
        float(
            np.max(local_S)
        ),
    )


# ============================================================
# SCORE DATASET
# ============================================================

def scores_for(
    pairs,
    cf_threshold: float,
    bandwidth: float,
):

    s_scores = []
    S_scores = []
    labels = []

    for path, label in pairs:

        image = load_gray(path)

        s, S_stat = statistical_score(
            image,
            cf_threshold,
            bandwidth,
        )

        s_scores.append(s)
        S_scores.append(S_stat)
        labels.append(label)

    return (
        np.asarray(
            s_scores,
            dtype=np.float64,
        ),
        np.asarray(
            S_scores,
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
# MEANINGFUL THRESHOLDS
# ============================================================

def candidate_thresholds(
    scores: np.ndarray,
) -> np.ndarray:

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    scores = scores[
        np.isfinite(scores)
    ]

    if len(scores) == 0:

        raise ValueError(
            "No finite scores."
        )

    unique = np.unique(
        scores
    )

    if len(unique) == 1:

        return unique

    mids = (
        unique[:-1]
        + unique[1:]
    ) / 2.0

    return np.unique(
        np.concatenate(
            [
                unique,
                mids,
            ]
        )
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    anomaly_if: str,
):

    thresholds = (
        candidate_thresholds(
            scores
        )
    )

    best = None

    for threshold in thresholds:

        if anomaly_if == "low":

            pred = (
                scores
                <= threshold
            ).astype(int)

        elif anomaly_if == "high":

            pred = (
                scores
                >= threshold
            ).astype(int)

        else:

            raise ValueError(
                anomaly_if
            )

        m = metrics(
            pred,
            y,
        )

        result = {
            "threshold":
                float(threshold),

            "metrics":
                m,
        }

        if best is None:

            best = result

        elif (
            m["f1"]
            > best["metrics"]["f1"]
        ):

            best = result

        elif (
            np.isclose(
                m["f1"],
                best["metrics"]["f1"],
            )
            and
            m["fpr"]
            < best["metrics"]["fpr"]
        ):

            best = result

    return best


# ============================================================
# TRAIN SEARCH
#
# Window is FIXED.
#
# Search only:
#     CF percentile
#     KDE bandwidth
# ============================================================

def tune_statistical(
    train_pairs,
):

    all_results = []

    total = (
        len(CF_THRESHOLDS)
        * len(KDE_BANDWIDTHS)
    )

    print(
        "\n========================================"
    )

    print(
        "PAPER KDE/CF SEARCH -- FIXED WINDOW"
    )

    print(
        "========================================"
    )

    print(
        f"W      = {WINDOW_WIDTH_FRAC}"
    )

    print(
        f"H      = {WINDOW_HEIGHT_FRAC}"
    )

    print(
        f"stride = {STRIDE_FRAC}"
    )

    print(
        f"Total configurations = {total}"
    )

    count = 0

    for cf in CF_THRESHOLDS:

        for bandwidth in KDE_BANDWIDTHS:

            count += 1

            print(
                f"\nStarting "
                f"[{count}/{total}] "
                f"CF={cf:.3f} "
                f"KDE_h={bandwidth:.1f}",
                flush=True,
            )

            (
                s_scores,
                S_scores,
                y,
            ) = scores_for(
                train_pairs,
                cf,
                bandwidth,
            )

            # -----------------------------------------------
            # Interpretation A:
            #
            # Eq. 9 text:
            # smaller s = more anomalous
            # -----------------------------------------------

            result_s = (
                find_best_threshold(
                    s_scores,
                    y,
                    anomaly_if="low",
                )
            )

            # -----------------------------------------------
            # Interpretation B:
            #
            # Eq. 10:
            # larger S_stat = more anomalous
            # -----------------------------------------------

            result_S = (
                find_best_threshold(
                    S_scores,
                    y,
                    anomaly_if="high",
                )
            )

            # These should have essentially the same ranking
            # because S_stat is monotonic in s.
            candidates = [
                (
                    "s_low",
                    result_s,
                ),
                (
                    "S_high",
                    result_S,
                ),
            ]

            for score_type, result in candidates:

                m = result[
                    "metrics"
                ]

                row = {
                    "rank": 0,
                    "cf_threshold": cf,
                    "kde_bandwidth": bandwidth,
                    "score_type": score_type,
                    "decision_threshold":
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

                print(
                    f"    {score_type:6s} "
                    f"d={result['threshold']:.8f} "
                    f"P={m['precision']:.3f} "
                    f"R={m['recall']:.3f} "
                    f"FPR={m['fpr']:.3f} "
                    f"F1={m['f1']:.3f}"
                )

            # -----------------------------------------------
            # Diagnostic distributions
            # -----------------------------------------------

            pos_s = (
                s_scores[
                    y == 1
                ]
            )

            neg_s = (
                s_scores[
                    y == 0
                ]
            )

            print(
                f"    s mean: "
                f"debris={np.mean(pos_s):.8f} "
                f"normal={np.mean(neg_s):.8f}"
            )

            print(
                f"    s median: "
                f"debris={np.median(pos_s):.8f} "
                f"normal={np.median(neg_s):.8f}"
            )

    # ========================================================
    # SORT
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
    # SAVE ALL
    # ========================================================

    fieldnames = list(
        all_results[0].keys()
    )

    all_path = (
        DATA
        / "statistical_kde_fixed_window_all.csv"
    )

    with open(
        all_path,
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

    top10 = all_results[:10]

    top_path = (
        DATA
        / "statistical_kde_fixed_window_top10.csv"
    )

    with open(
        top_path,
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

    print(
        "\n========================================"
    )

    print(
        "TOP 10 TRAIN RESULTS"
    )

    print(
        "========================================"
    )

    for row in top10:

        print(
            f"#{row['rank']:2d} "
            f"CF={row['cf_threshold']:.3f} "
            f"h={row['kde_bandwidth']:.1f} "
            f"type={row['score_type']:6s} "
            f"d={row['decision_threshold']:.8f} "
            f"P={row['precision']:.3f} "
            f"R={row['recall']:.3f} "
            f"FPR={row['fpr']:.3f} "
            f"F1={row['f1']:.3f}"
        )

    print(
        f"\nSaved all results:\n"
        f"{all_path}"
    )

    print(
        f"\nSaved top 10:\n"
        f"{top_path}"
    )

    return all_results[0]


# ============================================================
# TEST
# ============================================================

def evaluate_test(
    test_pairs,
    best,
):

    (
        s_scores,
        S_scores,
        y,
    ) = scores_for(
        test_pairs,
        best["cf_threshold"],
        best["kde_bandwidth"],
    )

    threshold = (
        best[
            "decision_threshold"
        ]
    )

    if (
        best["score_type"]
        == "s_low"
    ):

        scores = s_scores

        pred = (
            scores <= threshold
        ).astype(int)

    else:

        scores = S_scores

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
        pred,
        m,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nSTATISTICAL KDE DEBRIS SCRIPT STARTED",
        flush=True,
    )

    # ========================================================
    # ORIGINAL IMAGES
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
        f"\n  debris    = {len(pos)}"
        f"\n  no debris = {len(neg)}"
        f"\n  total     = "
        f"{len(pos) + len(neg)}"
    )

    # ========================================================
    # SAME PREPROCESSING/SPLIT
    # ========================================================

    target_size = (
        find_target_resolution(
            pos + neg
        )
    )

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
        f"\n  test  = {len(test)}"
    )

    write_split(
        train,
        test,
        "train",
        "test",
        target_size,
    )

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
    # TRAIN SEARCH
    # ========================================================

    best = tune_statistical(
        train_processed
    )

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
        f"W                 = "
        f"{WINDOW_WIDTH_FRAC}"
    )

    print(
        f"H                 = "
        f"{WINDOW_HEIGHT_FRAC}"
    )

    print(
        f"stride            = "
        f"{STRIDE_FRAC}"
    )

    print(
        f"CF                = "
        f"{best['cf_threshold']}"
    )

    print(
        f"KDE bandwidth     = "
        f"{best['kde_bandwidth']}"
    )

    print(
        f"score type        = "
        f"{best['score_type']}"
    )

    print(
        f"decision threshold= "
        f"{best['decision_threshold']:.8f}"
    )

    print(
        f"Precision         = "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall            = "
        f"{best['recall']:.4f}"
    )

    print(
        f"FPR               = "
        f"{best['fpr']:.4f}"
    )

    print(
        f"F1                = "
        f"{best['f1']:.4f}"
    )

    # ========================================================
    # TEST ONCE
    # ========================================================

    (
        test_scores,
        test_y,
        test_pred,
        test_m,
    ) = evaluate_test(
        test_processed,
        best,
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Precision = "
        f"{test_m['precision']:.4f}"
    )

    print(
        f"Recall    = "
        f"{test_m['recall']:.4f}"
    )

    print(
        f"FPR       = "
        f"{test_m['fpr']:.4f}"
    )

    print(
        f"F1        = "
        f"{test_m['f1']:.4f}"
    )

    print(
        f"TP={test_m['tp']} "
        f"FP={test_m['fp']} "
        f"TN={test_m['tn']} "
        f"FN={test_m['fn']}"
    )

    # ========================================================
    # TEST DISTRIBUTION
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
        "\nTEST SCORE DISTRIBUTION"
    )

    print(
        f"Debris: "
        f"mean={np.mean(positive_scores):.8f} "
        f"median={np.median(positive_scores):.8f}"
    )

    print(
        f"Normal: "
        f"mean={np.mean(negative_scores):.8f} "
        f"median={np.median(negative_scores):.8f}"
    )

    print(
        "\nDONE.",
        flush=True,
    )


if __name__ == "__main__":
    main()
