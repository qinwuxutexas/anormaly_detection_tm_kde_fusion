"""
Two-stage debris detector:

Stage 1:
    Directional semivariogram screening.
    Cheap spatial detector used to identify candidate windows.

Stage 2:
    KDE / CDF statistical analysis.
    Applied ONLY to candidate windows.

For bright debris on dark background:
    intensity is inverted:
        I_inv = 255 - I

    so bright debris becomes a low-intensity minority population,
    allowing the lower-CDF statistical formulation to be used.

Preprocessing:
    - same 70/30 stratified split
    - same highest-resolution candidate target
    - same resize method
    - same Gaussian blur (5, 5)
    - processed images saved to:
        yes/train
        yes/test
        no/train
        no/test
"""

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
# WINDOW GEOMETRY
#
# Keep width fixed.
# Test a few heights because we discussed that a somewhat
# taller window may better establish the normal stripe/background
# spatial structure.
# ============================================================

WINDOW_WIDTH_FRAC = 0.05

WINDOW_HEIGHT_FRACS = [
    0.25,
    0.50,
    0.75,
]

STRIDE_FRAC = 0.50


# ============================================================
# STAGE 1 -- SEMIVARIOGRAM
# ============================================================

# Multiple small lags are still very cheap.
SEMIVAR_LAGS = [
    1,
    2,
    4,
]


# Candidate threshold will be tuned from TRAIN data.
#
# We tune using quantiles rather than arbitrary absolute
# semivariance values because image contrast affects the
# absolute magnitude strongly.
#
# Lower quantile = more candidate windows = higher recall.
SEMIVAR_QUANTILES = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# STAGE 2 -- KDE
# ============================================================

# Lower CDF fraction after inversion.
CF_THRESHOLDS = [
    0.02,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
]


KDE_BANDWIDTHS = [
    1.0,
    2.0,
    4.0,
    8.0,
]


PAPER_C = 2.55


# ============================================================
# NUMERICAL CONSTANT
# ============================================================

EPS = 1e-12


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
    Candidate images:
        resize to target_size + Gaussian blur.
    """

    if target_size is not None:

        h, w = img.shape[:2]

        target_w, target_h = target_size

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
# WINDOW GENERATOR
# ============================================================

def iter_windows(
    image: np.ndarray,
    height_frac: float,
):

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
                * height_frac
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

            yield (
                x,
                y,
                win_w,
                win_h,
                patch,
            )


# ============================================================
# STAGE 1
# DIRECTIONAL SEMIVARIOGRAM
# ============================================================

def semivariance_x(
    patch: np.ndarray,
    lag: int,
) -> float:

    """
    Horizontal semivariance:

        gamma_x(h)
          = 1/2 * mean(
                [I(x+h,y)-I(x,y)]^2
            )
    """

    p = patch.astype(
        np.float32
    )

    if p.shape[1] <= lag:

        return 0.0

    diff = (
        p[:, lag:]
        - p[:, :-lag]
    )

    return float(
        0.5
        * np.mean(
            diff * diff
        )
    )


def semivariance_y(
    patch: np.ndarray,
    lag: int,
) -> float:

    """
    Vertical semivariance:

        gamma_y(h)
          = 1/2 * mean(
                [I(x,y+h)-I(x,y)]^2
            )
    """

    p = patch.astype(
        np.float32
    )

    if p.shape[0] <= lag:

        return 0.0

    diff = (
        p[lag:, :]
        - p[:-lag, :]
    )

    return float(
        0.5
        * np.mean(
            diff * diff
        )
    )


def semivariogram_features(
    patch: np.ndarray,
) -> dict:

    """
    Compute directional semivariogram features.

    Normal vertical stripe:
        gamma_y relatively small
        gamma_x relatively large

    Debris:
        vertical continuity disturbed
        gamma_y and/or gamma_y/gamma_x increase
    """

    features = {}

    gamma_x_values = []
    gamma_y_values = []
    ratio_values = []

    for lag in SEMIVAR_LAGS:

        gx = semivariance_x(
            patch,
            lag,
        )

        gy = semivariance_y(
            patch,
            lag,
        )

        ratio = (
            gy
            / (gx + EPS)
        )

        features[
            f"gamma_x_{lag}"
        ] = gx

        features[
            f"gamma_y_{lag}"
        ] = gy

        features[
            f"ratio_{lag}"
        ] = ratio

        gamma_x_values.append(
            gx
        )

        gamma_y_values.append(
            gy
        )

        ratio_values.append(
            ratio
        )

    # Aggregate across lags
    features[
        "gamma_x_mean"
    ] = float(
        np.mean(
            gamma_x_values
        )
    )

    features[
        "gamma_y_mean"
    ] = float(
        np.mean(
            gamma_y_values
        )
    )

    features[
        "ratio_mean"
    ] = float(
        np.mean(
            ratio_values
        )
    )

    features[
        "gamma_y_max"
    ] = float(
        np.max(
            gamma_y_values
        )
    )

    features[
        "ratio_max"
    ] = float(
        np.max(
            ratio_values
        )
    )

    return features


# ============================================================
# COLLECT SEMIVARIOGRAM WINDOWS
# ============================================================

def collect_spatial_windows(
    image: np.ndarray,
    height_frac: float,
):

    windows = []

    for (
        x,
        y,
        win_w,
        win_h,
        patch,
    ) in iter_windows(
        image,
        height_frac,
    ):

        f = semivariogram_features(
            patch
        )

        windows.append(
            {
                "x": x,
                "y": y,
                "w": win_w,
                "h": win_h,
                "patch": patch,

                "gamma_y":
                    f["gamma_y_mean"],

                "gamma_x":
                    f["gamma_x_mean"],

                "ratio":
                    f["ratio_mean"],

                "gamma_y_max":
                    f["gamma_y_max"],

                "ratio_max":
                    f["ratio_max"],
            }
        )

    return windows


# ============================================================
# STAGE 1 CANDIDATE GATE
# ============================================================

def candidate_windows(
    windows,
    gamma_y_threshold: float,
    ratio_threshold: float,
):

    """
    Candidate if EITHER:

        vertical semivariance is unusually high

    OR:

        vertical/horizontal anisotropy ratio is unusually high.

    Stage 1 is deliberately permissive because its job
    is HIGH RECALL, not final classification.
    """

    candidates = []

    for item in windows:

        if (
            item["gamma_y"]
            >= gamma_y_threshold
            or
            item["ratio"]
            >= ratio_threshold
        ):

            candidates.append(
                item
            )

    return candidates


# ============================================================
# KDE
# ============================================================

def kde_pdf_cdf(
    patch: np.ndarray,
    bandwidth: float,
):

    """
    Gaussian KDE approximation on intensity grid [0,255].

    Histogram is first computed, then Gaussian-smoothed
    along the intensity axis.
    """

    hist = np.bincount(
        patch.ravel(),
        minlength=256,
    ).astype(
        np.float64
    )

    total = hist.sum()

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
            np.zeros(
                256,
                dtype=np.float64,
            ),
        )

    hist /= total

    # --------------------------------------------------------
    # Gaussian KDE approximation
    # --------------------------------------------------------

    radius = max(
        1,
        int(
            np.ceil(
                4.0 * bandwidth
            )
        ),
    )

    x = np.arange(
        -radius,
        radius + 1,
        dtype=np.float64,
    )

    kernel = np.exp(
        -0.5
        * (x / bandwidth) ** 2
    )

    kernel /= kernel.sum()

    pdf = np.convolve(
        hist,
        kernel,
        mode="same",
    )

    pdf_sum = pdf.sum()

    if pdf_sum > 0:

        pdf /= pdf_sum

    cdf = np.cumsum(
        pdf
    )

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
# PAPER-STYLE LOWER-TAIL SCORE
# AFTER INTENSITY INVERSION
# ============================================================

def kde_patch_score(
    patch: np.ndarray,
    cf_threshold: float,
    bandwidth: float,
) -> float:

    """
    Bright debris -> low-intensity minority after inversion.

        I_inv = 255 - I

    Then measure the lower CDF tail.

    Larger tail width corresponds to stronger separation
    between the bright debris population and the dominant
    dark background population.
    """

    # --------------------------------------------------------
    # IMPORTANT:
    # invert intensity for bright-debris detection
    # --------------------------------------------------------

    inv = (
        255
        - patch
    ).astype(
        np.uint8
    )

    intensity, pdf, cdf = (
        kde_pdf_cdf(
            inv,
            bandwidth,
        )
    )

    # --------------------------------------------------------
    # Effective start of KDE support
    # --------------------------------------------------------

    support_floor = 1e-4

    support = np.flatnonzero(
        cdf >= support_floor
    )

    if len(support) == 0:

        return 0.0

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
    # Selected lower-CDF point
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
        intensity[
            upper_idx
        ]
    )

    upper_F = float(
        cdf[
            upper_idx
        ]
    )

    delta_I = (
        upper_I
        - start_I
    )

    delta_F = (
        upper_F
        - start_F
    )

    if (
        delta_I <= 0
        or delta_F <= 0
    ):

        return 0.0

    # --------------------------------------------------------
    # Paper-style slope
    # --------------------------------------------------------

    s = (
        delta_F
        * PAPER_C
        / delta_I
    )

    # --------------------------------------------------------
    # We want a score where HIGHER means stronger anomaly.
    #
    # Small s corresponds to a broader lower-tail interval.
    # Use reciprocal so larger = more anomalous.
    #
    # This avoids compressing everything around 0.5 with
    # the previous sigmoid.
    # --------------------------------------------------------

    anomaly_score = (
        1.0
        / (s + EPS)
    )

    return float(
        anomaly_score
    )


# ============================================================
# STAGE 2 IMAGE SCORE
# ============================================================

def kde_candidate_score(
    candidates,
    cf_threshold: float,
    bandwidth: float,
) -> float:

    """
    KDE is evaluated ONLY on Stage-1 candidate windows.

    Image score = maximum candidate KDE anomaly score.
    """

    if not candidates:

        return 0.0

    scores = []

    for item in candidates:

        score = kde_patch_score(
            item["patch"],
            cf_threshold,
            bandwidth,
        )

        scores.append(
            score
        )

    if not scores:

        return 0.0

    return float(
        np.max(
            scores
        )
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
# THRESHOLD SEARCH
# ============================================================

def candidate_thresholds(
    scores: np.ndarray,
):

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


def find_best_threshold(
    scores: np.ndarray,
    y: np.ndarray,
):

    thresholds = (
        candidate_thresholds(
            scores
        )
    )

    best = None

    for threshold in thresholds:

        # HIGH KDE anomaly score = debris
        pred = (
            scores >= threshold
        ).astype(int)

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
# PRECOMPUTE SPATIAL FEATURES
#
# Important for speed:
#
# semivariograms are calculated ONCE per image/height.
# We do not recompute them for every KDE setting.
# ============================================================

def precompute_spatial(
    pairs,
    height_frac: float,
):

    records = []

    print(
        f"\nPrecomputing semivariograms "
        f"for H={height_frac:.2f}..."
    )

    for i, (
        path,
        label,
    ) in enumerate(
        pairs,
        start=1,
    ):

        if (
            i == 1
            or i % 20 == 0
            or i == len(pairs)
        ):

            print(
                f"  image "
                f"{i}/{len(pairs)}",
                flush=True,
            )

        image = load_gray(
            path
        )

        windows = (
            collect_spatial_windows(
                image,
                height_frac,
            )
        )

        records.append(
            {
                "path": path,
                "label": label,
                "windows": windows,
            }
        )

    return records


# ============================================================
# GET GLOBAL TRAIN SEMIVARIOGRAM THRESHOLDS
# ============================================================

def spatial_thresholds_from_quantile(
    records,
    quantile: float,
):

    all_gamma_y = []
    all_ratio = []

    for record in records:

        for item in record[
            "windows"
        ]:

            all_gamma_y.append(
                item["gamma_y"]
            )

            all_ratio.append(
                item["ratio"]
            )

    if not all_gamma_y:

        raise RuntimeError(
            "No spatial windows found."
        )

    gamma_y_threshold = float(
        np.quantile(
            all_gamma_y,
            quantile,
        )
    )

    ratio_threshold = float(
        np.quantile(
            all_ratio,
            quantile,
        )
    )

    return (
        gamma_y_threshold,
        ratio_threshold,
    )


# ============================================================
# SCORE DATASET USING CASCADE
# ============================================================

def cascade_scores(
    records,
    gamma_y_threshold: float,
    ratio_threshold: float,
    cf_threshold: float,
    bandwidth: float,
):

    scores = []
    labels = []

    candidate_counts = []
    total_window_counts = []

    for record in records:

        windows = record[
            "windows"
        ]

        candidates = (
            candidate_windows(
                windows,
                gamma_y_threshold,
                ratio_threshold,
            )
        )

        score = (
            kde_candidate_score(
                candidates,
                cf_threshold,
                bandwidth,
            )
        )

        scores.append(
            score
        )

        labels.append(
            record["label"]
        )

        candidate_counts.append(
            len(candidates)
        )

        total_window_counts.append(
            len(windows)
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
        np.asarray(
            candidate_counts,
            dtype=int,
        ),
        np.asarray(
            total_window_counts,
            dtype=int,
        ),
    )


# ============================================================
# TRAIN SEARCH
# ============================================================

def tune_cascade(
    train_pairs,
):

    all_results = []

    best = None

    total = (
        len(
            WINDOW_HEIGHT_FRACS
        )
        *
        len(
            SEMIVAR_QUANTILES
        )
        *
        len(
            CF_THRESHOLDS
        )
        *
        len(
            KDE_BANDWIDTHS
        )
    )

    print(
        "\n========================================"
    )

    print(
        "SEMIVARIOGRAM -> KDE CASCADE"
    )

    print(
        "========================================"
    )

    print(
        f"Fixed W      = "
        f"{WINDOW_WIDTH_FRAC}"
    )

    print(
        f"Fixed stride = "
        f"{STRIDE_FRAC}"
    )

    print(
        f"Semivar lags = "
        f"{SEMIVAR_LAGS}"
    )

    print(
        f"Total configurations = "
        f"{total}"
    )

    count = 0

    # ========================================================
    # HEIGHT
    # ========================================================

    for height_frac in (
        WINDOW_HEIGHT_FRACS
    ):

        # ----------------------------------------------------
        # Compute spatial features only once for this H
        # ----------------------------------------------------

        records = (
            precompute_spatial(
                train_pairs,
                height_frac,
            )
        )

        # ====================================================
        # STAGE-1 QUANTILE
        # ====================================================

        for spatial_q in (
            SEMIVAR_QUANTILES
        ):

            (
                gamma_y_threshold,
                ratio_threshold,
            ) = (
                spatial_thresholds_from_quantile(
                    records,
                    spatial_q,
                )
            )

            # =================================================
            # KDE
            # =================================================

            for (
                cf_threshold,
                bandwidth,
            ) in product(
                CF_THRESHOLDS,
                KDE_BANDWIDTHS,
            ):

                count += 1

                print(
                    f"\nStarting "
                    f"[{count}/{total}] "
                    f"H={height_frac:.2f} "
                    f"spatial_q={spatial_q:.2f} "
                    f"CF={cf_threshold:.3f} "
                    f"KDE_h={bandwidth:.1f}",
                    flush=True,
                )

                (
                    scores,
                    y,
                    candidate_counts,
                    total_counts,
                ) = cascade_scores(
                    records,
                    gamma_y_threshold,
                    ratio_threshold,
                    cf_threshold,
                    bandwidth,
                )

                result = (
                    find_best_threshold(
                        scores,
                        y,
                    )
                )

                m = result[
                    "metrics"
                ]

                total_candidates = int(
                    candidate_counts.sum()
                )

                total_windows = int(
                    total_counts.sum()
                )

                candidate_fraction = (
                    total_candidates
                    / total_windows
                    if total_windows
                    else 0.0
                )

                row = {
                    "rank": 0,

                    "width_frac":
                        WINDOW_WIDTH_FRAC,

                    "height_frac":
                        height_frac,

                    "stride_frac":
                        STRIDE_FRAC,

                    "spatial_quantile":
                        spatial_q,

                    "gamma_y_threshold":
                        gamma_y_threshold,

                    "ratio_threshold":
                        ratio_threshold,

                    "cf_threshold":
                        cf_threshold,

                    "kde_bandwidth":
                        bandwidth,

                    "decision_threshold":
                        result["threshold"],

                    "candidate_fraction":
                        candidate_fraction,

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
                    f"    candidates="
                    f"{candidate_fraction:.3f} "
                    f"d={result['threshold']:.6f} "
                    f"P={m['precision']:.3f} "
                    f"R={m['recall']:.3f} "
                    f"FPR={m['fpr']:.3f} "
                    f"F1={m['f1']:.3f}",
                    flush=True,
                )

                # --------------------------------------------
                # Overall best:
                #
                # 1. F1
                # 2. lower FPR
                # 3. higher recall
                # 4. fewer KDE candidate windows
                # --------------------------------------------

                if best is None:

                    best = row

                    print(
                        "    <-- NEW BEST"
                    )

                else:

                    key_new = (
                        row["f1"],
                        -row["fpr"],
                        row["recall"],
                        -row[
                            "candidate_fraction"
                        ],
                    )

                    key_best = (
                        best["f1"],
                        -best["fpr"],
                        best["recall"],
                        -best[
                            "candidate_fraction"
                        ],
                    )

                    if (
                        key_new
                        > key_best
                    ):

                        best = row

                        print(
                            "    <-- NEW BEST"
                        )

    # ========================================================
    # SORT
    # ========================================================

    all_results.sort(
        key=lambda x: (
            -x["f1"],
            x["fpr"],
            -x["recall"],
            x["candidate_fraction"],
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

    all_path = (
        DATA
        / "semivariogram_kde_all_results.csv"
    )

    fieldnames = list(
        all_results[0].keys()
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

    top10 = (
        all_results[:10]
    )

    top_path = (
        DATA
        / "semivariogram_kde_top10.csv"
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
            f"H={row['height_frac']:.2f} "
            f"Q={row['spatial_quantile']:.2f} "
            f"CF={row['cf_threshold']:.3f} "
            f"h={row['kde_bandwidth']:.1f} "
            f"cand={row['candidate_fraction']:.3f} "
            f"d={row['decision_threshold']:.6f} "
            f"P={row['precision']:.3f} "
            f"R={row['recall']:.3f} "
            f"FPR={row['fpr']:.3f} "
            f"F1={row['f1']:.3f}"
        )

    print(
        f"\nSaved ALL results:\n"
        f"{all_path}"
    )

    print(
        f"\nSaved TOP 10:\n"
        f"{top_path}"
    )

    return all_results[0]


# ============================================================
# FINAL TEST
# ============================================================

def evaluate_test(
    test_pairs,
    best,
):

    # --------------------------------------------------------
    # Spatial features for TEST
    # --------------------------------------------------------

    records = (
        precompute_spatial(
            test_pairs,
            best["height_frac"],
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Freeze thresholds selected from TRAIN.
    # Do NOT recompute quantiles from test data.
    # --------------------------------------------------------

    (
        scores,
        y,
        candidate_counts,
        total_counts,
    ) = cascade_scores(
        records,
        best[
            "gamma_y_threshold"
        ],
        best[
            "ratio_threshold"
        ],
        best[
            "cf_threshold"
        ],
        best[
            "kde_bandwidth"
        ],
    )

    threshold = (
        best[
            "decision_threshold"
        ]
    )

    pred = (
        scores
        >= threshold
    ).astype(int)

    m = metrics(
        pred,
        y,
    )

    candidate_fraction = (
        candidate_counts.sum()
        / total_counts.sum()
        if total_counts.sum()
        else 0.0
    )

    return (
        scores,
        y,
        pred,
        m,
        candidate_fraction,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nSEMIVARIOGRAM + KDE "
        "DEBRIS SCRIPT STARTED",
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
        f"\n  debris     = {len(pos)}"
        f"\n  no debris  = {len(neg)}"
        f"\n  total      = "
        f"{len(pos) + len(neg)}"
    )

    # ========================================================
    # TARGET RESOLUTION
    # ========================================================

    target_size = (
        find_target_resolution(
            pos + neg
        )
    )

    # ========================================================
    # STRATIFIED SPLIT
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
        f"\n  test  = {len(test)}"
    )

    # ========================================================
    # RESIZE + GAUSSIAN BLUR + SAVE
    # ========================================================

    write_split(
        train,
        test,
        "train",
        "test",
        target_size,
    )

    # ========================================================
    # LOAD PROCESSED DATA
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
    # TRAIN SEARCH
    # ========================================================

    best = tune_cascade(
        train_processed
    )

    # ========================================================
    # BEST TRAIN CONFIGURATION
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
        f"W                  = "
        f"{best['width_frac']}"
    )

    print(
        f"H                  = "
        f"{best['height_frac']}"
    )

    print(
        f"stride             = "
        f"{best['stride_frac']}"
    )

    print(
        f"spatial quantile   = "
        f"{best['spatial_quantile']}"
    )

    print(
        f"gamma_y threshold  = "
        f"{best['gamma_y_threshold']:.6f}"
    )

    print(
        f"ratio threshold    = "
        f"{best['ratio_threshold']:.6f}"
    )

    print(
        f"CF                 = "
        f"{best['cf_threshold']}"
    )

    print(
        f"KDE bandwidth      = "
        f"{best['kde_bandwidth']}"
    )

    print(
        f"decision threshold = "
        f"{best['decision_threshold']:.6f}"
    )

    print(
        f"candidate fraction = "
        f"{best['candidate_fraction']:.4f}"
    )

    print(
        "\nTRAIN METRICS:"
    )

    print(
        f"Accuracy  = "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Precision = "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall    = "
        f"{best['recall']:.4f}"
    )

    print(
        f"FPR       = "
        f"{best['fpr']:.4f}"
    )

    print(
        f"F1        = "
        f"{best['f1']:.4f}"
    )

    print(
        f"TP={best['tp']} "
        f"FP={best['fp']} "
        f"TN={best['tn']} "
        f"FN={best['fn']}"
    )

    # ========================================================
    # TEST
    # ========================================================

    (
        test_scores,
        test_y,
        test_pred,
        test_m,
        test_candidate_fraction,
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
        f"Candidate fraction = "
        f"{test_candidate_fraction:.4f}"
    )

    print(
        f"Accuracy  = "
        f"{test_m['accuracy']:.4f}"
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
    # SCORE DISTRIBUTION
    # ========================================================

    pos_scores = (
        test_scores[
            test_y == 1
        ]
    )

    neg_scores = (
        test_scores[
            test_y == 0
        ]
    )

    print(
        "\nTEST SCORE DISTRIBUTION:"
    )

    if len(pos_scores):

        print(
            f"Debris: "
            f"mean={pos_scores.mean():.6f} "
            f"median={np.median(pos_scores):.6f} "
            f"min={pos_scores.min():.6f} "
            f"max={pos_scores.max():.6f}"
        )

    if len(neg_scores):

        print(
            f"No debris: "
            f"mean={neg_scores.mean():.6f} "
            f"median={np.median(neg_scores):.6f} "
            f"min={neg_scores.min():.6f} "
            f"max={neg_scores.max():.6f}"
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
