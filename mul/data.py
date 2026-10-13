"""
Small datasets - every one of them downloads in seconds (or ships with scikit-learn).

    digits       sklearn built-in, 1,797 handwritten 8x8 digits     notebooks 01-08 (the "lab")
    adult        Hugging Face  scikit-learn/adult-census-income      notebook 09 (Right to be Forgotten)
    olivetti     sklearn, 400 face photos of 40 people (AT&T)        notebook 10 (consent revocation)
    sms_spam     Hugging Face  ucirvine/sms_spam                     notebook 11 (poisoned data)

Downloads are cached in ./data (git-ignored).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------------------------
def load_digits():
    """1,797 8x8 grayscale digits, pixels scaled to [0, 1]. Returns X (n, 64), y (n,)."""
    from sklearn.datasets import load_digits as _ld
    d = _ld()
    return (d.data / 16.0).astype(np.float32), d.target.astype(np.int64)


def train_test(X, y, test_size=0.3, seed=0, stratify=True):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            stratify=y if stratify else None)


def forget_split(n, n_forget=None, frac=None, seed=0, candidates=None):
    """
    Pick which training examples are 'deleted'. Returns (forget_idx, retain_idx).
    `candidates` restricts the choice (e.g. only rows of one person, or one demographic group).
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(n) if candidates is None else np.asarray(candidates)
    k = n_forget if n_forget is not None else int(round(frac * len(pool)))
    f = np.sort(rng.choice(pool, size=k, replace=False))
    r = np.setdiff1d(np.arange(n), f)
    return f, r


# ---------------------------------------------------------------------------------------------
def _hf_file(repo_id, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                           cache_dir=str(DATA_DIR / "hf"))


def load_adult(n_max=None, seed=0):
    """
    UCI Adult / Census Income (CC0, via Hugging Face 'scikit-learn/adult-census-income').
    Task: does a person earn > $50K? The same dataset the book uses for its fairness chapter.

    Returns X (one-hot + standardised, float32), y (0/1), and `meta` - a DataFrame with the raw
    sensitive attributes (sex, race, age) aligned row-by-row with X, for fairness analysis.
    """
    df = pd.read_csv(_hf_file("scikit-learn/adult-census-income", "adult.csv"))
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    if n_max is not None and len(df) > n_max:
        df = df.sample(n=n_max, random_state=seed).reset_index(drop=True)
    y = (df["income"].str.strip() == ">50K").astype(np.int64).to_numpy()
    meta = df[["sex", "race", "age"]].copy()
    num = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
    cat = ["workclass", "education", "marital.status", "occupation", "relationship", "race",
           "sex", "native.country"]
    Xn = df[num].astype(float)
    Xn = (Xn - Xn.mean()) / Xn.std()
    Xc = pd.get_dummies(df[cat].astype(str), dtype=float)
    X = pd.concat([Xn, Xc], axis=1).to_numpy(np.float32)
    return X, y, meta


def load_olivetti(side=32):
    """
    Olivetti faces (AT&T Laboratories Cambridge): 40 people x 10 photos, 64x64 grayscale.
    Downsampled to `side` x `side` so a small CNN trains in seconds. Returns X (n, side*side), y.
    """
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces(data_home=str(DATA_DIR / "sklearn"), shuffle=False)
    imgs = d.images
    if side != 64:
        f = 64 // side
        imgs = imgs.reshape(len(imgs), side, f, side, f).mean(axis=(2, 4))
    return imgs.reshape(len(imgs), -1).astype(np.float32), d.target.astype(np.int64)


def load_sms_spam():
    """SMS Spam Collection (UCI, via Hugging Face 'ucirvine/sms_spam'). Returns texts, labels (1 = spam)."""
    df = pd.read_parquet(_hf_file("ucirvine/sms_spam", "plain_text/train-00000-of-00001.parquet"))
    return df["sms"].str.strip().tolist(), df["label"].to_numpy(np.int64)
