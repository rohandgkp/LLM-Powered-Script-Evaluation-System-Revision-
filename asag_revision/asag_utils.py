"""Shared utilities for the ASAG revision experiments (imported by every notebook)."""
import string
import numpy as np
import pandas as pd

SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------- labels
LABELS5 = ["correct", "contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]

def to3(label):  # SemEval-2013 3-way scheme
    return label if label in ("correct", "contradictory") else "incorrect"

def to2(label):  # SemEval-2013 2-way scheme
    return "correct" if label == "correct" else "incorrect"

SCHEMES = {"5way": lambda l: l, "3way": to3, "2way": to2}

# ---------------------------------------------------------------- grouped splits
def grouped_folds(groups, k=N_FOLDS, seed=SEED):
    """Assign each row a fold id so that all answers to one question share a fold."""
    uniq = np.array(sorted(pd.unique(groups)))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {g: i % k for i, g in enumerate(uniq[perm])}
    return np.array([fold_of[g] for g in groups])

def grouped_holdout(groups, frac, seed=SEED):
    """Boolean mask selecting ~frac of the questions (all their answers)."""
    uniq = np.array(sorted(pd.unique(groups)))
    n = max(1, int(round(frac * len(uniq))))
    chosen = set(np.random.default_rng(seed).choice(uniq, n, replace=False))
    return np.array([g in chosen for g in groups])

# ---------------------------------------------------------------- metrics
def _qwk(a, b, k=11):
    """Quadratic weighted kappa for integer ratings 0..k-1 (fast numpy version)."""
    O = np.zeros((k, k)); np.add.at(O, (a, b), 1)
    W = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1 - (W * O).sum() / den if den > 0 else np.nan

def _corr(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    yi = np.clip(np.floor(y + 0.5), 0, 10).astype(int)
    pi = np.clip(np.floor(p + 0.5), 0, 10).astype(int)
    ry, rp = pd.Series(y).rank().values, pd.Series(p).rank().values
    return {"pearson": _corr(y, p), "spearman": _corr(ry, rp),
            "mae": np.abs(y - p).mean(), "rmse": np.sqrt(((y - p) ** 2).mean()),
            "qwk": _qwk(yi, pi)}

def cls_metrics(y, p):
    """Accuracy, macro-F1 and weighted-F1 over the classes present in y or p."""
    y, p = np.asarray(y), np.asarray(p)
    f1s, support = [], []
    for c in np.union1d(y, p):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        support.append(np.sum(y == c))
    f1s, support = np.array(f1s), np.array(support)
    return {"acc": np.mean(y == p), "macro_f1": f1s.mean(),
            "weighted_f1": (f1s * support).sum() / support.sum()}

def bootstrap_ci(y, p, metric_fn, groups=None, n_boot=1000, seed=SEED, alpha=0.05):
    """95% CIs. If groups is given, resamples whole questions (cluster bootstrap)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx_sets = None
    else:
        groups = np.asarray(groups)
        uniq = pd.unique(groups)
        idx_sets = [np.flatnonzero(groups == g) for g in uniq]
    stats = []
    for _ in range(n_boot):
        if idx_sets is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            idx = np.concatenate([idx_sets[i] for i in rng.integers(0, len(idx_sets), len(idx_sets))])
        stats.append(metric_fn(y[idx], p[idx]))
    df = pd.DataFrame(stats)
    return {k: (np.nanpercentile(df[k], 100 * alpha / 2), np.nanpercentile(df[k], 100 * (1 - alpha / 2)))
            for k in df.columns}

def summarise(y, p, metric_fn, groups=None, folds=None, n_boot=1000):
    """One row: pooled metric, 95% CI and (if folds given) per-fold mean ± std."""
    y, p = np.asarray(y), np.asarray(p)
    pooled = metric_fn(y, p)
    ci = bootstrap_ci(y, p, metric_fn, groups, n_boot)
    row = {}
    for k, v in pooled.items():
        row[k] = v
        row[f"{k}_ci"] = f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
    if folds is not None:
        folds = np.asarray(folds)
        per = pd.DataFrame([metric_fn(y[folds == f], p[folds == f]) for f in np.unique(folds)])
        for k in per.columns:
            row[f"{k}_fold_mean"] = per[k].mean()
            row[f"{k}_fold_std"] = per[k].std(ddof=1)
    return row

# ---------------------------------------------------------------- lexical components
_STOP, _PUNCT = None, set(string.punctuation)

def content_tokens(text):
    """Lower-cased word tokens without stop words and pure-punctuation tokens."""
    global _STOP
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    return [w for w in word_tokenize(str(text).lower())
            if w not in _STOP and not all(c in _PUNCT for c in w)]

def keyword_score(key, student):
    """Eq. 1: 10 * |K ∩ S| / |K| over content-token sets."""
    K, S = set(content_tokens(key)), set(content_tokens(student))
    return 10 * len(K & S) / len(K) if K else 0.0

_SYN_CACHE = {}

def _synonyms(word):
    if word not in _SYN_CACHE:
        from nltk.corpus import wordnet as wn
        _SYN_CACHE[word] = {l.lower().replace("_", " ") for s in wn.synsets(word) for l in s.lemma_names()}
    return _SYN_CACHE[word]

def synonym_score(key, student):
    """Eq. 2: key tokens NOT matched exactly but matched by a WordNet synonym of a
    student token, normalised by |K|. Disjoint from keyword_score, so
    keyword + synonym <= 10 (no double counting)."""
    K, S = set(content_tokens(key)), set(content_tokens(student))
    if not K:
        return 0.0
    syn_S = set().union(*[_synonyms(w) for w in S]) if S else set()
    return 10 * len({k for k in K - S if k in syn_S}) / len(K)

E_MAX = 6  # errors at which the grammar score reaches 0

def grammar_score(n_errors, e_max=E_MAX):
    """Eq. 3: 10 * max(0, 1 - E / E_max)."""
    return 10 * max(0.0, 1 - n_errors / e_max)

def context_score(cos):
    """Eq. 5: cosine similarity clipped to [0, 1] and scaled to 0-10."""
    return 10 * np.clip(cos, 0, 1)

W_FIXED = {"context": 0.45, "keyword": 0.30, "synonym": 0.15, "grammar": 0.10}

def composite_fixed(df):
    """Eq. 6 with the original hand-set weights."""
    return sum(w * df[c] for c, w in W_FIXED.items())
