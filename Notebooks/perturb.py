"""Rule-based answer perturbations (robustness sets) and counter-example
augmentation for training. Used by Notebooks 6-10."""
import re
import numpy as np
import pandas as pd
from asag_utils import content_tokens

NEG_WORDS = re.compile(r"\b(not|no|never|none|nothing|cannot|without)\b|n't", re.I)
AUX = re.compile(r"\b(is|are|was|were|can|could|will|would|should|must|may|might|does|do|did|has|have|had)\b", re.I)
META = re.compile(r"paraphras|^\s*(here is|sure|answer:)", re.I)

def negate(a):
    """Insert 'not' after the first auxiliary/copula; otherwise prefix 'It is not the case that'."""
    m = AUX.search(a)
    if m:
        return a[:m.end()] + " not" + a[m.end():]
    return "It is not the case that " + a[0].lower() + a[1:]

def negation_form(a):
    return "auxiliary" if AUX.search(a) else "prefix"

def keyword_salad(ref, rng):
    toks = content_tokens(ref)
    return " ".join(rng.permutation(toks)) if toks else ref

def typos(a, rng, rate=0.15):
    words = a.split()
    cand = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    if not cand:
        return a
    n = max(1, int(round(rate * len(words))))
    for i in rng.choice(cand, min(n, len(cand)), replace=False):
        w = words[i]; j = rng.integers(1, len(w) - 1); op = rng.integers(3)
        words[i] = (w[:j] + w[j + 1] + w[j] + w[j + 2:] if op == 0 else
                    w[:j] + w[j + 1:] if op == 1 else w[:j] + w[j] + w[j:])
    return " ".join(words)

def clean_paraphrase(text):
    """Drop a leading meta line such as 'Here is the paraphrased answer:'."""
    lines = [l for l in str(text).strip().splitlines() if l.strip()]
    if len(lines) > 1 and META.search(lines[0]):
        lines = lines[1:]
    return " ".join(lines).strip().strip('"')

def valid_pairs(rs):
    """Text-only exclusion rules for the robustness analysis (decided without looking at scores):
    paraphrase: meta text, original <= 3 words, or paraphrase > 2x the original's length;
    typos: variant identical to the original."""
    orig = rs[rs.variant == "original"].set_index("orig_uid").student_answer.astype(str)
    o = rs.orig_uid.map(orig)
    n_o = o.str.split().str.len(); n_v = rs.student_answer.astype(str).str.split().str.len()
    bad_para = (rs.variant == "paraphrase") & (rs.student_answer.astype(str).str.contains(META) | (n_o <= 3) | (n_v > 2 * n_o))
    bad_typo = (rs.variant == "typos") & (rs.student_answer.astype(str) == o)
    return ~(bad_para | bad_typo)

# ---------------------------------------------------------------- counter-example augmentation
AUG_RATIO = 0.20        # synthetic examples added = 20% of the training answers
AUG_TYPES = ["negated", "keyword_salad", "swapped"]
AUG_TARGET = {"mohler": {"negated": 0, "keyword_salad": 0, "swapped": 0},
              "scientsbank": {"negated": "contradictory", "keyword_salad": "irrelevant", "swapped": "irrelevant"}}

def make_counterexamples(train, dataset, seed, ratio=AUG_RATIO):
    """Synthetic wrong answers built ONLY from the given training answers.
    Sources: fully correct training answers without negation words. Each source yields one
    variant; types are balanced. Swapped answers come from a different training question."""
    rng = np.random.default_rng(seed)
    good = train[(train.y >= 8) if dataset == "mohler" else (train.label == "correct")]
    good = good[~good.student_answer.astype(str).str.contains(NEG_WORDS)]
    n = min(int(round(ratio * len(train))), len(good))
    src = good.sample(n, random_state=seed)
    types = np.resize(AUG_TYPES, n); rng.shuffle(types)
    rows = []
    for (_, r), t in zip(src.iterrows(), types):
        a = str(r.student_answer)
        if t == "negated":
            text = negate(a)
        elif t == "keyword_salad":
            text = keyword_salad(str(r.reference_answer), rng)
        else:
            donors = good[good.group != r.group]
            text = str(donors.sample(1, random_state=int(rng.integers(10**9))).iloc[0].student_answer)
        rows.append({"question": r.question, "reference_answer": r.reference_answer, "student_answer": text,
                     "group": r.group, "aug_type": t, "target": AUG_TARGET[dataset][t]})
    return pd.DataFrame(rows)
