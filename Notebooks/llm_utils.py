"""Shared LLM code for all Llama notebooks: ONE prompt per dataset, used for
zero-shot inference, fine-tuning and fine-tuned inference alike."""
import numpy as np
import torch

BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"               # same base model as the original paper
INSTRUCT_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # zero-shot comparison only
MAX_LEN = 512

# ---------------------------------------------------------------- prompts
MOHLER_PROMPT = """You are grading a student's short answer. Compare the student answer with the reference answer and assign a score from 0 (completely incorrect) to 10 (completely correct). Judge meaning, not wording: paraphrases and synonyms are acceptable, but an answer that contradicts the reference answer (for example through negation) is incorrect.

### Question:
{question}

### Reference Answer:
{reference}

### Student Answer:
{student}

### Score (0-10):
"""

SCIENTS_PROMPT = """You are grading a student's short answer. Compare the student answer with the reference answer and reply with exactly one label:
correct - a complete and correct paraphrase of the reference answer;
partial - correct but missing information from the reference answer;
contradictory - contradicts the reference answer;
irrelevant - on topic but does not answer the question;
non-domain - not about the subject at all (e.g. "I don't know").

### Question:
{question}

### Reference Answer:
{reference}

### Student Answer:
{student}

### Label:
"""

SCORE_TARGETS = [str(i) for i in range(11)]                      # Mohler targets "0".."10"
LABEL_WORDS = {"correct": "correct", "partially_correct_incomplete": "partial",
               "contradictory": "contradictory", "irrelevant": "irrelevant",
               "non_domain": "non-domain"}                        # dataset label -> target word

LABEL_ORDER = list(LABEL_WORDS)          # candidate order = LABELS5 order

def label_candidates(chat):
    return [(" " if chat else "") + LABEL_WORDS[l] for l in LABEL_ORDER]

def labels_from_logprobs(lp):
    """SciEntsBank: renormalised probabilities over the 5 labels, the probability mass on
    valid labels, and predictions for each scheme. 3-way / 2-way predictions sum the
    5-way probabilities within each coarse class before taking the argmax."""
    p = np.exp(lp); mass = p.sum(1); q = p / mass[:, None]
    L = np.array(LABEL_ORDER)
    i_c, i_x = LABEL_ORDER.index("correct"), LABEL_ORDER.index("contradictory")
    p3 = np.stack([q[:, i_c], q[:, i_x], 1 - q[:, i_c] - q[:, i_x]], 1)
    preds = {"5way": L[q.argmax(1)],
             "3way": np.array(["correct", "contradictory", "incorrect"])[p3.argmax(1)],
             "2way": np.where(q[:, i_c] >= 0.5, "correct", "incorrect")}
    return q, mass, preds

def make_prompts(df, dataset):
    tmpl = MOHLER_PROMPT if dataset == "mohler" else SCIENTS_PROMPT
    return [tmpl.format(question=q, reference=r, student=s)
            for q, r, s in zip(df.question.astype(str), df.reference_answer.astype(str),
                               df.student_answer.astype(str))]

INSTRUCT_PREFILL = {"mohler": "Score: ", "scientsbank": "Label:"}
# Llama-3 tokenisation: digits never carry a leading space (so the score prefill ends in
# a space), whereas words do (so the label prefill does not, and label candidates for the
# chat format start with a space). The sanity checks in each notebook assert this.

def to_chat(tok, prompts, prefill=""):
    """Same prompt text, wrapped in the Instruct chat template. `prefill` starts the
    assistant turn so the next token is the answer (without it the Instruct model
    begins with preamble such as '**' or 'I would', and puts ~0 mass on the answer)."""
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True) + prefill for p in prompts]

# ---------------------------------------------------------------- tokens
def candidate_ids(tok, words):
    """First token of each candidate answer; asserts they are distinct.
    For scores 0-10 each number is a single Llama-3 token (asserted)."""
    ids = [tok(w, add_special_tokens=False)["input_ids"] for w in words]
    firsts = [i[0] for i in ids]
    assert len(set(firsts)) == len(firsts), f"candidate first tokens collide: {dict(zip(words, ids))}"
    return firsts, ids

# ---------------------------------------------------------------- model
def load_hf_4bit(name, adapter=None):
    """Plain transformers + bitsandbytes (fp16 compute for T4). Optional LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"      # never cut the end of a prompt
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda:0")
    # The Unsloth checkpoints store bf16 as the 4-bit compute dtype; the T4 has no native
    # bf16, so force fp16 compute in every 4-bit layer.
    import bitsandbytes as bnb
    layers = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    print(f"{len(layers)} 4-bit layers; compute dtype before: {sorted({str(m.compute_dtype) for m in layers})} -> torch.float16")
    for m in layers:
        m.compute_dtype = torch.float16
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok

@torch.no_grad()
def candidate_logprobs(model, tok, texts, cand_ids, batch_size=8, add_special_tokens=True):
    """Log-probabilities of each candidate as the NEXT token after each text.
    Returns (n_texts, n_candidates) log-probs over the FULL vocabulary."""
    order = np.argsort([len(t) for t in texts])[::-1]   # length-sorted batches = less padding
    out = np.zeros((len(texts), len(cand_ids)), np.float32)
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  add_special_tokens=add_special_tokens, truncation=True, max_length=MAX_LEN).to(model.device)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)       # correct positions under left padding
        logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()  # last position only
        out[idx] = torch.log_softmax(logits, -1)[:, cand_ids].cpu().numpy()
    return out

def score_from_logprobs(lp):
    """Mohler: probability-weighted expected score, argmax score, and probability
    mass the model puts on valid answers (format compliance)."""
    p = np.exp(lp)
    mass = p.sum(1)
    q = p / mass[:, None]
    return q @ np.arange(lp.shape[1]), q.argmax(1), mass

# ---------------------------------------------------------------- LoRA fine-tuning
LORA = dict(r=16, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
TRAIN = dict(epochs=2, lr=2e-4, batch_size=8, weight_decay=0.01, warmup_frac=0.05, max_grad_norm=1.0,
             evals_per_epoch=2)

def build_examples(tok, prompts, targets):
    """(prompt ids, target ids + EOS). Prompt tokenised exactly as at inference, so the
    first target token is the token scored by candidate_logprobs."""
    ex = []
    for p, t in zip(prompts, targets):
        p_ids = tok(p, add_special_tokens=True)["input_ids"]
        t_ids = tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        p_ids = p_ids[-(MAX_LEN - len(t_ids)):]                   # left-truncate if ever too long
        ex.append((p_ids, t_ids))
    return ex

def collate_left(examples, pad_id, device):
    """Left-padded batch. All targets sit at the END of each row, so only the last
    K = max target length + 1 positions need logits (big memory/time saving)."""
    L = max(len(p) + len(t) for p, t in examples)
    K = max(len(t) for _, t in examples) + 1
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    lab = torch.full((len(examples), K - 1), -100, dtype=torch.long)
    for i, (p, t) in enumerate(examples):
        seq = p + t
        ids[i, L - len(seq):] = torch.tensor(seq)
        att[i, L - len(seq):] = 1
        lab[i, K - 1 - len(t):] = torch.tensor(t)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), att.to(device), pos.to(device), lab.to(device), K

def target_loss(model, batch):
    ids, att, pos, lab, K = batch
    logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K).logits
    # logits at positions L-K..L-2 predict tokens at L-K+1..L-1
    return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                             lab.reshape(-1), ignore_index=-100)

def add_lora(model, seed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(bias="none", task_type="CAUSAL_LM", **LORA))
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable), "LoRA weights must be fp32 for fp16 AMP"
    return model, sum(p.numel() for p in trainable)

def _micro_batches(lengths, bs, rng):
    """Shuffle, then sort within windows of 16 batches by length (less padding), then shuffle batches."""
    perm = rng.permutation(len(lengths))
    win = bs * 16
    batches = []
    for s in range(0, len(perm), win):
        w = sorted(perm[s:s + win], key=lambda i: lengths[i])
        batches += [w[i:i + bs] for i in range(0, len(w), bs)]
    return [batches[i] for i in rng.permutation(len(batches))]

def train_lora(model, tok, examples, eval_fn, seed, log=print, ckpt_path=None, **overrides):
    """Custom fp16-AMP loop. Evaluates eval_fn() `evals_per_epoch` times per epoch and
    keeps the LoRA weights with the lowest validation NLL. Returns (history, best_eval).
    If ckpt_path is given, the full training state is saved there at every evaluation and
    training resumes from it after a disconnect (identical batch order: the whole schedule
    is drawn up front from the seed)."""
    import os, time, math
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_linear_schedule_with_warmup
    cfg = {**TRAIN, **overrides}
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(examples) / cfg["batch_size"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_frac"] * total), total)
    scaler = torch.amp.GradScaler("cuda")
    eval_at = {round(steps_per_epoch * (e + f / cfg["evals_per_epoch"]))
               for e in range(cfg["epochs"]) for f in range(1, cfg["evals_per_epoch"] + 1)}
    lengths = [len(p) + len(t) for p, t in examples]
    schedule = [b for _ in range(cfg["epochs"]) for b in _micro_batches(lengths, cfg["batch_size"], rng)]
    assert len(schedule) == total
    hist = {"train": [], "eval": []}
    best, best_state, step, elapsed, run = None, None, 0, 0.0, []
    if ckpt_path and os.path.exists(f"{ckpt_path}/state.pt"):
        st = torch.load(f"{ckpt_path}/state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        hist, best, best_state, step, elapsed = st["hist"], st["best"], st["best_state"], st["step"], st["elapsed"]
        log(f"resumed from step {step}/{total}")
    t0 = time.time() - elapsed
    while step < total:
        model.train()
        batch = collate_left([examples[i] for i in schedule[step]], tok.pad_token_id, model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = target_loss(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        step += 1; run.append(loss.item())
        if step % 10 == 0:
            hist["train"].append({"step": step, "loss": float(np.mean(run)), "sec": time.time() - t0}); run = []
            if step % 50 == 0:
                log(f"step {step}/{total}  loss {hist['train'][-1]['loss']:.4f}  {(time.time() - t0) / step:.2f}s/step")
        if step in eval_at:
            model.eval()
            ev = {"step": step, "epoch": step / steps_per_epoch, **eval_fn()}
            hist["eval"].append(ev); log(f"eval {ev}")
            if best is None or ev["val_nll"] < best["val_nll"]:
                best = ev
                best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            if ckpt_path and step < total:
                os.makedirs(ckpt_path, exist_ok=True)
                cur = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
                torch.save({"lora": cur, "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "hist": hist, "best": best, "best_state": best_state,
                            "step": step, "elapsed": time.time() - t0}, f"{ckpt_path}/state.tmp")
                os.replace(f"{ckpt_path}/state.tmp", f"{ckpt_path}/state.pt")
                log(f"checkpoint saved at step {step}")
    set_peft_model_state_dict(model, best_state)
    model.eval()
    hist.update(total_steps=total, sec_per_step=(time.time() - t0) / total, config=cfg, lora=LORA)
    return hist, best
