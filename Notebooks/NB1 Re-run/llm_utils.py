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

def make_prompts(df, dataset):
    tmpl = MOHLER_PROMPT if dataset == "mohler" else SCIENTS_PROMPT
    return [tmpl.format(question=q, reference=r, student=s)
            for q, r, s in zip(df.question.astype(str), df.reference_answer.astype(str),
                               df.student_answer.astype(str))]

INSTRUCT_PREFILL = {"mohler": "Score: ", "scientsbank": "Label: "}

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
