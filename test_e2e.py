"""
End-to-end smoke test for transformers 4.53.0 migration.

Calls the SAME existing functions used by runner.sh:
  1) load_model_runner  → HFModel (model_utils.py)
  2) model_runner.predict_classification  (NLU — main_nlu_prompt_batch.py path)
  3) model_runner.predict_generation      (NLG — main_nlg_prompt_batch.py path)
  4) model_runner.predict_generation with ChatMessage (LLM Judge generate path)

Uses sshleifer/tiny-gpt2 on CPU.  HFModel is monkey-patched so that
flash_attention_2 / bfloat16 / .to("cuda") are replaced with CPU equivalents,
but every other code path (rope_scaling, _get_terminator, logprobs, generate)
runs unchanged.

Usage:
    pip install -r requirements.txt
    python test_e2e.py
"""

import sys, os, textwrap
from unittest.mock import patch

# ── make evaluation/ importable ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
results = []

TINY_MODEL = "sshleifer/tiny-gpt2"


def run_test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  {FAIL}  {name}: {e}")
        results.append((name, False, str(e)))


# ═══════════════════════════════════════════════════════════
# 0. CPU monkey-patches for HFModel
#    (swap flash_attn / bfloat16 / .to("cuda") → CPU-safe)
# ═══════════════════════════════════════════════════════════
import model_utils
from model_utils import (
    HFModel,
    ChatMessage,
    _get_and_verify_max_len,
    load_model_runner,
    MAX_GENERATION_LENGTH,
)

_orig_hf_init = HFModel.__init__


def _cpu_hf_init(self, model_name_or_path, compile=False):
    """Same as HFModel.__init__ but loads on CPU without flash_attention_2."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    tokenizer.padding_side = "left"
    model_max_length = min(_get_and_verify_max_len(model.config), 8192)
    self.max_generation_length = MAX_GENERATION_LENGTH
    self.model_max_length = model_max_length
    print("model_max_length", self.model_max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.bos_token
            if tokenizer.bos_token is not None
            else tokenizer.eos_token
        )
    # tiny-gpt2 has no chat_template; real eval models (Llama, Qwen, etc.)
    # always have one. Set a minimal template so predict_generation works.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
    model.eval()
    self.model_name = model_name_or_path
    self.model = model
    self.tokenizer = tokenizer


_orig_get_logprobs = HFModel._get_logprobs


@torch.inference_mode()
def _cpu_get_logprobs(self, model, model_name, tokenizer, inputs, label_ids=None, label_attn=None):
    """Same as HFModel._get_logprobs but .to('cpu') instead of .to('cuda')."""
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=self.model_max_length - self.max_generation_length,
    )  # stays on CPU — no .to("cuda")
    if "sea-lion" in model_name and "token_type_ids" in enc.keys():
        del enc["token_type_ids"]
    logits = model(**enc).logits
    output_ids = enc["input_ids"][:, 1:]
    logprobs = torch.gather(
        F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)
    ).squeeze(dim=-1)
    logprobs[enc["attention_mask"][:, :-1] == 0] = 0
    return logprobs.sum(dim=1).cpu()


# Apply patches
HFModel.__init__ = _cpu_hf_init
HFModel._get_logprobs = _cpu_get_logprobs


# ═══════════════════════════════════════════════════════════
# 1. load_model_runner  (same call as all 3 eval scripts)
# ═══════════════════════════════════════════════════════════
print("\n=== 1. load_model_runner ===")

model_runner = None


def test_load_model_runner():
    global model_runner
    set_seed(42)
    model_runner = load_model_runner(TINY_MODEL)
    assert isinstance(model_runner, HFModel), f"expected HFModel, got {type(model_runner)}"
    assert model_runner.model is not None
    assert model_runner.tokenizer is not None
    assert model_runner.model_max_length > 0
    print(f"    model_max_length = {model_runner.model_max_length}")

run_test(f"load_model_runner('{TINY_MODEL}')", test_load_model_runner)


# ═══════════════════════════════════════════════════════════
# 2. _get_and_verify_max_len  (rope_scaling migration)
# ═══════════════════════════════════════════════════════════
print("\n=== 2. _get_and_verify_max_len ===")


class FakeConfig:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_rope_default_no_factor():
    """rope_type='default' without factor (new in 4.46+) must not crash."""
    cfg = FakeConfig(max_position_embeddings=8192,
                     rope_scaling={"rope_type": "default"})
    assert _get_and_verify_max_len(cfg) == 8192

run_test("rope_type='default' (no factor) — no crash", test_rope_default_no_factor)


def test_rope_linear_factor():
    cfg = FakeConfig(max_position_embeddings=4096,
                     rope_scaling={"type": "linear", "factor": 2.0})
    assert _get_and_verify_max_len(cfg) == 8192

run_test("rope_type='linear' factor=2 → 8192", test_rope_linear_factor)


def test_rope_unknown_no_factor():
    cfg = FakeConfig(max_position_embeddings=4096,
                     rope_scaling={"type": "some_future_type"})
    assert _get_and_verify_max_len(cfg) == 4096

run_test("unknown rope type without factor — graceful", test_rope_unknown_no_factor)


# ═══════════════════════════════════════════════════════════
# 3. _get_terminator  (unk filtering migration)
# ═══════════════════════════════════════════════════════════
print("\n=== 3. _get_terminator ===")


def test_terminator():
    terms = model_runner._get_terminator()
    tok = model_runner.tokenizer
    assert tok.eos_token_id in terms
    # GPT-2 has no chat tokens → only eos should be present
    assert len(terms) == 1, f"expected [eos], got {terms}"

run_test("_get_terminator — only eos, no phantom unk tokens", test_terminator)


# ═══════════════════════════════════════════════════════════
# 4. predict_classification  (NLU path — like main_nlu_prompt_batch.py)
#    runner.sh: python evaluation/main_nlu_prompt_batch.py tha $MODEL 4
# ═══════════════════════════════════════════════════════════
print("\n=== 4. predict_classification (NLU path) ===")

# Use a real Thai prompt template from prompt_utils (sentiment analysis)
from prompt_utils import get_prompt

TASK_PROMPTS = get_prompt("tha")


def test_nlu_single():
    """Single sample, SA task — mirrors main_nlu_prompt_batch.py logic."""
    prompt_template = TASK_PROMPTS["SA"][0]  # sentiment analysis prompt
    labels = ["positive", "negative", "neutral"]
    sample_text = "วันนี้อากาศดีมาก ฉันมีความสุข"  # "The weather is great, I'm happy"

    prompt = prompt_template.replace("[INPUT]", sample_text)
    if "[OPTIONS]" in prompt:
        prompt = prompt.replace("[OPTIONS]", ", ".join(labels))

    prompts = [prompt]
    with torch.inference_mode():
        preds = model_runner.predict_classification(prompts, labels)
    assert len(preds) == 1
    assert preds[0] in range(len(labels)), f"pred index {preds[0]} out of range"

run_test("NLU: predict_classification single (SA)", test_nlu_single)


def test_nlu_batch():
    """Batch of 4 — same as BATCH_SIZE=4 in runner.sh."""
    prompt_template = TASK_PROMPTS["SA"][0]
    labels = ["positive", "negative", "neutral"]
    texts = [
        "วันนี้อากาศดีมาก ฉันมีความสุข",
        "อาหารร้านนี้แย่มาก ไม่อร่อยเลย",
        "ฉันไม่แน่ใจว่ารู้สึกอย่างไร",
        "หนังเรื่องนี้สนุกมาก แนะนำเลย",
    ]
    prompts = []
    for text in texts:
        p = prompt_template.replace("[INPUT]", text)
        if "[OPTIONS]" in p:
            p = p.replace("[OPTIONS]", ", ".join(labels))
        prompts.append(p)

    with torch.inference_mode():
        preds = model_runner.predict_classification(prompts, labels)
    assert len(preds) == 4
    assert all(p in range(len(labels)) for p in preds), f"bad preds: {preds}"

run_test("NLU: predict_classification batch=4 (SA)", test_nlu_batch)


def test_nlu_qa():
    """QA task — mirrors the QA schema in main_nlu_prompt_batch.py."""
    import string
    prompt_template = TASK_PROMPTS["QA"][0]
    choices = ["กรุงเทพ", "เชียงใหม่", "ภูเก็ต"]
    choices_str = "\n".join(f"{string.ascii_lowercase[i]}. {c}" for i, c in enumerate(choices))
    labels = [string.ascii_lowercase[i] for i in range(len(choices))]

    prompt = prompt_template
    if "[CONTEXT]" in prompt:
        prompt = prompt.replace("[CONTEXT]", "เมืองหลวงของประเทศไทยคือกรุงเทพมหานคร")
    prompt = prompt.replace("[QUESTION]", "เมืองหลวงของประเทศไทยคือเมืองอะไร?")
    prompt = prompt.replace("[ANSWER_CHOICES]", choices_str)

    with torch.inference_mode():
        preds = model_runner.predict_classification([prompt], labels)
    assert len(preds) == 1
    assert preds[0] in range(len(labels))

run_test("NLU: predict_classification (QA)", test_nlu_qa)


# ═══════════════════════════════════════════════════════════
# 5. predict_generation with str prompts  (NLG path)
#    runner.sh: python evaluation/main_nlg_prompt_batch.py tha $MODEL 0 4
# ═══════════════════════════════════════════════════════════
print("\n=== 5. predict_generation — str prompts (NLG path) ===")


def test_nlg_single():
    """Single summarization prompt — mirrors main_nlg_prompt_batch.py."""
    prompt_template = TASK_PROMPTS["SUM"][0]
    text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย มีประชากรมากกว่า 10 ล้านคน"
    prompt = prompt_template.replace("[INPUT]", text)

    with torch.inference_mode():
        preds = model_runner.predict_generation([prompt])
    assert len(preds) == 1
    assert isinstance(preds[0], str)

run_test("NLG: predict_generation single (SUM)", test_nlg_single)


def test_nlg_batch():
    """Batch of 4 — translation prompts from NLG pipeline."""
    prompt_template = TASK_PROMPTS["MT"][0]
    from prompt_utils import get_lang_name
    texts = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I like Thai food very much.",
        "Bangkok is the capital of Thailand.",
    ]
    prompts = []
    for text in texts:
        p = prompt_template.replace("[INPUT]", text)
        p = p.replace("[SOURCE]", get_lang_name("tha", "eng"))
        p = p.replace("[TARGET]", get_lang_name("tha", "tha"))
        prompts.append(p)

    with torch.inference_mode():
        preds = model_runner.predict_generation(prompts)
    assert len(preds) == 4
    assert all(isinstance(p, str) for p in preds)

run_test("NLG: predict_generation batch=4 (MT)", test_nlg_batch)


# ═══════════════════════════════════════════════════════════
# 6. predict_generation with ChatMessage (LLM Judge path)
#    runner.sh: python evaluation/main_llm_judge_batch.py $MODEL ...
#    LLMJudgeEvalHandler.generate() calls predict_generation
#    with List[List[ChatMessage]].
# ═══════════════════════════════════════════════════════════
print("\n=== 6. predict_generation — ChatMessage (LLM Judge path) ===")


def test_judge_generate_single_turn():
    """Single-turn conversation — mirrors LLMJudgeEvalHandler.generate()."""
    conversations = [
        [ChatMessage(role="user", content="อธิบายว่า AI คืออะไร")],
    ]
    with torch.inference_mode():
        preds = model_runner.predict_generation(conversations)
    assert len(preds) == 1
    assert isinstance(preds[0], str)

run_test("Judge: predict_generation single-turn ChatMessage", test_judge_generate_single_turn)


def test_judge_generate_multi_turn():
    """Multi-turn (2 turns) — same path as mt-bench multi-turn evaluation."""
    conversations = [
        [
            ChatMessage(role="user", content="อธิบายว่า AI คืออะไร"),
            ChatMessage(role="assistant", content="AI คือปัญญาประดิษฐ์"),
            ChatMessage(role="user", content="ยกตัวอย่างการใช้งาน AI"),
        ],
        [
            ChatMessage(role="user", content="สวัสดี คุณชื่ออะไร"),
        ],
    ]
    with torch.inference_mode():
        preds = model_runner.predict_generation(conversations)
    assert len(preds) == 2
    assert all(isinstance(p, str) for p in preds)

run_test("Judge: predict_generation multi-turn ChatMessage batch=2", test_judge_generate_multi_turn)


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"  Total: {total}  |  {PASS}: {passed}  |  {FAIL}: {failed}")

if failed:
    print("\n  Failed:")
    for name, ok, err in results:
        if not ok:
            print(f"    - {name}: {err}")
    print()
    sys.exit(1)
else:
    print(f"\n  All {total} tests passed! transformers 4.53.0 migration OK 🎉\n")
    sys.exit(0)
