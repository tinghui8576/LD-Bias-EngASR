"""Function used to compute metrics during ASR training/evaluation."""

import torch
from torchmetrics.text import (
    WordInfoLost,
    WordErrorRate,
    CharErrorRate,
    MatchErrorRate,
)

from hyperpyyaml import load_hyperpyyaml
from whisper.normalizers import EnglishTextNormalizer


# =========================
# BASIC METRICS
# =========================
WER = WordErrorRate()
WIL = WordInfoLost()
CER = CharErrorRate()
MER = MatchErrorRate()


# =========================
# SEMDIST
# =========================
semdist_hparams = load_hyperpyyaml("""
semdist_model_name: KennethTM/MiniLM-L6-danish-encoder
semdist_model_device: cpu

semdist_stats: !new:speechbrain.utils.semdist.SemDistStats
    lm: !new:speechbrain.lobes.models.huggingface_transformers.TextEncoder
        source: !ref <semdist_model_name>
        save_path: pretrained_models/
        device: !ref <semdist_model_device>
    method: meanpool
""")


# =========================
# PATCH TOKENIZER
# =========================
# Force truncation for RoBERTa max length
text_encoder = semdist_hparams["semdist_stats"].lm

original_tokenizer = text_encoder.tokenizer


def patched_tokenizer(*args, **kwargs):
    kwargs["truncation"] = True
    kwargs["max_length"] = 512
    kwargs["padding"] = True
    return original_tokenizer(*args, **kwargs)


text_encoder.tokenizer = patched_tokenizer


# =========================
# METRICS FUNCTION
# =========================
def metrics(output, ground_truth):
    """
    Calculate:
    - WER
    - CER
    - MER
    - WIL
    - Ember
    - SemDist
    """

    pred = output
    ref = ground_truth
    normalize
    pred = [EnglishTextNormalizer()(output)]
    ref = [EnglishTextNormalizer()(ground_truth)]

    # =========================
    # SEMDIST
    # =========================
    semdist_hparams["semdist_stats"].clear()

    try:
        semdist_hparams["semdist_stats"].append(
            ids=[0],
            predict=[pred],
            target=[ref],
        )

        semdist_value = (
            semdist_hparams["semdist_stats"]
            .summarize()["semdist"]
        )

    except Exception as e:
        print(f"SemDist failed: {e}")
        semdist_value = 0.0

    # =========================
    # FINAL RESULTS
    # =========================
    return {
        "WER": round(WER(pred, ref).item(), 3),
        "CER": round(CER(pred, ref).item(), 3),
        "MER": round(MER(pred, ref).item(), 3),
        "WIL": round(WIL(pred, ref).item(), 3),
        "SemDist": round(semdist_value, 3),
    }