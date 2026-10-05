"""
Stage 2 (Veda): Clause classification using ONLY the LegalBERT model
fine-tuned on the original LEDGAR clause categories.

The base pretrained LegalBERT checkpoint is used only as the starting point
during fine-tuning. At runtime, this module loads only the saved model from:

    legalbert_finetuned/

There is intentionally no zero-shot, prototype, keyword, or pretrained-model
fallback. If the fine-tuned model is unavailable, classification stops with a
clear error.
"""

from __future__ import annotations

import os

from interfaces import Clause, Classification, TypeScore, Explanation
from embeddings import STOPWORDS
from xai_llm import generate_rationale


# This is still used by the uncertainty module as a baseline.
# It is NOT used to decide the clause category.
LOW_CONF_THRESHOLD = 0.60

# Folder created by train_classifier.py
FINETUNED_DIR = "legalbert_finetuned"

# Cache the model after loading it once.
_finetuned_backend = "unset"


def _explain(
    text: str,
    attn_pairs: list[tuple[str, float]],
    top_n: int = 5,
):
    """
    Extract the most salient tokens from the final LegalBERT attention layer.

    This function is used only after the fine-tuned model has successfully
    produced a classification.
    """

    ranked = sorted(
        (
            pair
            for pair in attn_pairs
            if pair[0]
            not in ("[CLS]", "[SEP]", "[PAD]")
            and not pair[0].startswith("##")
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    tokens = []

    for token, _score in ranked:

        clean_token = token.lower().strip(
            ".,;:()[]\"'"
        )

        if (
            clean_token
            and clean_token not in STOPWORDS
            and clean_token not in tokens
        ):
            tokens.append(clean_token)

        if len(tokens) >= top_n:
            break

    return tokens


class _FineTunedBackend:
    """
    Wrapper around the LegalBERT sequence-classification model trained on
    the original LEDGAR categories.
    """

    def __init__(
        self,
        tokenizer,
        model,
        torch_module,
    ):

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module

        # Transformers may load JSON keys as strings.
        # Convert them to integer keys for reliable lookup.
        self.id2label = {
            int(label_id): label_name
            for label_id, label_name
            in model.config.id2label.items()
        }

    def classify(
        self,
        text: str,
    ):

        torch = self._torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():

            output = self._model(
                **inputs,
                output_attentions=True,
            )

        # Convert the model's logits into probabilities.
        probabilities = torch.softmax(
            output.logits[0],
            dim=-1,
        ).tolist()

        confidence_by_type = {

            self.id2label[index]: probability

            for index, probability
            in enumerate(probabilities)

        }

        # XAI: average all attention heads and inspect
        # the attention from [CLS] to every input token.
        last_layer_attention = (
            output.attentions[-1][0]
        )

        cls_attention = (
            last_layer_attention
            .mean(dim=0)[0]
        )

        tokens = (
            self._tokenizer
            .convert_ids_to_tokens(
                inputs["input_ids"][0]
            )
        )

        attention_pairs = list(
            zip(
                tokens,
                cls_attention.tolist(),
            )
        )

        return (
            confidence_by_type,
            attention_pairs,
        )


def _load_finetuned_model():

    global _finetuned_backend

    if _finetuned_backend != "unset":

        return _finetuned_backend

    if not os.path.isdir(
        FINETUNED_DIR
    ):

        raise RuntimeError(

            "\nLEDGAR-fine-tuned LegalBERT "
            "model was not found.\n\n"

            "Classification is configured to use "
            "ONLY the fine-tuned model.\n\n"

            "Run this command from the project "
            "folder:\n\n"

            "    python train_classifier.py\n\n"

            "After training finishes, verify that "
            "the legalbert_finetuned/ folder was "
            "created, then restart Streamlit.\n"

        )

    try:

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                FINETUNED_DIR
            )
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                FINETUNED_DIR,
                attn_implementation="eager",
            )
        )

        model.eval()

    except Exception as error:

        raise RuntimeError(

            "\nThe LEDGAR-fine-tuned LegalBERT "
            "model folder exists, but the model "
            "could not be loaded.\n\n"

            f"Folder: {FINETUNED_DIR}/\n"

            f"Original error: "
            f"{error.__class__.__name__}: "
            f"{error}\n\n"

            "Delete the incomplete model folder "
            "if necessary and run:\n\n"

            "    python train_classifier.py\n"

        ) from error

    # Verify that this is a real classification model.
    if not hasattr(
        model,
        "classifier",
    ):

        raise RuntimeError(

            "The model in "
            f"{FINETUNED_DIR}/ does not appear "
            "to contain a sequence-classification "
            "head.\n\n"

            "Run:\n\n"

            "    python train_classifier.py\n"

        )

    _finetuned_backend = (
        _FineTunedBackend(
            tokenizer,
            model,
            torch,
        )
    )

    print(
        "\n[classifier] "
        "LEDGAR-fine-tuned LegalBERT "
        "loaded successfully."
    )

    print(
        "[classifier] "
        f"Model directory: "
        f"{FINETUNED_DIR}/"
    )

    print(
        "[classifier] "
        "Classification source: "
        "LEDGAR-fine-tuned LegalBERT"
    )

    print(
        "[classifier] "
        "Number of output categories: "
        f"{len(_finetuned_backend.id2label)}"
    )

    return _finetuned_backend


def classify_clause(
    clause: Clause,
) -> Classification:

    text = clause.text_en.strip()

    if not text:

        raise ValueError(
            "The clause text is empty and "
            "cannot be classified."
        )

    # This loads ONLY the LEDGAR-fine-tuned model.
    backend = _load_finetuned_model()

    (
        confidence_by_type,
        attention_pairs,
    ) = backend.classify(
        text
    )

    types_ranked = sorted(

        confidence_by_type,

        key=confidence_by_type.get,

        reverse=True,

    )

    predicted_type = (
        types_ranked[0]
    )

    confidence = (
        confidence_by_type[
            predicted_type
        ]
    )

    top_k = [

        TypeScore(

            type=clause_type,

            score=round(

                confidence_by_type[
                    clause_type
                ],

                4,

            ),

        )

        for clause_type
        in types_ranked[:3]

    ]

    salient_tokens = _explain(

        text,

        attention_pairs,

    )

    model_source = (
        "LEDGAR-fine-tuned "
        "LegalBERT"
    )

    template_rationale = (

        f"Predicted "
        f"'{predicted_type}' "

        f"using "
        f"{model_source} "

        f"with confidence "
        f"{confidence:.2f}."

        +

        (
            " Salient terms: "
            + ", ".join(
                salient_tokens
            )
            + "."

            if salient_tokens

            else ""

        )

    )

    llm_rationale = (
        generate_rationale(

            text,

            predicted_type,

            confidence,

            salient_tokens,

        )
    )

    rationale = (

        llm_rationale

        if llm_rationale

        else template_rationale

    )

    return Classification(

        clause_id=(
            clause.clause_id
        ),

        predicted_type=(
            predicted_type
        ),

        confidence=round(
            confidence,
            4,
        ),

        top_k=top_k,

        explanation=Explanation(

            # Changed from "LegalBERT attention"
            # because interfaces.py only accepts
            # "attention" or "shap".
            method="attention",

            salient_tokens=(
                salient_tokens
            ),

            rationale=(
                rationale
            ),

        ),

        low_conf_baseline=(

            confidence
            < LOW_CONF_THRESHOLD

        ),

    )


if __name__ == "__main__":

    demo_clause = Clause(

        clause_id="demo_1",

        index=0,

        heading="",

        text_original="",

        text_en=(

            "The Vendor shall "
            "indemnify, defend, "
            "and hold harmless "
            "the Client from "
            "all claims, losses, "
            "damages, and expenses."

        ),

        page=1,

        char_span=(0, 0),

    )

    result = classify_clause(
        demo_clause
    )

    print(
        result.model_dump_json(
            indent=2
        )
    )

