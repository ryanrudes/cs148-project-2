from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Hashable, Iterable, Union

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from digit_classifier.foundation_models import (
    FoundationModelConfig,
    FoundationModelFamily,
    compute_foundation_model_features,
    get_foundation_model,
    load_foundation_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)
console = Console()

DIGIT_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")

VIEWPOINT_OPTIONS = [
    "a top-down",
    "an overhead",
    "a flat lay",
    "a close-up overhead",
    "a bird's-eye",
]

MEDIUM_OPTIONS = [
    "image of",
    "photo of",
    "photograph of",
    "picture of",
    "view of",
]

OBJECT_PHRASE_OPTIONS = [
    "objects forming",
    "objects arranged as",
    "objects forming the shape of",
    "real-world objects forming",
    "everyday objects forming",
    "non-handwritten objects forming",
]

TARGET_PHRASE_OPTIONS = [
    "the number {digit}",
    "the digit {digit}",
]

REALISM_OPTIONS = [
    "",
    "in a real-world scene",
    "made from objects",
    "formed by objects",
    "formed by real-world objects",
    "formed by everyday objects",
]

DEFAULT_SEED_PROMPTS = [
    "a top-down image of objects forming the number {digit}",
    "a top-down photo of objects forming the number {digit}",
    "an overhead image of the number {digit} formed by objects",
    "an overhead photo of the number {digit} formed by objects",
    "an overhead image of the number {digit} formed by everyday objects",
    "a top-down image of the number {digit} formed by real-world objects",
    "a non-handwritten digit {digit} formed by objects",
    "a close-up photograph of the number {digit} formed by objects",
    "this photograph shows the number {digit} formed by objects",
    "an overhead image of objects forming the number {digit}",
]


@dataclass(frozen=True)
class PromptCandidate:
    template: str
    use_word: bool


@dataclass(frozen=True)
class PromptEnsembleCandidate:
    members: tuple[PromptCandidate, ...]


@dataclass(frozen=True)
class SlotPromptCandidate:
    viewpoint: str
    medium: str
    object_phrase: str
    target_phrase: str
    realism_phrase: str
    use_word: bool


Candidate = Union[PromptCandidate, PromptEnsembleCandidate, SlotPromptCandidate]


@dataclass
class CandidateScore:
    candidate: Candidate
    mean_inner_score: float
    outer_mean_score: float
    outer_std_score: float
    selected_count: int


@dataclass
class GenerationSummary:
    generation: int
    best: CandidateScore
    mean_outer_score: float
    population_size: int
    mode: str
    phase: str
    search_stats: "SearchStats"
    population_stats: "PopulationStats"


@dataclass
class OuterFoldResult:
    fold_index: int
    candidate: Candidate
    inner_score: float
    test_score: float


@dataclass
class EvolutionConfig:
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    population_size: int = 24
    generations: int = 8
    elite_size: int = 6
    children_per_generation: int = 18
    outer_folds: int = 5
    inner_folds: int = 3
    mutation_rate: float = 0.45
    max_new_tokens: int = 220
    temperature: float = 0.9
    top_p: float = 0.95
    max_attempts_per_batch: int = 3
    batch_mutation_size: int = 6
    random_seed: int = 0
    cache_dir: str = "cache/prompt_evolution"
    use_mps: bool = True
    digit_only: bool = True
    text_batch_size: int = 16
    two_phase: bool = False
    ensemble: bool = False
    slot_based: bool = False
    phase1_generations: int = 4
    phase1_population_size: int = 24
    phase1_outer_folds: int = 3
    phase1_inner_folds: int = 2
    phase2_generations: int = 6
    phase2_population_size: int = 16
    phase2_outer_folds: int = 5
    phase2_inner_folds: int = 3
    phase2_seed_top_k: int = 12
    ensemble_size: int = 3
    ensemble_seed_pool_size: int = 12
    slot_mutation_rate: float = 0.35
    phase2_mode: str = "same"
    llm_mutation_batches_per_generation: int = 4
    max_llm_candidates_per_generation: int = 24
    max_novel_candidates_kept: int = 12
    near_duplicate_jaccard_threshold: float = 0.85


@dataclass
class SearchStats:
    llm_batches_attempted: int = 0
    llm_raw_templates: int = 0
    invalid_templates: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    accepted_novel_templates: int = 0
    lexical_mutations_used: int = 0
    crossover_children_used: int = 0


@dataclass
class PopulationStats:
    population_size: int
    diversity: int
    elite_count: int
    children_count: int
    llm_generated_children: int = 0
    lexical_mutation_children: int = 0
    crossover_children: int = 0
    recombined_children: int = 0
    filler_candidates: int = 0


@dataclass
class OffspringBuildResult:
    children: list[Candidate]
    search_stats: SearchStats
    llm_generated_children: int = 0
    lexical_mutation_children: int = 0
    crossover_children: int = 0
    recombined_children: int = 0


class PromptLLM:
    def __init__(self, model_name: str, device: str, max_new_tokens: int, temperature: float, top_p: float):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        log.info("Loading prompt-evolution LLM from %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = get_llm_torch_dtype(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        self.model.to(device)
        self.model.eval()

    def generate_templates(self, system_prompt: str, user_prompt: str, rng: random.Random) -> list[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        templates = extract_prompt_templates(text)
        rng.shuffle(templates)
        return templates


def extract_prompt_templates(text: str) -> list[str]:
    templates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^[-*\d\.)\s]+", "", stripped)
        stripped = stripped.strip().strip('"').strip("'")
        if len(stripped) < 10:
            continue
        normalized = canonical_template(stripped)
        if not has_only_digit_placeholder(normalized):
            continue
        if normalized not in templates:
            templates.append(normalized)
    return templates


def canonical_template(template: str) -> str:
    return re.sub(r"\s+", " ", template.strip())


def has_only_digit_placeholder(template: str) -> bool:
    placeholders = PLACEHOLDER_RE.findall(template)
    return placeholders.count("{digit}") == 1 and len(placeholders) == 1


def instantiate_template(template: str, digit: str) -> str:
    return template.replace("{digit}", digit)


def get_best_device(use_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if use_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_llm_torch_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def get_model_move_dtype(device: str) -> torch.dtype | None:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return None
    return None


def build_candidate(template: str, use_word: bool) -> PromptCandidate | None:
    normalized = canonical_template(template)
    if not has_only_digit_placeholder(normalized):
        return None
    return PromptCandidate(template=normalized, use_word=use_word)


def build_ensemble_candidate(members: Iterable[PromptCandidate]) -> PromptEnsembleCandidate:
    sorted_members = tuple(sorted(set(members), key=lambda m: (m.template, m.use_word)))
    return PromptEnsembleCandidate(members=sorted_members)


def build_slot_candidate(
    viewpoint: str,
    medium: str,
    object_phrase: str,
    target_phrase: str,
    realism_phrase: str,
    use_word: bool,
) -> SlotPromptCandidate | None:
    candidate = SlotPromptCandidate(
        viewpoint=viewpoint,
        medium=medium,
        object_phrase=object_phrase,
        target_phrase=target_phrase,
        realism_phrase=realism_phrase,
        use_word=use_word,
    )
    template = slot_candidate_template(candidate)
    if not has_only_digit_placeholder(template):
        return None
    return candidate


def candidate_prompts(candidate: PromptCandidate) -> list[str]:
    if candidate.use_word:
        return [instantiate_template(candidate.template, DIGIT_WORDS[i]) for i in range(10)]
    return [instantiate_template(candidate.template, str(i)) for i in range(10)]


def candidate_template(candidate: PromptCandidate) -> str:
    return candidate.template


def slot_candidate_template(candidate: SlotPromptCandidate) -> str:
    parts = [
        candidate.viewpoint,
        candidate.medium,
        candidate.object_phrase,
        candidate.target_phrase,
        candidate.realism_phrase,
    ]
    text = " ".join(part for part in parts if part)
    return canonical_template(text)


def render_candidate_prompts(candidate: PromptCandidate | SlotPromptCandidate) -> list[str]:
    if isinstance(candidate, PromptCandidate):
        return candidate_prompts(candidate)
    template = slot_candidate_template(candidate)
    if candidate.use_word:
        return [instantiate_template(template, DIGIT_WORDS[i]) for i in range(10)]
    return [instantiate_template(template, str(i)) for i in range(10)]


def render_ensemble_member_prompts(candidate: PromptEnsembleCandidate) -> list[list[str]]:
    return [candidate_prompts(member) for member in candidate.members]


def format_candidate_description(candidate: Candidate) -> str:
    if isinstance(candidate, PromptCandidate):
        return candidate.template
    if isinstance(candidate, SlotPromptCandidate):
        return slot_candidate_template(candidate)
    if isinstance(candidate, PromptEnsembleCandidate):
        return " | ".join(f"{i+1}) {m.template}" for i, m in enumerate(candidate.members))
    raise ValueError(f"Unknown candidate type: {type(candidate)}")


def format_candidate_kind(candidate: Candidate) -> str:
    if isinstance(candidate, PromptEnsembleCandidate):
        return "ensemble"
    if isinstance(candidate, PromptCandidate):
        return "word" if candidate.use_word else "digit"
    if isinstance(candidate, SlotPromptCandidate):
        return "word" if candidate.use_word else "digit"
    raise ValueError(f"Unknown candidate type: {type(candidate)}")


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    if isinstance(candidate, PromptCandidate):
        return {"template": candidate.template, "use_word": candidate.use_word}
    if isinstance(candidate, PromptEnsembleCandidate):
        return {
            "members": [
                {"template": m.template, "use_word": m.use_word}
                for m in candidate.members
            ]
        }
    if isinstance(candidate, SlotPromptCandidate):
        return {
            "viewpoint": candidate.viewpoint,
            "medium": candidate.medium,
            "object_phrase": candidate.object_phrase,
            "target_phrase": candidate.target_phrase,
            "realism_phrase": candidate.realism_phrase,
            "use_word": candidate.use_word,
            "rendered_template": slot_candidate_template(candidate),
        }
    raise ValueError(f"Unknown candidate type: {type(candidate)}")


def candidate_equality_key(candidate: Candidate) -> Hashable:
    if isinstance(candidate, PromptCandidate):
        return ("single", candidate.template, candidate.use_word)
    if isinstance(candidate, SlotPromptCandidate):
        return ("slot", slot_candidate_template(candidate), candidate.use_word)
    if isinstance(candidate, PromptEnsembleCandidate):
        return ("ensemble", tuple(sorted((m.template, m.use_word) for m in candidate.members)))
    raise ValueError(f"Unknown candidate type: {type(candidate)}")


def cfg_mode_key(cfg: EvolutionConfig) -> str:
    if cfg.ensemble:
        return "ensemble"
    if cfg.slot_based:
        return "slot"
    return "single"


def mode_label_from_key(mode: str) -> str:
    if mode == "ensemble":
        return "ensemble refinement"
    if mode == "slot":
        return "slot-based structured search"
    return "free-form prompt evolution"


def cfg_mode_label(cfg: EvolutionConfig) -> str:
    return mode_label_from_key(cfg_mode_key(cfg))


def similarity_normalize_template(template: str) -> str:
    text = canonical_template(template).lower()
    text = text.replace("top down", "top-down")
    text = text.replace("close up", "close-up")
    text = text.replace("real world", "real-world")
    text = text.replace("photograph of", "photo of")
    text = text.replace("picture of", "photo of")
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s{}]", " ", text)
    text = re.sub(r"\bphotograph\b", "photo", text)
    text = re.sub(r"\bpicture\b", "photo", text)
    return canonical_template(text)


def similarity_token_set(template: str) -> set[str]:
    normalized = similarity_normalize_template(template)
    return {token for token in normalized.split() if token}


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def is_near_duplicate_template(
    template: str,
    existing_token_sets: Iterable[set[str]],
    threshold: float,
) -> bool:
    tokens = similarity_token_set(template)
    return any(jaccard_similarity(tokens, existing_tokens) >= threshold for existing_tokens in existing_token_sets)


def make_single_candidate_pool(
    templates: Iterable[str],
    existing_population: Iterable[PromptCandidate],
    cfg: EvolutionConfig,
    target_size: int,
    stats: SearchStats | None = None,
    enforce_near_duplicates: bool = True,
) -> list[PromptCandidate]:
    if target_size <= 0:
        return []

    existing_templates = {candidate.template for candidate in existing_population}
    candidate_keys = {candidate_equality_key(candidate) for candidate in existing_population}
    similarity_bank = [similarity_token_set(template) for template in existing_templates]
    accepted_templates: set[str] = set()
    accepted_candidates: list[PromptCandidate] = []
    use_word_options = (False,) if cfg.digit_only else (False, True)

    for raw_template in templates:
        normalized = canonical_template(raw_template)
        if not has_only_digit_placeholder(normalized):
            if stats is not None:
                stats.invalid_templates += 1
            continue
        if normalized in existing_templates or normalized in accepted_templates:
            if stats is not None:
                stats.exact_duplicates += 1
            continue
        if enforce_near_duplicates and is_near_duplicate_template(
            normalized,
            similarity_bank,
            cfg.near_duplicate_jaccard_threshold,
        ):
            if stats is not None:
                stats.near_duplicates += 1
            continue

        accepted_templates.add(normalized)
        similarity_bank.append(similarity_token_set(normalized))
        if len(accepted_candidates) < target_size:
            if stats is not None:
                stats.accepted_novel_templates += 1

            for use_word in use_word_options:
                candidate = PromptCandidate(template=normalized, use_word=use_word)
                key = candidate_equality_key(candidate)
                if key in candidate_keys:
                    continue
                candidate_keys.add(key)
                accepted_candidates.append(candidate)
                if len(accepted_candidates) >= target_size:
                    break

    return accepted_candidates[:target_size]


def make_slot_candidate_pool(
    proposals: Iterable[SlotPromptCandidate],
    existing_population: Iterable[SlotPromptCandidate],
    target_size: int,
) -> list[SlotPromptCandidate]:
    if target_size <= 0:
        return []

    seen = {candidate_equality_key(candidate) for candidate in existing_population}
    accepted: list[SlotPromptCandidate] = []
    for candidate in proposals:
        if not has_only_digit_placeholder(slot_candidate_template(candidate)):
            continue
        key = candidate_equality_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(candidate)
        if len(accepted) >= target_size:
            break
    return accepted[:target_size]


def make_ensemble_candidate_pool(
    member_groups: Iterable[Iterable[PromptCandidate]],
    existing_population: Iterable[PromptEnsembleCandidate],
    target_size: int,
) -> list[PromptEnsembleCandidate]:
    if target_size <= 0:
        return []

    seen = {candidate_equality_key(candidate) for candidate in existing_population}
    accepted: list[PromptEnsembleCandidate] = []
    for members in member_groups:
        ensemble = build_ensemble_candidate(members)
        if not ensemble.members:
            continue
        key = candidate_equality_key(ensemble)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(ensemble)
        if len(accepted) >= target_size:
            break
    return accepted[:target_size]


def _get_prompts_for_single_like(candidate: PromptCandidate | SlotPromptCandidate) -> list[str]:
    return render_candidate_prompts(candidate)


def score_predictions_with_nested_cv(
    candidate_predictions: dict[Candidate, torch.Tensor],
    labels: torch.Tensor,
    outer_folds: int,
    inner_folds: int,
    random_seed: int,
) -> tuple[list[CandidateScore], list[OuterFoldResult]]:
    prompt_inner_scores: dict[Candidate, list[float]] = {candidate: [] for candidate in candidate_predictions}
    prompt_selection_counts: dict[Candidate, int] = {candidate: 0 for candidate in candidate_predictions}
    prompt_outer_scores: dict[Candidate, list[float]] = {candidate: [] for candidate in candidate_predictions}
    outer_fold_results: list[OuterFoldResult] = []

    num_examples = len(labels)
    outer_generator = torch.Generator().manual_seed(random_seed)
    outer_permutation = torch.randperm(num_examples, generator=outer_generator)
    outer_fold_indices = list(torch.tensor_split(outer_permutation, outer_folds))

    for outer_fold_idx in range(outer_folds):
        console.print(f"[dim]  Outer fold {outer_fold_idx + 1}/{outer_folds}...[/dim]")
        test_indices = outer_fold_indices[outer_fold_idx]
        dev_indices = torch.cat([
            outer_fold_indices[i] for i in range(outer_folds) if i != outer_fold_idx
        ])

        inner_generator = torch.Generator().manual_seed(random_seed + outer_fold_idx + 1)
        inner_permutation = dev_indices[torch.randperm(len(dev_indices), generator=inner_generator)]
        inner_fold_indices = list(torch.tensor_split(inner_permutation, inner_folds))

        best_candidate: Candidate | None = None
        best_inner_score = float("-inf")

        for candidate, preds in candidate_predictions.items():
            inner_fold_scores: list[float] = []
            for inner_fold_idx in range(inner_folds):
                inner_val_indices = inner_fold_indices[inner_fold_idx]
                inner_val_correct = (preds[inner_val_indices] == labels[inner_val_indices]).sum().item()
                inner_val_accuracy = inner_val_correct / len(inner_val_indices)
                inner_fold_scores.append(inner_val_accuracy)

            mean_inner_score = sum(inner_fold_scores) / len(inner_fold_scores)
            prompt_inner_scores[candidate].append(mean_inner_score)

            if mean_inner_score > best_inner_score:
                best_inner_score = mean_inner_score
                best_candidate = candidate

        assert best_candidate is not None
        prompt_selection_counts[best_candidate] += 1

        for candidate, preds in candidate_predictions.items():
            outer_test_correct = (preds[test_indices] == labels[test_indices]).sum().item()
            outer_test_accuracy = outer_test_correct / len(test_indices)
            prompt_outer_scores[candidate].append(outer_test_accuracy)

        best_preds = candidate_predictions[best_candidate]
        best_outer_correct = (best_preds[test_indices] == labels[test_indices]).sum().item()
        best_outer_accuracy = best_outer_correct / len(test_indices)
        outer_fold_results.append(
            OuterFoldResult(
                fold_index=outer_fold_idx + 1,
                candidate=best_candidate,
                inner_score=best_inner_score,
                test_score=best_outer_accuracy,
            )
        )

    scores: list[CandidateScore] = []
    for candidate in candidate_predictions:
        inner_values = prompt_inner_scores[candidate]
        outer_values = prompt_outer_scores[candidate]
        outer_tensor = torch.tensor(outer_values, dtype=torch.float32)
        scores.append(
            CandidateScore(
                candidate=candidate,
                mean_inner_score=sum(inner_values) / len(inner_values),
                outer_mean_score=sum(outer_values) / len(outer_values),
                outer_std_score=outer_tensor.std(unbiased=False).item(),
                selected_count=prompt_selection_counts[candidate],
            )
        )

    scores.sort(key=lambda x: (x.mean_inner_score, x.outer_mean_score), reverse=True)
    return scores, outer_fold_results


def evaluate_single_like_population(
    candidates: list[PromptCandidate] | list[SlotPromptCandidate],
    normalized_image_features: torch.Tensor,
    labels: torch.Tensor,
    model: Any,
    processor: Any,
    device: str,
    logit_scale: torch.Tensor,
    outer_folds: int,
    inner_folds: int,
    random_seed: int,
    text_batch_size: int,
) -> tuple[list[CandidateScore], list[OuterFoldResult]]:
    prompt_predictions: dict[PromptCandidate | SlotPromptCandidate, torch.Tensor] = {}

    with torch.inference_mode():
        for start in range(0, len(candidates), text_batch_size):
            batch_candidates = candidates[start : start + text_batch_size]
            batch_prompts = [_get_prompts_for_single_like(c) for c in batch_candidates]
            flat_prompts = [prompt for prompts in batch_prompts for prompt in prompts]

            text_inputs = processor(
                text=flat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_latents = model.text_model(**text_inputs)
            text_features = model.text_projection(text_latents.pooler_output)
            normalized_text_features = F.normalize(text_features, dim=-1)
            normalized_text_features = normalized_text_features.view(len(batch_candidates), 10, -1)

            batch_logits = logit_scale * torch.einsum(
                "nd,bcd->bnc",
                normalized_image_features,
                normalized_text_features,
            )
            batch_preds = batch_logits.argmax(dim=-1)

            for idx, candidate in enumerate(batch_candidates):
                prompt_predictions[candidate] = batch_preds[idx]

    return score_predictions_with_nested_cv(
        candidate_predictions=prompt_predictions,
        labels=labels,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        random_seed=random_seed,
    )


def evaluate_ensemble_population(
    candidates: list[PromptEnsembleCandidate],
    normalized_image_features: torch.Tensor,
    labels: torch.Tensor,
    model: Any,
    processor: Any,
    device: str,
    logit_scale: torch.Tensor,
    outer_folds: int,
    inner_folds: int,
    random_seed: int,
    text_batch_size: int,
) -> tuple[list[CandidateScore], list[OuterFoldResult]]:
    prompt_predictions: dict[PromptEnsembleCandidate, torch.Tensor] = {}

    with torch.inference_mode():
        for start in range(0, len(candidates), text_batch_size):
            batch_candidates = candidates[start : start + text_batch_size]
            batch_member_prompts = [render_ensemble_member_prompts(c) for c in batch_candidates]
            flat_prompts = [
                p for member_prompts in batch_member_prompts for prompts in member_prompts for p in prompts
            ]

            text_inputs = processor(
                text=flat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_latents = model.text_model(**text_inputs)
            text_features = model.text_projection(text_latents.pooler_output)
            normalized_text_features = F.normalize(text_features, dim=-1)

            offset = 0
            for candidate in batch_candidates:
                member_features_list: list[torch.Tensor] = []
                for _ in range(len(candidate.members)):
                    member_feats = normalized_text_features[offset : offset + 10]
                    member_features_list.append(member_feats)
                    offset += 10
                member_features = torch.stack(member_features_list, dim=0)
                mean_text_features = member_features.mean(dim=0)
                mean_text_features = F.normalize(mean_text_features, dim=-1)
                batch_logits = logit_scale * (
                    normalized_image_features @ mean_text_features.T
                )
                batch_preds = batch_logits.argmax(dim=-1)
                prompt_predictions[candidate] = batch_preds
    return score_predictions_with_nested_cv(
        candidate_predictions=prompt_predictions,
        labels=labels,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        random_seed=random_seed,
    )


def evaluate_population_dispatch(
    candidates: list[Candidate],
    normalized_image_features: torch.Tensor,
    labels: torch.Tensor,
    model: Any,
    processor: Any,
    device: str,
    logit_scale: torch.Tensor,
    outer_folds: int,
    inner_folds: int,
    random_seed: int,
    text_batch_size: int,
    cfg: EvolutionConfig,
) -> tuple[list[CandidateScore], list[OuterFoldResult]]:
    if cfg.ensemble:
        return evaluate_ensemble_population(
            candidates,
            normalized_image_features,
            labels,
            model,
            processor,
            device,
            logit_scale,
            outer_folds,
            inner_folds,
            random_seed,
            text_batch_size,
        )
    return evaluate_single_like_population(
        candidates,
        normalized_image_features,
        labels,
        model,
        processor,
        device,
        logit_scale,
        outer_folds,
        inner_folds,
        random_seed,
        text_batch_size,
    )


def render_candidate_table(scores: list[CandidateScore], top_k: int = 10) -> None:
    for rank, score in enumerate(scores[:top_k], start=1):
        kind = format_candidate_kind(score.candidate)
        log.info(
            "#%d %s (%s) | inner=%.4f outer=%.4f ± %.4f selected=%d",
            rank,
            format_candidate_description(score.candidate),
            kind,
            score.mean_inner_score,
            score.outer_mean_score,
            score.outer_std_score,
            score.selected_count,
        )


def phase_name_from_prefix(phase_prefix: str) -> str:
    if not phase_prefix:
        return "main"
    return phase_prefix.rstrip("_")


def build_generation_panel(summary: GenerationSummary, diversity: int, device: str) -> Panel:
    kind = format_candidate_kind(summary.best.candidate)
    body = (
        f"[bold]Phase[/bold]: {summary.phase}\n"
        f"[bold]Mode[/bold]: {mode_label_from_key(summary.mode)}\n"
        f"[bold]Generation[/bold]: {summary.generation}\n"
        f"[bold]Device[/bold]: {device}\n"
        f"[bold]Population[/bold]: {summary.population_size}\n"
        f"[bold]Diversity[/bold]: {diversity}\n"
        f"[bold]Population mean outer score[/bold]: {summary.mean_outer_score:.4f}\n"
        f"[bold]Best candidate[/bold]: {format_candidate_description(summary.best.candidate)} ({kind})\n"
        f"[bold]Best inner score[/bold]: {summary.best.mean_inner_score:.4f}\n"
        f"[bold]Best outer score[/bold]: {summary.best.outer_mean_score:.4f} ± {summary.best.outer_std_score:.4f}\n"
        f"[bold]Selected count[/bold]: {summary.best.selected_count}"
    )
    return Panel(body, title="Prompt Evolution", expand=False)


def build_search_stats_table(search_stats: SearchStats) -> Table:
    table = Table(title="Generation Search Stats", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("LLM batches attempted", str(search_stats.llm_batches_attempted))
    table.add_row("LLM raw templates", str(search_stats.llm_raw_templates))
    table.add_row("Invalid templates", str(search_stats.invalid_templates))
    table.add_row("Exact duplicates", str(search_stats.exact_duplicates))
    table.add_row("Near-duplicates", str(search_stats.near_duplicates))
    table.add_row("Accepted novel templates", str(search_stats.accepted_novel_templates))
    table.add_row("Lexical mutations used", str(search_stats.lexical_mutations_used))
    table.add_row("Crossover children used", str(search_stats.crossover_children_used))
    return table


def build_population_stats_table(population_stats: PopulationStats) -> Table:
    table = Table(title="Population Composition", show_lines=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Population size", str(population_stats.population_size))
    table.add_row("Diversity", str(population_stats.diversity))
    table.add_row("Elites kept", str(population_stats.elite_count))
    table.add_row("Children kept", str(population_stats.children_count))
    table.add_row("LLM-generated children", str(population_stats.llm_generated_children))
    table.add_row("Lexical mutation children", str(population_stats.lexical_mutation_children))
    table.add_row("Crossover children", str(population_stats.crossover_children))
    table.add_row("Recombined children", str(population_stats.recombined_children))
    table.add_row("Filler candidates", str(population_stats.filler_candidates))
    return table


def render_generation_rich(
    generation: int,
    summary: GenerationSummary,
    scores: list[CandidateScore],
    outer_results: list[OuterFoldResult],
    diversity: int,
    device: str,
) -> None:
    console.rule(f"Generation {generation}")
    console.print(build_generation_panel(summary, diversity, device))

    score_table = Table(title="Top Candidates", show_lines=False)
    score_table.add_column("Rank", justify="right")
    score_table.add_column("Template")
    score_table.add_column("Kind")
    score_table.add_column("Inner", justify="right")
    score_table.add_column("Outer", justify="right")
    score_table.add_column("Std", justify="right")
    score_table.add_column("Selected", justify="right")

    for rank, score in enumerate(scores[: min(8, len(scores))], start=1):
        score_table.add_row(
            str(rank),
            format_candidate_description(score.candidate),
            format_candidate_kind(score.candidate),
            f"{score.mean_inner_score:.4f}",
            f"{score.outer_mean_score:.4f}",
            f"{score.outer_std_score:.4f}",
            str(score.selected_count),
        )

    console.print(score_table)
    console.print(build_search_stats_table(summary.search_stats))
    console.print(build_population_stats_table(summary.population_stats))

    fold_table = Table(title="Outer Fold Winners", show_lines=False)
    fold_table.add_column("Fold", justify="right")
    fold_table.add_column("Template")
    fold_table.add_column("Kind")
    fold_table.add_column("Inner", justify="right")
    fold_table.add_column("Test", justify="right")

    for result in outer_results:
        fold_table.add_row(
            str(result.fold_index),
            format_candidate_description(result.candidate),
            format_candidate_kind(result.candidate),
            f"{result.inner_score:.4f}",
            f"{result.test_score:.4f}",
        )

    console.print(fold_table)


def population_diversity(population: Iterable[Candidate]) -> int:
    return len({candidate_equality_key(c) for c in population})


def choose_parents(elites: list[CandidateScore], rng: random.Random) -> tuple[CandidateScore, CandidateScore]:
    if len(elites) == 1:
        return elites[0], elites[0]
    parent_a = rng.choice(elites)
    parent_b = rng.choice(elites)
    while parent_b == parent_a and len(elites) > 1:
        parent_b = rng.choice(elites)
    return parent_a, parent_b


def build_system_prompt() -> str:
    return (
        "You generate concise CLIP prompt templates for zero-shot recognition of real-world digits. "
        "Every prompt must contain the literal token {digit} exactly once. "
        "Focus on prompts describing photographed or top-down scenes where objects form a digit or number. "
        "Return only a plain newline-separated list of prompt templates, with no commentary."
    )


def build_mutation_prompt(
    elites: list[CandidateScore],
    existing_templates: set[str],
    batch_size: int,
) -> str:
    elite_lines = []
    for score in elites[: min(8, len(elites))]:
        kind = format_candidate_kind(score.candidate)
        desc = format_candidate_description(score.candidate)
        elite_lines.append(
            f"- {desc} [{kind}] inner={score.mean_inner_score:.4f} outer={score.outer_mean_score:.4f}"
        )

    existing_lines = sorted(existing_templates)
    existing_preview = "\n".join(f"- {line}" for line in existing_lines[:80])
    elite_block = "\n".join(elite_lines)

    return (
        f"Top-performing prompt templates so far:\n{elite_block}\n\n"
        "Generate improved prompt templates for CLIP zero-shot recognition of photographed real-world digits. "
        "Return a mixed batch: roughly one third local variants, one third moderate variants, and one third bolder but still task-relevant variants. "
        "Keep the domain grounded in photographed or real-world scenes where objects form a digit or number. "
        "You may explore different semantic angles such as shape, arrangement, composition, object-made numerals, found-object digits, and non-handwritten digits. "
        "Do not introduce new placeholder names (only {digit} is allowed). Do not add commentary or category labels. "
        "Favor concise prompt templates and preserve task relevance. "
        f"Produce at least {batch_size} and at most {batch_size * 2} new templates. "
        "Do not repeat any template from the existing set below.\n\n"
        f"Existing templates:\n{existing_preview}"
    )


def collect_llm_template_proposals(
    llm: PromptLLM,
    elites: list[CandidateScore],
    existing_templates: set[str],
    cfg: EvolutionConfig,
    rng: random.Random,
    search_stats: SearchStats,
) -> list[str]:
    system_prompt = build_system_prompt()
    templates: list[str] = []

    for batch_index in range(cfg.llm_mutation_batches_per_generation):
        if len(templates) >= cfg.max_llm_candidates_per_generation:
            break

        remaining_budget = cfg.max_llm_candidates_per_generation - len(templates)
        batch_size = max(1, min(cfg.batch_mutation_size, remaining_budget))
        mutation_prompt = build_mutation_prompt(elites, existing_templates, batch_size)

        for attempt in range(cfg.max_attempts_per_batch):
            search_stats.llm_batches_attempted += 1
            console.print(
                f"[dim]  LLM mutation batch {batch_index + 1}/{cfg.llm_mutation_batches_per_generation} "
                f"(attempt {attempt + 1}/{cfg.max_attempts_per_batch})...[/dim]"
            )
            mutated_templates = llm.generate_templates(system_prompt, mutation_prompt, rng)
            search_stats.llm_raw_templates += len(mutated_templates)
            if not mutated_templates:
                continue
            templates.extend(mutated_templates[:remaining_budget])
            break

    return templates[: cfg.max_llm_candidates_per_generation]


def generate_lexical_template_proposals(
    elites: list[CandidateScore],
    target_size: int,
    rng: random.Random,
) -> list[str]:
    substitutions = [
        ("overhead", "top-down"),
        ("top-down", "overhead"),
        ("photo", "image"),
        ("image", "photo"),
        ("number", "digit"),
        ("digit", "number"),
        ("formed by", "made from"),
        ("made from", "formed by"),
        ("objects", "real-world objects"),
        ("objects", "everyday objects"),
        ("objects forming", "objects arranged as"),
        ("objects arranged as", "objects forming"),
        ("number", "non-handwritten digit"),
        ("digit", "number"),
    ]
    proposals: list[str] = []
    if target_size <= 0:
        return proposals

    max_attempts = max(target_size * 6, 18)
    attempts = 0
    while len(proposals) < target_size * 3 and attempts < max_attempts:
        attempts += 1
        fallback_parent = rng.choice(elites).candidate
        if not isinstance(fallback_parent, PromptCandidate):
            continue
        old, new = rng.choice(substitutions)
        mutated_template = canonical_template(fallback_parent.template.replace(old, new, 1))
        proposals.append(mutated_template)
    return proposals


def extract_single_member_scores(
    elites: list[CandidateScore],
    limit: int = 20,
) -> list[CandidateScore]:
    extracted: list[CandidateScore] = []
    seen: set[Hashable] = set()
    for score in elites:
        candidate = score.candidate
        if isinstance(candidate, PromptCandidate):
            members = [candidate]
        elif isinstance(candidate, PromptEnsembleCandidate):
            members = list(candidate.members)
        else:
            members = []
        for member in members:
            key = candidate_equality_key(member)
            if key in seen:
                continue
            seen.add(key)
            extracted.append(
                CandidateScore(
                    candidate=member,
                    mean_inner_score=score.mean_inner_score,
                    outer_mean_score=score.outer_mean_score,
                    outer_std_score=score.outer_std_score,
                    selected_count=score.selected_count,
                )
            )
            if len(extracted) >= limit:
                return extracted
    return extracted


def make_offspring(
    llm: PromptLLM,
    elites: list[CandidateScore],
    target_children: int,
    existing_population: list[PromptCandidate],
    cfg: EvolutionConfig,
    rng: random.Random,
) -> OffspringBuildResult:
    search_stats = SearchStats()
    existing_templates = {candidate.template for candidate in existing_population}

    raw_llm_templates = collect_llm_template_proposals(
        llm=llm,
        elites=elites,
        existing_templates=existing_templates,
        cfg=cfg,
        rng=rng,
        search_stats=search_stats,
    )
    llm_children = make_single_candidate_pool(
        templates=raw_llm_templates,
        existing_population=existing_population,
        cfg=cfg,
        target_size=min(target_children, cfg.max_novel_candidates_kept),
        stats=search_stats,
    )

    lexical_templates = generate_lexical_template_proposals(
        elites=elites,
        target_size=target_children - len(llm_children),
        rng=rng,
    )
    lexical_children = make_single_candidate_pool(
        templates=lexical_templates,
        existing_population=existing_population + llm_children,
        cfg=cfg,
        target_size=max(0, target_children - len(llm_children)),
    )
    search_stats.lexical_mutations_used = len(lexical_children)

    children = (llm_children + lexical_children)[:target_children]
    if len(children) % 5 == 0 or len(children) >= target_children:
        console.print(f"[dim]  Offspring: {len(children)}/{target_children}[/dim]")

    return OffspringBuildResult(
        children=children,
        search_stats=search_stats,
        llm_generated_children=len(llm_children),
        lexical_mutation_children=len(lexical_children),
        crossover_children=0,
        recombined_children=0,
    )


def make_slot_offspring(
    elites: list[CandidateScore],
    target_children: int,
    existing_population: list[SlotPromptCandidate],
    cfg: EvolutionConfig,
    rng: random.Random,
) -> OffspringBuildResult:
    children: list[SlotPromptCandidate] = []
    use_word_options = (False,) if cfg.digit_only else (False, True)
    search_stats = SearchStats()
    recombined_children = 0

    while len(children) < target_children:
        parent_a, parent_b = choose_parents(elites, rng)
        a = parent_a.candidate
        b = parent_b.candidate
        if not isinstance(a, SlotPromptCandidate) or not isinstance(b, SlotPromptCandidate):
            continue
        child = SlotPromptCandidate(
            viewpoint=rng.choice([a.viewpoint, b.viewpoint]),
            medium=rng.choice([a.medium, b.medium]),
            object_phrase=rng.choice([a.object_phrase, b.object_phrase]),
            target_phrase=rng.choice([a.target_phrase, b.target_phrase]),
            realism_phrase=rng.choice([a.realism_phrase, b.realism_phrase]),
            use_word=rng.choice([a.use_word, b.use_word]),
        )
        accepted = make_slot_candidate_pool([child], existing_population + children, 1)
        if accepted:
            children.extend(accepted)
            recombined_children += len(accepted)
        if len(children) % 5 == 0 or len(children) >= target_children:
            console.print(f"[dim]  Offspring: {len(children)}/{target_children}[/dim]")
        if len(children) >= target_children:
            break

        if rng.random() < cfg.slot_mutation_rate:
            mut_parent = rng.choice(elites).candidate
            if not isinstance(mut_parent, SlotPromptCandidate):
                continue
            mut_candidate = SlotPromptCandidate(
                viewpoint=rng.choice(VIEWPOINT_OPTIONS) if rng.random() < 0.4 else mut_parent.viewpoint,
                medium=rng.choice(MEDIUM_OPTIONS) if rng.random() < 0.4 else mut_parent.medium,
                object_phrase=rng.choice(OBJECT_PHRASE_OPTIONS) if rng.random() < 0.4 else mut_parent.object_phrase,
                target_phrase=rng.choice(TARGET_PHRASE_OPTIONS) if rng.random() < 0.4 else mut_parent.target_phrase,
                realism_phrase=rng.choice(REALISM_OPTIONS) if rng.random() < 0.4 else mut_parent.realism_phrase,
                use_word=rng.choice(use_word_options) if rng.random() < 0.3 else mut_parent.use_word,
            )
            accepted = make_slot_candidate_pool([mut_candidate], existing_population + children, 1)
            if accepted:
                children.extend(accepted)
                recombined_children += len(accepted)
            if len(children) % 5 == 0 or len(children) >= target_children:
                console.print(f"[dim]  Offspring: {len(children)}/{target_children}[/dim]")

    return OffspringBuildResult(
        children=children[:target_children],
        search_stats=search_stats,
        recombined_children=recombined_children,
    )


def make_ensemble_offspring(
    llm: PromptLLM,
    elites: list[CandidateScore],
    target_children: int,
    existing_population: list[PromptEnsembleCandidate],
    cfg: EvolutionConfig,
    rng: random.Random,
) -> OffspringBuildResult:
    children: list[PromptEnsembleCandidate] = []
    search_stats = SearchStats()
    elite_pool: list[PromptCandidate] = []
    for score in elites[: min(20, len(elites))]:
        c = score.candidate
        if isinstance(c, PromptEnsembleCandidate):
            elite_pool.extend(c.members)
        elif isinstance(c, PromptCandidate):
            elite_pool.append(c)
    elite_pool = list({(m.template, m.use_word): m for m in elite_pool}.values())
    if not elite_pool:
        for template in DEFAULT_SEED_PROMPTS:
            cand = build_candidate(template, False)
            if cand:
                elite_pool.append(cand)

    ensemble_size = cfg.ensemble_size
    recombined_children = 0
    llm_generated_children = 0

    llm_member_scores = extract_single_member_scores(elites)
    llm_member_templates = collect_llm_template_proposals(
        llm=llm,
        elites=llm_member_scores,
        existing_templates={candidate.template for candidate in elite_pool},
        cfg=cfg,
        rng=rng,
        search_stats=search_stats,
    ) if llm_member_scores else []
    llm_member_candidates = make_single_candidate_pool(
        templates=llm_member_templates,
        existing_population=elite_pool,
        cfg=cfg,
        target_size=min(cfg.max_novel_candidates_kept, max(1, target_children)),
        stats=search_stats,
    )

    # Build extended pool for fallback when crossover saturates (elite overlap limits unique combos)
    extended_pool = list(elite_pool)
    extended_pool.extend(llm_member_candidates)
    extended_pool = list({(m.template, m.use_word): m for m in extended_pool}.values())
    for template in DEFAULT_SEED_PROMPTS:
        cand = build_candidate(template, False)
        if cand and (cand.template, cand.use_word) not in {(m.template, m.use_word) for m in extended_pool}:
            extended_pool.append(cand)

    stuck_count = 0
    stuck_threshold = 150
    max_attempts = max(target_children * 60, 200)
    attempts = 0

    while len(children) < target_children and attempts < max_attempts:
        attempts += 1
        if stuck_count >= stuck_threshold and len(extended_pool) >= ensemble_size:
            console.print(f"[dim]  Offspring stuck at {len(children)}/{target_children}, using extended pool...[/dim]")
            stuck_count = 0
            chosen = rng.sample(extended_pool, ensemble_size)
            accepted = make_ensemble_candidate_pool([chosen], existing_population + children, 1)
            if accepted:
                children.extend(accepted)
                recombined_children += len(accepted)
                if len(children) % 5 == 0 or len(children) >= target_children:
                    console.print(f"[dim]  Offspring: {len(children)}/{target_children}[/dim]")
            continue

        parent_a, parent_b = choose_parents(elites, rng)
        a = parent_a.candidate
        b = parent_b.candidate
        members_a = a.members if isinstance(a, PromptEnsembleCandidate) else [a]
        members_b = b.members if isinstance(b, PromptEnsembleCandidate) else [b]
        pool = list({(m.template, m.use_word): m for m in members_a + members_b}.values())
        while len(pool) < ensemble_size and elite_pool:
            pool.append(rng.choice(elite_pool))
            pool = list({(m.template, m.use_word): m for m in pool}.values())
        if len(pool) < ensemble_size:
            for template in DEFAULT_SEED_PROMPTS:
                cand = build_candidate(template, False)
                if cand and (cand.template, cand.use_word) not in {(m.template, m.use_word) for m in pool}:
                    pool.append(cand)
                    if len(pool) >= ensemble_size:
                        break
        if len(pool) < ensemble_size:
            stuck_count += 1
            continue
        chosen = rng.sample(pool, ensemble_size)
        accepted = make_ensemble_candidate_pool([chosen], existing_population + children, 1)
        if not accepted:
            stuck_count += 1
            continue
        stuck_count = 0
        children.extend(accepted)
        recombined_children += len(accepted)
        if len(children) % 5 == 0 or len(children) >= target_children:
            console.print(f"[dim]  Offspring: {len(children)}/{target_children}[/dim]")
        if len(children) >= target_children:
            break

        if rng.random() < cfg.mutation_rate and llm_member_candidates:
            parent = rng.choice(elites).candidate
            base_members = list(parent.members) if isinstance(parent, PromptEnsembleCandidate) else [parent]
            idx = rng.randint(0, len(base_members) - 1) if len(base_members) > 1 else 0
            replacement = rng.choice(llm_member_candidates)
            new_members = base_members[:idx] + [replacement] + base_members[idx + 1 :]
            new_members = list({(m.template, m.use_word): m for m in new_members}.values())
            while len(new_members) < ensemble_size and extended_pool:
                new_members.append(rng.choice(extended_pool))
                new_members = list({(m.template, m.use_word): m for m in new_members}.values())
            if len(new_members) >= ensemble_size:
                chosen = rng.sample(new_members, ensemble_size)
                accepted = make_ensemble_candidate_pool([chosen], existing_population + children, 1)
                if accepted:
                    stuck_count = 0
                    children.extend(accepted)
                    llm_generated_children += len(accepted)
                    if len(children) % 5 == 0 or len(children) >= target_children:
                        console.print(f"[dim]  Offspring: {len(children)}/{target_children}[/dim]")

    return OffspringBuildResult(
        children=children[:target_children],
        search_stats=search_stats,
        llm_generated_children=llm_generated_children,
        recombined_children=recombined_children,
    )


def make_offspring_dispatch(
    llm: PromptLLM,
    elites: list[CandidateScore],
    target_children: int,
    existing_population: list[Candidate],
    cfg: EvolutionConfig,
    rng: random.Random,
) -> OffspringBuildResult:
    if cfg.ensemble:
        return make_ensemble_offspring(
            llm, elites, target_children,
            [c for c in existing_population if isinstance(c, PromptEnsembleCandidate)],
            cfg, rng,
        )
    if cfg.slot_based:
        return make_slot_offspring(
            elites, target_children,
            [c for c in existing_population if isinstance(c, SlotPromptCandidate)],
            cfg, rng,
        )
    return make_offspring(
        llm, elites, target_children,
        [c for c in existing_population if isinstance(c, PromptCandidate)],
        cfg, rng,
    )


def save_generation(
    output_dir: Path,
    generation: int,
    scores: list[CandidateScore],
    summary: GenerationSummary,
    phase_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dict = candidate_to_dict(summary.best.candidate)
    best_dict["mean_inner_score"] = summary.best.mean_inner_score
    best_dict["outer_mean_score"] = summary.best.outer_mean_score
    best_dict["outer_std_score"] = summary.best.outer_std_score
    best_dict["selected_count"] = summary.best.selected_count
    payload = {
        "generation": generation,
        "mode": summary.mode,
        "phase": summary.phase,
        "search_stats": asdict(summary.search_stats),
        "population_stats": asdict(summary.population_stats),
        "summary": {
            "generation": summary.generation,
            "best": best_dict,
            "mean_outer_score": summary.mean_outer_score,
            "population_size": summary.population_size,
        },
        "scores": [
            {
                **candidate_to_dict(score.candidate),
                "mean_inner_score": score.mean_inner_score,
                "outer_mean_score": score.outer_mean_score,
                "outer_std_score": score.outer_std_score,
                "selected_count": score.selected_count,
            }
            for score in scores
        ],
    }
    filename = f"{phase_prefix}generation_{generation:03d}.json" if phase_prefix else f"generation_{generation:03d}.json"
    with (output_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def initialize_slot_population(
    cfg: EvolutionConfig, rng: random.Random, target_size: int
) -> list[SlotPromptCandidate]:
    use_word_options = (False,) if cfg.digit_only else (False, True)
    proposals: list[SlotPromptCandidate] = []
    while len(proposals) < target_size * 3:
        proposals.append(
            SlotPromptCandidate(
            viewpoint=rng.choice(VIEWPOINT_OPTIONS),
            medium=rng.choice(MEDIUM_OPTIONS),
            object_phrase=rng.choice(OBJECT_PHRASE_OPTIONS),
            target_phrase=rng.choice(TARGET_PHRASE_OPTIONS),
            realism_phrase=rng.choice(REALISM_OPTIONS),
            use_word=rng.choice(use_word_options),
        )
        )
    return make_slot_candidate_pool(proposals, [], target_size)


def initialize_ensemble_population_from_seed_pool(
    seed_pool: Iterable[PromptCandidate],
    cfg: EvolutionConfig,
    rng: random.Random,
    target_size: int,
) -> list[PromptEnsembleCandidate]:
    unique_seed_pool = list({(m.template, m.use_word): m for m in seed_pool}.values())
    top_k = min(cfg.ensemble_seed_pool_size, len(unique_seed_pool))
    unique_seed_pool = unique_seed_pool[:top_k]

    if len(unique_seed_pool) < cfg.ensemble_size:
        for template in DEFAULT_SEED_PROMPTS:
            cand = build_candidate(template, False)
            if cand and (cand.template, cand.use_word) not in {(m.template, m.use_word) for m in unique_seed_pool}:
                unique_seed_pool.append(cand)
                if len(unique_seed_pool) >= cfg.ensemble_size:
                    break

    proposals: list[list[PromptCandidate]] = []
    max_attempts = max(target_size * 20, 100)
    attempts = 0
    while len(proposals) < target_size * 3 and attempts < max_attempts:
        attempts += 1
        if len(unique_seed_pool) < cfg.ensemble_size:
            break
        proposals.append(rng.sample(unique_seed_pool, cfg.ensemble_size))
    return make_ensemble_candidate_pool(proposals, [], target_size)


def initialize_ensemble_population(
    cfg: EvolutionConfig, rng: random.Random, target_size: int
) -> list[PromptEnsembleCandidate]:
    seed_pool: list[PromptCandidate] = []
    for template in DEFAULT_SEED_PROMPTS:
        for use_word in ((False,) if cfg.digit_only else (False, True)):
            cand = build_candidate(template, use_word)
            if cand:
                seed_pool.append(cand)
    return initialize_ensemble_population_from_seed_pool(seed_pool, cfg, rng, target_size)


def initialize_population_dispatch(
    cfg: EvolutionConfig, rng: random.Random, target_size: int
) -> list[Candidate]:
    if cfg.ensemble:
        return initialize_ensemble_population(cfg, rng, target_size)
    if cfg.slot_based:
        return initialize_slot_population(cfg, rng, target_size)
    return initialize_population(cfg, rng, target_size)


def initialize_population(
    cfg: EvolutionConfig, rng: random.Random, target_size: int
) -> list[PromptCandidate]:
    population_templates: list[str] = []
    for template in DEFAULT_SEED_PROMPTS:
        for use_word in ((False,) if cfg.digit_only else (False, True)):
            candidate = build_candidate(template, use_word)
            if candidate is not None:
                population_templates.append(candidate.template)

    rng.shuffle(population_templates)
    unique_population = make_single_candidate_pool(
        templates=population_templates,
        existing_population=[],
        cfg=cfg,
        target_size=target_size,
        enforce_near_duplicates=False,
    )

    substitutions = [
        ("top-down", "overhead"),
        ("overhead", "top-down"),
        ("image", "photo"),
        ("photo", "image"),
        ("number", "digit"),
        ("digit", "number"),
        ("formed by", "made from"),
        ("made from", "formed by"),
        ("objects", "everyday objects"),
        ("objects", "real-world objects"),
    ]

    while len(unique_population) < target_size:
        parent = rng.choice(unique_population)
        old, new = rng.choice(substitutions)
        mutated_template = canonical_template(parent.template.replace(old, new, 1))
        proposals = make_single_candidate_pool(
            templates=[mutated_template],
            existing_population=unique_population,
            cfg=cfg,
            target_size=1,
            enforce_near_duplicates=False,
        )
        if proposals:
            unique_population.extend(proposals)

    return unique_population[:target_size]


def _run_evolution_phase(
    cfg: EvolutionConfig,
    population: list[Candidate],
    num_generations: int,
    outer_folds: int,
    inner_folds: int,
    phase_population_size: int,
    phase_elite_size: int,
    phase_children: int,
    phase_prefix: str,
    seed_offset: int,
    llm: PromptLLM,
    model: Any,
    processor: Any,
    normalized_image_features: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    device: str,
    output_dir: Path,
    rng: random.Random,
) -> tuple[list[GenerationSummary], list[Candidate], list[CandidateScore]]:
    history: list[GenerationSummary] = []
    phase_name = phase_name_from_prefix(phase_prefix)
    mode_key = cfg_mode_key(cfg)
    for generation in range(num_generations):
        phase_label = phase_name.replace("_", " ").title() if phase_prefix else "Generation"
        log.info("=== %s %d ===", phase_label, generation)
        console.print(f"[dim]  Evaluating {len(population)} candidates (nested CV, {outer_folds}x{inner_folds} folds)...[/dim]")
        scores, outer_results = evaluate_population_dispatch(
            candidates=population,
            normalized_image_features=normalized_image_features,
            labels=labels,
            model=model,
            processor=processor,
            device=device,
            logit_scale=logit_scale,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            random_seed=cfg.random_seed + seed_offset + generation * 17,
            text_batch_size=cfg.text_batch_size,
            cfg=cfg,
        )
        console.print("[green]  ✓[/green] Evaluation complete.")

        best = scores[0]
        mean_outer_score = sum(score.outer_mean_score for score in scores) / len(scores)
        diversity = population_diversity(population)
        log.info(
            "%s %d best: %s (%s) | inner=%.4f outer=%.4f ± %.4f selected=%d",
            phase_label,
            generation,
            format_candidate_description(best.candidate),
            format_candidate_kind(best.candidate),
            best.mean_inner_score,
            best.outer_mean_score,
            best.outer_std_score,
            best.selected_count,
        )

        search_stats = SearchStats()
        population_stats = PopulationStats(
            population_size=len(population),
            diversity=diversity,
            elite_count=0,
            children_count=0,
        )
        if generation < num_generations - 1:
            console.print("[dim]  Generating offspring...[/dim]")
            elites = scores[: min(phase_elite_size, len(scores))]
            elite_candidates = [score.candidate for score in elites]
            target_children = min(phase_children, phase_population_size - len(elite_candidates))
            offspring_result = make_offspring_dispatch(
                llm=llm,
                elites=elites,
                target_children=target_children,
                existing_population=elite_candidates,
                cfg=cfg,
                rng=rng,
            )
            children = list(offspring_result.children)
            filler_candidates = 0
            if len(elite_candidates) + len(children) < phase_population_size:
                filler_population = initialize_population_dispatch(
                    cfg, rng, phase_population_size
                )
                existing_keys = {candidate_equality_key(c) for c in elite_candidates + children}
                for candidate in filler_population:
                    key = candidate_equality_key(candidate)
                    if key in existing_keys:
                        continue
                    children.append(candidate)
                    filler_candidates += 1
                    existing_keys.add(key)
                    if len(elite_candidates) + len(children) >= phase_population_size:
                        break
            search_stats = offspring_result.search_stats
            population_stats = PopulationStats(
                population_size=len(population),
                diversity=diversity,
                elite_count=len(elite_candidates),
                children_count=len(children),
                llm_generated_children=offspring_result.llm_generated_children,
                lexical_mutation_children=offspring_result.lexical_mutation_children,
                crossover_children=offspring_result.crossover_children,
                recombined_children=offspring_result.recombined_children,
                filler_candidates=filler_candidates,
            )
            summary = GenerationSummary(
                generation=generation,
                best=best,
                mean_outer_score=mean_outer_score,
                population_size=len(population),
                mode=mode_key,
                phase=phase_name,
                search_stats=search_stats,
                population_stats=population_stats,
            )
            history.append(summary)
            render_generation_rich(generation, summary, scores, outer_results, diversity, device)
            save_generation(output_dir, generation, scores, summary, phase_prefix=phase_prefix)
            population = (elite_candidates + children)[:phase_population_size]
            console.print("[green]  ✓[/green] Offspring generated.")
        else:
            summary = GenerationSummary(
                generation=generation,
                best=best,
                mean_outer_score=mean_outer_score,
                population_size=len(population),
                mode=mode_key,
                phase=phase_name,
                search_stats=search_stats,
                population_stats=population_stats,
            )
            history.append(summary)
            render_generation_rich(generation, summary, scores, outer_results, diversity, device)
            save_generation(output_dir, generation, scores, summary, phase_prefix=phase_prefix)

    scores, _ = evaluate_population_dispatch(
        candidates=population,
        normalized_image_features=normalized_image_features,
        labels=labels,
        model=model,
        processor=processor,
        device=device,
        logit_scale=logit_scale,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        random_seed=cfg.random_seed + seed_offset + num_generations * 17,
        text_batch_size=cfg.text_batch_size,
        cfg=cfg,
    )
    return history, population, scores


def resolve_phase_configs(cfg: EvolutionConfig) -> tuple[EvolutionConfig, EvolutionConfig]:
    if not cfg.two_phase or cfg.phase2_mode == "same":
        return cfg, cfg

    phase1_cfg = replace(
        cfg,
        ensemble=False,
        slot_based=False,
        two_phase=False,
        phase2_mode="same",
    )
    if cfg.phase2_mode == "ensemble":
        phase2_cfg = replace(
            cfg,
            ensemble=True,
            slot_based=False,
            two_phase=False,
            phase2_mode="same",
        )
    else:
        phase2_cfg = replace(
            cfg,
            ensemble=False,
            slot_based=False,
            two_phase=False,
            phase2_mode="same",
        )
    return phase1_cfg, phase2_cfg


def evolve_prompts(cfg: EvolutionConfig, clip_cfg: FoundationModelConfig) -> None:
    rng = random.Random(cfg.random_seed)
    device = get_best_device(cfg.use_mps)
    phase1_cfg, phase2_cfg = resolve_phase_configs(cfg)
    console.print("[bold]Starting evolution run[/bold]")
    console.print(f"[dim]Loading LLM ({cfg.llm_model})...[/dim]")
    llm = PromptLLM(
        model_name=cfg.llm_model,
        device=device,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    model, processor = load_foundation_model(clip_cfg.model)
    model_move_dtype = get_model_move_dtype(device)
    if model_move_dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=model_move_dtype)
    model.eval()
    logit_scale = model.logit_scale.exp()
    console.print("[green]✓[/green] CLIP model loaded.")

    console.print("[dim]Computing image features (may load from cache)...[/dim]")
    with torch.inference_mode():
        _, _, normalized_image_features, labels = compute_foundation_model_features(
            model,
            processor,
            clip_cfg,
            device,
        )
    console.print("[green]✓[/green] Image features ready.")

    output_dir = Path(cfg.cache_dir)
    panel_body = (
        f"[bold]Device[/bold]: {device}\n"
        f"[bold]Digit only[/bold]: {cfg.digit_only}\n"
        f"[bold]Text batch size[/bold]: {cfg.text_batch_size}"
    )
    if cfg.two_phase:
        panel_body += (
            f"\n[bold]Phase 1 Mode[/bold]: {cfg_mode_label(phase1_cfg)}\n"
            f"[bold]Phase 2 Mode[/bold]: {cfg_mode_label(phase2_cfg)}\n"
            f"[bold]Phase 1[/bold]: pop={cfg.phase1_population_size} gen={cfg.phase1_generations} "
            f"folds={cfg.phase1_outer_folds}x{cfg.phase1_inner_folds}\n"
            f"[bold]Phase 2[/bold]: pop={cfg.phase2_population_size} gen={cfg.phase2_generations} "
            f"folds={cfg.phase2_outer_folds}x{cfg.phase2_inner_folds} seed_top_k={cfg.phase2_seed_top_k}"
        )
    else:
        panel_body += (
            f"\n[bold]Mode[/bold]: {cfg_mode_label(cfg)}\n"
            f"\n[bold]Population size[/bold]: {cfg.population_size}\n"
            f"[bold]Generations[/bold]: {cfg.generations}\n"
            f"[bold]Outer folds[/bold]: {cfg.outer_folds}\n"
            f"[bold]Inner folds[/bold]: {cfg.inner_folds}"
        )
    console.print(Panel(panel_body, title="Evolution Run", expand=False))

    console.print("[dim]Initializing population...[/dim]")
    if cfg.two_phase:
        population = initialize_population_dispatch(
            phase1_cfg, rng, cfg.phase1_population_size
        )
        console.print(f"[green]✓[/green] Phase 1 population initialized ({len(population)} candidates).")
        console.print("[bold cyan]── Phase 1 ──[/bold cyan]")
        phase1_elite = max(1, cfg.phase1_population_size // 4)
        phase1_children = cfg.phase1_population_size - phase1_elite
        phase1_history, _, phase1_scores = _run_evolution_phase(
            cfg=phase1_cfg,
            population=population,
            num_generations=cfg.phase1_generations,
            outer_folds=cfg.phase1_outer_folds,
            inner_folds=cfg.phase1_inner_folds,
            phase_population_size=cfg.phase1_population_size,
            phase_elite_size=phase1_elite,
            phase_children=phase1_children,
            phase_prefix="phase1_",
            seed_offset=0,
            llm=llm,
            model=model,
            processor=processor,
            normalized_image_features=normalized_image_features,
            labels=labels,
            logit_scale=logit_scale,
            device=device,
            output_dir=output_dir,
            rng=rng,
        )
        best_phase1 = max(phase1_history, key=lambda h: h.best.outer_mean_score)
        console.print("[green]✓[/green] Phase 1 complete.")

        console.print(f"[dim]Seeding Phase 2 from top {cfg.phase2_seed_top_k} candidates...[/dim]")
        top_k = min(cfg.phase2_seed_top_k, len(phase1_scores))
        seed_candidates = [phase1_scores[i].candidate for i in range(top_k)]
        seed_scores = phase1_scores[:top_k]
        phase2_elite = max(1, cfg.phase2_population_size // 4)
        phase2_children = cfg.phase2_population_size - phase2_elite
        if phase2_cfg.ensemble:
            single_seed_pool = [candidate for candidate in seed_candidates if isinstance(candidate, PromptCandidate)]
            phase2_population = initialize_ensemble_population_from_seed_pool(
                single_seed_pool,
                phase2_cfg,
                rng,
                cfg.phase2_population_size,
            )
            existing_keys = {candidate_equality_key(c) for c in phase2_population}
            while len(phase2_population) < cfg.phase2_population_size:
                more = initialize_population_dispatch(
                    phase2_cfg, rng, cfg.phase2_population_size
                )
                for c in more:
                    if candidate_equality_key(c) not in existing_keys:
                        phase2_population.append(c)
                        existing_keys.add(candidate_equality_key(c))
                        if len(phase2_population) >= cfg.phase2_population_size:
                            break
            phase2_population = phase2_population[: cfg.phase2_population_size]
        else:
            need_filler = cfg.phase2_population_size - len(seed_candidates)
            if need_filler > 0:
                filler_result = make_offspring_dispatch(
                    llm=llm,
                    elites=seed_scores,
                    target_children=need_filler,
                    existing_population=seed_candidates,
                    cfg=phase2_cfg,
                    rng=rng,
                )
                phase2_population = seed_candidates + filler_result.children
            else:
                phase2_population = seed_candidates[: cfg.phase2_population_size]
            existing_keys = {candidate_equality_key(c) for c in phase2_population}
            while len(phase2_population) < cfg.phase2_population_size:
                more = initialize_population_dispatch(
                    phase2_cfg, rng, cfg.phase2_population_size
                )
                for c in more:
                    if candidate_equality_key(c) not in existing_keys:
                        phase2_population.append(c)
                        existing_keys.add(candidate_equality_key(c))
                        if len(phase2_population) >= cfg.phase2_population_size:
                            break
            phase2_population = phase2_population[: cfg.phase2_population_size]

        console.print(f"[green]✓[/green] Phase 2 population ready ({len(phase2_population)} candidates).")
        console.print("[bold cyan]── Phase 2 ──[/bold cyan]")
        phase2_history, _, phase2_scores = _run_evolution_phase(
            cfg=phase2_cfg,
            population=phase2_population,
            num_generations=cfg.phase2_generations,
            outer_folds=cfg.phase2_outer_folds,
            inner_folds=cfg.phase2_inner_folds,
            phase_population_size=cfg.phase2_population_size,
            phase_elite_size=phase2_elite,
            phase_children=phase2_children,
            phase_prefix="phase2_",
            seed_offset=1000,
            llm=llm,
            model=model,
            processor=processor,
            normalized_image_features=normalized_image_features,
            labels=labels,
            logit_scale=logit_scale,
            device=device,
            output_dir=output_dir,
            rng=rng,
        )
        best_phase2 = max(phase2_history, key=lambda h: h.best.outer_mean_score)
        final_best = phase2_scores[0] if phase2_scores else best_phase2.best

        console.rule("Two-Phase Summary")
        summary_panel = Panel(
            f"[bold]Best Phase 1[/bold]: {format_candidate_description(best_phase1.best.candidate)} ({format_candidate_kind(best_phase1.best.candidate)})\n"
            f"  inner={best_phase1.best.mean_inner_score:.4f} outer={best_phase1.best.outer_mean_score:.4f}\n"
            f"[bold]Best Phase 2[/bold]: {format_candidate_description(best_phase2.best.candidate)} ({format_candidate_kind(best_phase2.best.candidate)})\n"
            f"  inner={best_phase2.best.mean_inner_score:.4f} outer={best_phase2.best.outer_mean_score:.4f}\n"
            f"[bold]Final selected[/bold]: {format_candidate_description(final_best.candidate)} ({format_candidate_kind(final_best.candidate)})\n"
            f"  inner={final_best.mean_inner_score:.4f} outer={final_best.outer_mean_score:.4f}",
            title="Two-Phase Evolution Complete",
            expand=False,
        )
        console.print(summary_panel)
        console.print("[green]✓[/green] Phase 2 complete.")
        log.info(
            "Two-phase complete | Phase 1 best: %s | Phase 2 best: %s | Final: %s",
            format_candidate_description(best_phase1.best.candidate),
            format_candidate_description(best_phase2.best.candidate),
            format_candidate_description(final_best.candidate),
        )
        console.print("[bold green]Evolution complete.[/bold green]")
    else:
        population = initialize_population_dispatch(cfg, rng, cfg.population_size)
        console.print(f"[green]✓[/green] Population initialized ({len(population)} candidates).")
        history, _, _ = _run_evolution_phase(
            cfg=cfg,
            population=population,
            num_generations=cfg.generations,
            outer_folds=cfg.outer_folds,
            inner_folds=cfg.inner_folds,
            phase_population_size=cfg.population_size,
            phase_elite_size=cfg.elite_size,
            phase_children=cfg.children_per_generation,
            phase_prefix="",
            seed_offset=0,
            llm=llm,
            model=model,
            processor=processor,
            normalized_image_features=normalized_image_features,
            labels=labels,
            logit_scale=logit_scale,
            device=device,
            output_dir=output_dir,
            rng=rng,
        )
        if history:
            best_generation = max(history, key=lambda item: item.best.outer_mean_score)
            log.info(
                "Best overall generation: %d | %s (%s) | inner=%.4f outer=%.4f ± %.4f",
                best_generation.generation,
                format_candidate_description(best_generation.best.candidate),
                format_candidate_kind(best_generation.best.candidate),
                best_generation.best.mean_inner_score,
                best_generation.best.outer_mean_score,
                best_generation.best.outer_std_score,
            )
        console.print("[bold green]Evolution complete.[/bold green]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve CLIP prompts with an LLM-guided genetic algorithm.")
    parser.add_argument("--clip-repo", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elite-size", type=int, default=6)
    parser.add_argument("--children-per-generation", type=int, default=18)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.45)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default="cache/prompt_evolution")
    parser.add_argument("--no-mps", action="store_true")
    parser.add_argument("--allow-word-prompts", action="store_true")
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--two-phase", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--slot-based", action="store_true")
    parser.add_argument("--phase1-generations", type=int, default=4)
    parser.add_argument("--phase1-population-size", type=int, default=24)
    parser.add_argument("--phase1-outer-folds", type=int, default=3)
    parser.add_argument("--phase1-inner-folds", type=int, default=2)
    parser.add_argument("--phase2-generations", type=int, default=6)
    parser.add_argument("--phase2-population-size", type=int, default=16)
    parser.add_argument("--phase2-outer-folds", type=int, default=5)
    parser.add_argument("--phase2-inner-folds", type=int, default=3)
    parser.add_argument("--phase2-seed-top-k", type=int, default=12)
    parser.add_argument("--phase2-mode", type=str, default="same", choices=["same", "ensemble", "single"])
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--ensemble-seed-pool-size", type=int, default=12)
    parser.add_argument("--slot-mutation-rate", type=float, default=0.35)
    parser.add_argument("--llm-mutation-batches-per-generation", type=int, default=4)
    parser.add_argument("--max-llm-candidates-per-generation", type=int, default=24)
    parser.add_argument("--max-novel-candidates-kept", type=int, default=12)
    return parser.parse_args()


def validate_evolution_config(cfg: EvolutionConfig) -> None:
    if cfg.ensemble and cfg.slot_based:
        raise ValueError(
            "Cannot use --ensemble and --slot-based together. "
            "Choose one candidate representation."
        )
    if not cfg.two_phase and cfg.phase2_mode != "same":
        raise ValueError("--phase2-mode requires --two-phase.")
    if cfg.two_phase and cfg.phase2_mode != "same" and (cfg.ensemble or cfg.slot_based):
        raise ValueError(
            "--phase2-mode single/ensemble currently requires the base run mode "
            "to be free-form single-template search."
        )


def main() -> None:
    args = parse_args()
    clip_architecture = get_foundation_model(args.clip_repo, FoundationModelFamily.CLIP)
    clip_cfg = FoundationModelConfig(model=clip_architecture, family=FoundationModelFamily.CLIP)
    evolution_cfg = EvolutionConfig(
        llm_model=args.llm_model,
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        children_per_generation=args.children_per_generation,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        mutation_rate=args.mutation_rate,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        random_seed=args.random_seed,
        cache_dir=args.cache_dir,
        use_mps=not args.no_mps,
        digit_only=not args.allow_word_prompts,
        text_batch_size=args.text_batch_size,
        two_phase=args.two_phase,
        ensemble=args.ensemble,
        slot_based=args.slot_based,
        phase1_generations=args.phase1_generations,
        phase1_population_size=args.phase1_population_size,
        phase1_outer_folds=args.phase1_outer_folds,
        phase1_inner_folds=args.phase1_inner_folds,
        phase2_generations=args.phase2_generations,
        phase2_population_size=args.phase2_population_size,
        phase2_outer_folds=args.phase2_outer_folds,
        phase2_inner_folds=args.phase2_inner_folds,
        phase2_seed_top_k=args.phase2_seed_top_k,
        phase2_mode=args.phase2_mode,
        ensemble_size=args.ensemble_size,
        ensemble_seed_pool_size=args.ensemble_seed_pool_size,
        slot_mutation_rate=args.slot_mutation_rate,
        llm_mutation_batches_per_generation=args.llm_mutation_batches_per_generation,
        max_llm_candidates_per_generation=args.max_llm_candidates_per_generation,
        max_novel_candidates_kept=args.max_novel_candidates_kept,
    )
    validate_evolution_config(evolution_cfg)
    evolve_prompts(evolution_cfg, clip_cfg)


if __name__ == "__main__":
    main()
