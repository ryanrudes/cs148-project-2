from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from digit_classifier.prompt_evolution_common import (
    PopulationStats,
    PromptLLM,
    SearchStats,
    save_json_artifact,
)
from digit_classifier.qwen_vl import (
    DEFAULT_QWEN_BATCH_SIZE,
    DEFAULT_QWEN_MAX_NEW_TOKENS,
    DEFAULT_QWEN_VL_REPO,
    QwenDatasetBundle,
    compute_qwen_eval_metrics,
    evaluate_qwen_zero_shot_prompt,
    list_qwen_vl_repos,
    load_qwen_dataset,
    load_qwen_vl_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)
console = Console()

EVOLUTION_SAMPLES_PER_CLASS = 50
HOLDOUT_SAMPLES_PER_CLASS = 50
DEFAULT_MUTATION_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_QWEN_EVOLUTION_CACHE_DIR = "cache/qwen_prompt_evolution"
DEFAULT_QWEN_SEED_PROMPTS = (
    "Identify the digit by focusing on the overall global shape of the figure in the image.",
    "Ignore background, color, and material details and classify the digit from the structure that is formed.",
    "Classify which numeral is formed by the arrangement shown in the image.",
    "Focus on the silhouette and outline of the arrangement to determine the digit.",
    "If the image is slightly ambiguous, choose the closest matching decimal digit based on geometry.",
    "Prioritize geometric structure over texture or style when deciding which digit is depicted.",
    "Recognize the single decimal digit shown by the visual arrangement in the image.",
    "Infer which digit is represented by the overall arrangement of visual elements in the image.",
)
OUTPUT_OVERRIDE_PATTERNS = (
    "reply with",
    "respond with",
    "output",
    "return only",
    "nothing else",
    "exactly one digit",
    "single digit character",
    "single character",
    "0-9",
    "0 to 9",
)


@dataclass(frozen=True)
class QwenPromptScore:
    prompt: str
    accuracy: float
    parse_rate: float
    invalid_count: int
    num_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "accuracy": self.accuracy,
            "parse_rate": self.parse_rate,
            "invalid_count": self.invalid_count,
            "num_samples": self.num_samples,
        }


@dataclass(frozen=True)
class QwenPopulationEvaluationResult:
    split_name: str
    scores: list[QwenPromptScore]
    cache_hits: int
    cache_misses: int

    @property
    def num_prompts(self) -> int:
        return len(self.scores)


@dataclass
class QwenEvolutionConfig:
    repo: str = DEFAULT_QWEN_VL_REPO
    device: str = "auto"
    llm_model: str = DEFAULT_MUTATION_LLM
    population_size: int = 8
    generations: int = 4
    elite_size: int = 3
    children_per_generation: int = 5
    random_seed: int = 0
    cache_dir: str = DEFAULT_QWEN_EVOLUTION_CACHE_DIR
    batch_size: int = DEFAULT_QWEN_BATCH_SIZE
    qwen_max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS
    llm_max_new_tokens: int = 220
    temperature: float = 0.9
    top_p: float = 0.95
    llm_mutation_batches_per_generation: int = 2
    mutation_batch_size: int = 6
    max_attempts_per_batch: int = 2
    max_llm_candidates_per_generation: int = 12


def canonicalize_qwen_candidate_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def validate_qwen_candidate_prompt(prompt: str) -> str:
    normalized = canonicalize_qwen_candidate_prompt(prompt)
    if len(normalized) < 20:
        raise ValueError("Prompt is too short to be a useful Qwen instruction")
    lowered = normalized.lower()
    if any(pattern in lowered for pattern in OUTPUT_OVERRIDE_PATTERNS):
        raise ValueError("Prompt attempts to override the fixed output-format contract")
    return normalized


def extract_qwen_instruction_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^[-*\d\.)\s]+", "", stripped)
        stripped = stripped.strip().strip('"').strip("'")
        if not stripped:
            continue
        try:
            normalized = validate_qwen_candidate_prompt(stripped)
        except ValueError:
            continue
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates


def rank_qwen_prompt_scores(scores: list[QwenPromptScore]) -> list[QwenPromptScore]:
    return sorted(scores, key=lambda score: (-score.accuracy, -score.parse_rate, score.prompt))


def build_qwen_evolution_system_prompt() -> str:
    return (
        "You generate concise instruction prompts for a vision-language model that classifies a single "
        "digit from an image. The framework already handles answer formatting and parsing. "
        "Do not include any reply-format instructions. Return only a plain newline-separated list "
        "of candidate instruction prompts with no commentary."
    )


def build_qwen_mutation_prompt(
    elites: list[QwenPromptScore],
    existing_prompts: set[str],
    batch_size: int,
) -> str:
    elite_lines = "\n".join(
        f"- {score.prompt} [accuracy={score.accuracy:.4f}, parse_rate={score.parse_rate:.4f}]"
        for score in elites
    )
    existing_preview = "\n".join(f"- {prompt}" for prompt in sorted(existing_prompts)[:80])
    return (
        f"Top-performing instruction prompts so far:\n{elite_lines}\n\n"
        "Generate improved instruction prompts for classifying a single decimal digit from an image with "
        "a vision-language model. Focus on instructions about global shape, geometry, structure, silhouette, "
        "and ignoring irrelevant style cues. The framework already enforces the response format, so do not "
        "mention how the model should answer.\n\n"
        f"Produce at least {batch_size} and at most {batch_size * 2} new prompts.\n"
        "Do not repeat any prompt from the existing set below.\n\n"
        f"Existing prompts:\n{existing_preview}"
    )


def build_qwen_prediction_cache_key(split_name: str, prompt: str) -> tuple[str, str]:
    return split_name, prompt


def _log_qwen_status(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def _truncate_qwen_prompt(prompt: str, *, max_chars: int = 72) -> str:
    normalized = canonicalize_qwen_candidate_prompt(prompt)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _create_qwen_population_progress(*, disable: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("hits {task.fields[cache_hits]} misses {task.fields[cache_misses]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=disable,
    )


def build_qwen_split_indices(labels: np.ndarray, random_seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(random_seed)
    evolution_indices: list[int] = []
    holdout_indices: list[int] = []
    label_array = np.asarray(labels, dtype=np.int64)

    for digit in range(10):
        digit_indices = np.flatnonzero(label_array == digit)
        required = EVOLUTION_SAMPLES_PER_CLASS + HOLDOUT_SAMPLES_PER_CLASS
        if len(digit_indices) < required:
            raise ValueError(
                f"Digit {digit} has only {len(digit_indices)} samples, but {required} are required"
            )
        shuffled = rng.permutation(digit_indices)
        evolution_indices.extend(int(index) for index in shuffled[:EVOLUTION_SAMPLES_PER_CLASS])
        holdout_indices.extend(
            int(index)
            for index in shuffled[
                EVOLUTION_SAMPLES_PER_CLASS : EVOLUTION_SAMPLES_PER_CLASS + HOLDOUT_SAMPLES_PER_CLASS
            ]
        )

    rng.shuffle(evolution_indices)
    rng.shuffle(holdout_indices)
    return {
        "evolution_indices": evolution_indices,
        "holdout_indices": holdout_indices,
    }


def get_or_create_qwen_split_plan(
    dataset_bundle: QwenDatasetBundle,
    *,
    cache_dir: str | Path,
    random_seed: int,
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    split_path = cache_path / f"mnist_split_seed_{random_seed}.json"
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("dataset_key") == dataset_bundle.dataset_key
            and payload.get("num_samples") == dataset_bundle.num_samples
        ):
            return payload

    split_indices = build_qwen_split_indices(dataset_bundle.labels, random_seed)
    payload = {
        "dataset_key": dataset_bundle.dataset_key,
        "source_path": dataset_bundle.source_path,
        "num_samples": dataset_bundle.num_samples,
        "random_seed": random_seed,
        "evolution_indices": split_indices["evolution_indices"],
        "holdout_indices": split_indices["holdout_indices"],
    }
    save_json_artifact(split_path, payload)
    return payload


def evaluate_qwen_prompt_population(
    prompts: list[str],
    *,
    model: Any,
    processor: Any,
    device,
    dataset_bundle: QwenDatasetBundle,
    split_name: str,
    indices: list[int],
    prediction_cache: dict[tuple[str, str], list[int | None]],
    batch_size: int,
    max_new_tokens: int,
    show_progress: bool = False,
) -> QwenPopulationEvaluationResult:
    scores: list[QwenPromptScore] = []
    label_subset = dataset_bundle.labels[np.asarray(indices, dtype=np.int64)]
    cache_hits = 0
    cache_misses = 0

    progress_cm = _create_qwen_population_progress(disable=not show_progress)
    with progress_cm as progress:
        task_id = progress.add_task(
            f"Scoring {split_name} prompts",
            total=len(prompts),
            cache_hits=0,
            cache_misses=0,
        )
        for prompt in prompts:
            cache_key = build_qwen_prediction_cache_key(split_name, prompt)
            predictions = prediction_cache.get(cache_key)
            prompt_label = _truncate_qwen_prompt(prompt)
            if predictions is None:
                cache_misses += 1
                progress.update(
                    task_id,
                    description=f"Scoring {split_name} prompt: {prompt_label}",
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                )
                _, predictions, _ = evaluate_qwen_zero_shot_prompt(
                    model,
                    processor,
                    device,
                    dataset_bundle,
                    prompt,
                    indices=indices,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                )
                prediction_cache[cache_key] = predictions
            else:
                cache_hits += 1
                progress.update(
                    task_id,
                    description=f"Scoring {split_name} prompt: {prompt_label} [cache]",
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                )
            metrics = compute_qwen_eval_metrics(predictions, label_subset)
            scores.append(
                QwenPromptScore(
                    prompt=prompt,
                    accuracy=metrics.accuracy,
                    parse_rate=metrics.parse_rate,
                    invalid_count=metrics.invalid_count,
                    num_samples=metrics.num_samples,
                )
            )
            progress.update(
                task_id,
                advance=1,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )

    return QwenPopulationEvaluationResult(
        split_name=split_name,
        scores=rank_qwen_prompt_scores(scores),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


def generate_qwen_lexical_proposals(
    elites: list[QwenPromptScore],
    target_size: int,
    rng: random.Random,
) -> list[str]:
    substitutions = [
        ("digit", "numeral"),
        ("numeral", "digit"),
        ("global shape", "overall shape"),
        ("overall shape", "global shape"),
        ("structure", "geometry"),
        ("geometry", "structure"),
        ("silhouette", "outline"),
        ("outline", "silhouette"),
        ("ignore", "de-emphasize"),
        ("de-emphasize", "ignore"),
    ]
    proposals: list[str] = []
    max_attempts = max(target_size * 6, 12)
    attempts = 0
    while len(proposals) < target_size * 3 and attempts < max_attempts:
        attempts += 1
        parent = rng.choice(elites).prompt
        old, new = rng.choice(substitutions)
        if old not in parent:
            continue
        mutated = canonicalize_qwen_candidate_prompt(parent.replace(old, new, 1))
        try:
            proposals.append(validate_qwen_candidate_prompt(mutated))
        except ValueError:
            continue
    return proposals


def collect_qwen_llm_proposals(
    llm: PromptLLM,
    elites: list[QwenPromptScore],
    existing_prompts: set[str],
    cfg: QwenEvolutionConfig,
    rng: random.Random,
    search_stats: SearchStats,
) -> list[str]:
    system_prompt = build_qwen_evolution_system_prompt()
    proposals: list[str] = []

    for batch_index in range(cfg.llm_mutation_batches_per_generation):
        if len(proposals) >= cfg.max_llm_candidates_per_generation:
            break
        remaining_budget = cfg.max_llm_candidates_per_generation - len(proposals)
        batch_size = max(1, min(cfg.mutation_batch_size, remaining_budget))
        user_prompt = build_qwen_mutation_prompt(elites, existing_prompts, batch_size)
        _log_qwen_status(
            f"Mutation batch {batch_index + 1}/{cfg.llm_mutation_batches_per_generation}: "
            f"requesting up to {batch_size} prompts from {cfg.llm_model}"
        )

        for _ in range(cfg.max_attempts_per_batch):
            search_stats.llm_batches_attempted += 1
            candidates = llm.generate_items(
                system_prompt,
                user_prompt,
                extract_qwen_instruction_candidates,
                shuffle_rng=rng,
            )
            search_stats.llm_raw_templates += len(candidates)
            if not candidates:
                _log_qwen_status("  Mutation LLM returned no valid candidates; retrying")
                continue
            proposals.extend(candidates[:remaining_budget])
            _log_qwen_status(f"  Collected {min(len(candidates), remaining_budget)} candidate prompts")
            break

    return proposals[: cfg.max_llm_candidates_per_generation]


def _deduplicate_qwen_candidates(
    proposals: list[str],
    *,
    existing_prompts: set[str],
    target_size: int,
    search_stats: SearchStats | None = None,
) -> list[str]:
    accepted: list[str] = []
    seen = set(existing_prompts)
    for proposal in proposals:
        try:
            normalized = validate_qwen_candidate_prompt(proposal)
        except ValueError:
            if search_stats is not None:
                search_stats.invalid_templates += 1
            continue
        if normalized in seen:
            if search_stats is not None:
                search_stats.exact_duplicates += 1
            continue
        accepted.append(normalized)
        seen.add(normalized)
        if search_stats is not None:
            search_stats.accepted_novel_templates += 1
        if len(accepted) >= target_size:
            break
    return accepted


def build_qwen_children(
    llm: PromptLLM,
    elites: list[QwenPromptScore],
    *,
    target_size: int,
    existing_population: list[str],
    cfg: QwenEvolutionConfig,
    rng: random.Random,
) -> tuple[list[str], SearchStats]:
    search_stats = SearchStats()
    existing_prompts = set(existing_population)

    llm_candidates = collect_qwen_llm_proposals(
        llm,
        elites,
        existing_prompts,
        cfg,
        rng,
        search_stats,
    )
    llm_children = _deduplicate_qwen_candidates(
        llm_candidates,
        existing_prompts=existing_prompts,
        target_size=target_size,
        search_stats=search_stats,
    )

    lexical_candidates = generate_qwen_lexical_proposals(
        elites,
        target_size=max(0, target_size - len(llm_children)),
        rng=rng,
    )
    lexical_children = _deduplicate_qwen_candidates(
        lexical_candidates,
        existing_prompts=existing_prompts | set(llm_children),
        target_size=max(0, target_size - len(llm_children)),
    )
    search_stats.lexical_mutations_used = len(lexical_children)
    return (llm_children + lexical_children)[:target_size], search_stats


def initialize_qwen_population(cfg: QwenEvolutionConfig) -> list[str]:
    population = []
    for prompt in DEFAULT_QWEN_SEED_PROMPTS:
        normalized = validate_qwen_candidate_prompt(prompt)
        if normalized not in population:
            population.append(normalized)
        if len(population) >= cfg.population_size:
            break
    return population


def ensure_qwen_population_size(
    population: list[str],
    *,
    target_size: int,
    rng: random.Random,
) -> list[str]:
    normalized_population = list(dict.fromkeys(population))
    if len(normalized_population) >= target_size:
        return normalized_population[:target_size]

    filler_pool = list(DEFAULT_QWEN_SEED_PROMPTS)
    while len(normalized_population) < target_size:
        base_prompt = canonicalize_qwen_candidate_prompt(rng.choice(filler_pool))
        if base_prompt not in normalized_population:
            normalized_population.append(base_prompt)
            continue
        alt_prompt = canonicalize_qwen_candidate_prompt(
            f"{base_prompt.rstrip('.')} Focus on the most informative global structure."
        )
        try:
            alt_prompt = validate_qwen_candidate_prompt(alt_prompt)
        except ValueError:
            continue
        if alt_prompt not in normalized_population:
            normalized_population.append(alt_prompt)
    return normalized_population[:target_size]


def render_qwen_generation(
    *,
    generation: int,
    evaluation: QwenPopulationEvaluationResult,
    population_stats: PopulationStats,
    search_stats: SearchStats,
) -> None:
    scores = evaluation.scores
    best = scores[0]
    console.rule(f"Qwen Generation {generation}")
    console.print(
        Panel(
            f"[bold]Best prompt[/bold]: {best.prompt}\n"
            f"[bold]Accuracy[/bold]: {best.accuracy:.4f}\n"
            f"[bold]Parse rate[/bold]: {best.parse_rate:.4f}\n"
            f"[bold]Population[/bold]: {population_stats.population_size}",
            title="Qwen Prompt Evolution",
            expand=False,
        )
    )

    table = Table(title="Top Prompts", show_lines=False)
    table.add_column("Rank", justify="right")
    table.add_column("Prompt")
    table.add_column("Accuracy", justify="right")
    table.add_column("Parse", justify="right")
    table.add_column("Invalid", justify="right")

    for rank, score in enumerate(scores[: min(8, len(scores))], start=1):
        table.add_row(
            str(rank),
            score.prompt,
            f"{score.accuracy:.4f}",
            f"{score.parse_rate:.4f}",
            str(score.invalid_count),
        )
    console.print(table)

    stats_table = Table(title="Generation Stats", show_lines=False)
    stats_table.add_column("Metric")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Population size", str(population_stats.population_size))
    stats_table.add_row("Diversity", str(population_stats.diversity))
    stats_table.add_row("Elites kept", str(population_stats.elite_count))
    stats_table.add_row("Children kept", str(population_stats.children_count))
    stats_table.add_row("Cache hits", str(evaluation.cache_hits))
    stats_table.add_row("Cache misses", str(evaluation.cache_misses))
    stats_table.add_row("LLM batches attempted", str(search_stats.llm_batches_attempted))
    stats_table.add_row("LLM raw prompts", str(search_stats.llm_raw_templates))
    stats_table.add_row("Accepted novel prompts", str(search_stats.accepted_novel_templates))
    stats_table.add_row("Lexical mutations used", str(search_stats.lexical_mutations_used))
    console.print(stats_table)


def save_qwen_generation_artifact(
    output_dir: str | Path,
    generation: int,
    evaluation: QwenPopulationEvaluationResult,
    *,
    population_stats: PopulationStats,
    search_stats: SearchStats,
) -> None:
    payload = {
        "generation": generation,
        "evaluation": {
            "split_name": evaluation.split_name,
            "num_prompts": evaluation.num_prompts,
            "cache_hits": evaluation.cache_hits,
            "cache_misses": evaluation.cache_misses,
        },
        "search_stats": asdict(search_stats),
        "population_stats": asdict(population_stats),
        "scores": [score.to_dict() for score in evaluation.scores],
    }
    save_json_artifact(Path(output_dir) / f"generation_{generation:03d}.json", payload)


def save_qwen_run_summary(
    output_dir: str | Path,
    *,
    cfg: QwenEvolutionConfig,
    split_plan: dict[str, Any],
    per_generation: list[dict[str, Any]],
    final_holdout: QwenPopulationEvaluationResult,
) -> None:
    payload = {
        "repo": cfg.repo,
        "llm_model": cfg.llm_model,
        "split_plan": split_plan,
        "generations": per_generation,
        "final_holdout_evaluation": {
            "split_name": final_holdout.split_name,
            "num_prompts": final_holdout.num_prompts,
            "cache_hits": final_holdout.cache_hits,
            "cache_misses": final_holdout.cache_misses,
        },
        "final_holdout_ranking": [score.to_dict() for score in final_holdout.scores],
        "best_prompt": final_holdout.scores[0].prompt if final_holdout.scores else None,
    }
    save_json_artifact(Path(output_dir) / "run_summary.json", payload)


def validate_qwen_evolution_config(cfg: QwenEvolutionConfig) -> None:
    if cfg.population_size < 1:
        raise ValueError("--population-size must be at least 1")
    if cfg.generations < 1:
        raise ValueError("--generations must be at least 1")
    if cfg.elite_size < 1:
        raise ValueError("--elite-size must be at least 1")
    if cfg.children_per_generation < 1:
        raise ValueError("--children-per-generation must be at least 1")
    if cfg.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if cfg.elite_size > cfg.population_size:
        raise ValueError("--elite-size cannot exceed --population-size")
    if cfg.children_per_generation > cfg.population_size:
        raise ValueError("--children-per-generation cannot exceed --population-size")


def run_qwen_prompt_evolution(cfg: QwenEvolutionConfig) -> dict[str, Any]:
    validate_qwen_evolution_config(cfg)
    rng = random.Random(cfg.random_seed)

    console.print("[bold]Starting Qwen prompt evolution[/bold]")
    model, processor, device = load_qwen_vl_model(cfg.repo, device=cfg.device)
    dataset_bundle = load_qwen_dataset("mnist")
    split_plan = get_or_create_qwen_split_plan(
        dataset_bundle,
        cache_dir=cfg.cache_dir,
        random_seed=cfg.random_seed,
    )
    llm = PromptLLM(
        model_name=cfg.llm_model,
        device=device.type,
        max_new_tokens=cfg.llm_max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    evolution_split_size = len(split_plan["evolution_indices"])
    holdout_split_size = len(split_plan["holdout_indices"])
    first_generation_image_evals = cfg.population_size * evolution_split_size
    first_generation_generate_calls = cfg.population_size * math.ceil(evolution_split_size / cfg.batch_size)
    console.print(
        Panel(
            f"[bold]Repo[/bold]: {cfg.repo}\n"
            f"[bold]Mutation LLM[/bold]: {cfg.llm_model}\n"
            f"[bold]Device[/bold]: {device.type}\n"
            f"[bold]Dataset[/bold]: {dataset_bundle.display_name} ({dataset_bundle.num_samples} samples)\n"
            f"[bold]Evolution split[/bold]: {evolution_split_size} samples\n"
            f"[bold]Holdout split[/bold]: {holdout_split_size} samples\n"
            f"[bold]Eval batch size[/bold]: {cfg.batch_size}\n"
            f"[bold]Population[/bold]: {cfg.population_size}\n"
            f"[bold]Generations[/bold]: {cfg.generations}\n"
            f"[bold]Elites[/bold]: {cfg.elite_size}\n"
            f"[bold]Children/gen[/bold]: {cfg.children_per_generation}\n"
            f"[bold]Gen 1 image evals[/bold]: {first_generation_image_evals}\n"
            f"[bold]Gen 1 generate() calls[/bold]: {first_generation_generate_calls}",
            title="Qwen Evolution Config",
            expand=False,
        )
    )

    population = ensure_qwen_population_size(
        initialize_qwen_population(cfg),
        target_size=cfg.population_size,
        rng=rng,
    )
    prediction_cache: dict[tuple[str, str], list[int | None]] = {}
    generation_records: list[dict[str, Any]] = []

    evolution_indices = split_plan["evolution_indices"]
    holdout_indices = split_plan["holdout_indices"]

    for generation in range(cfg.generations):
        _log_qwen_status(
            f"Generation {generation + 1}/{cfg.generations}: scoring {len(population)} prompts "
            f"on the evolution split ({len(evolution_indices)} samples)"
        )
        evaluation = evaluate_qwen_prompt_population(
            population,
            model=model,
            processor=processor,
            device=device,
            dataset_bundle=dataset_bundle,
            split_name="evolution",
            indices=evolution_indices,
            prediction_cache=prediction_cache,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.qwen_max_new_tokens,
            show_progress=True,
        )
        scores = evaluation.scores
        _log_qwen_status(
            f"Generation {generation + 1}/{cfg.generations}: "
            f"{evaluation.cache_hits} cache hits, {evaluation.cache_misses} new evaluations, "
            f"best accuracy {scores[0].accuracy:.4f}, parse {scores[0].parse_rate:.4f}"
        )

        if generation < cfg.generations - 1:
            elites = [score.prompt for score in scores[: cfg.elite_size]]
            _log_qwen_status(
                f"Generation {generation + 1}/{cfg.generations}: building {cfg.children_per_generation} children "
                f"from top {len(elites)} prompts"
            )
            children, search_stats = build_qwen_children(
                llm,
                scores[: cfg.elite_size],
                target_size=cfg.children_per_generation,
                existing_population=elites,
                cfg=cfg,
                rng=rng,
            )
            next_population = ensure_qwen_population_size(
                elites + children,
                target_size=cfg.population_size,
                rng=rng,
            )
            population_stats = PopulationStats(
                population_size=len(population),
                diversity=len(set(population)),
                elite_count=len(elites),
                children_count=len(next_population) - len(elites),
                llm_generated_children=max(0, len(children) - search_stats.lexical_mutations_used),
                lexical_mutation_children=search_stats.lexical_mutations_used,
                filler_candidates=max(0, len(next_population) - len(elites) - len(children)),
            )
            _log_qwen_status(
                f"Generation {generation + 1}/{cfg.generations}: next population has {len(next_population)} prompts "
                f"({population_stats.llm_generated_children} LLM, "
                f"{population_stats.lexical_mutation_children} lexical, "
                f"{population_stats.filler_candidates} filler)"
            )
            population = next_population
        else:
            search_stats = SearchStats()
            population_stats = PopulationStats(
                population_size=len(population),
                diversity=len(set(population)),
                elite_count=min(cfg.elite_size, len(population)),
                children_count=0,
            )

        render_qwen_generation(
            generation=generation,
            evaluation=evaluation,
            population_stats=population_stats,
            search_stats=search_stats,
        )
        save_qwen_generation_artifact(
            cfg.cache_dir,
            generation,
            evaluation,
            population_stats=population_stats,
            search_stats=search_stats,
        )
        generation_records.append(
            {
                "generation": generation,
                "evaluation": {
                    "split_name": evaluation.split_name,
                    "num_prompts": evaluation.num_prompts,
                    "cache_hits": evaluation.cache_hits,
                    "cache_misses": evaluation.cache_misses,
                },
                "search_stats": asdict(search_stats),
                "population_stats": asdict(population_stats),
                "scores": [score.to_dict() for score in scores],
            }
        )

    _log_qwen_status(
        f"Evaluating final population on the holdout split ({len(population)} prompts, "
        f"{len(holdout_indices)} samples)"
    )
    final_holdout_evaluation = evaluate_qwen_prompt_population(
        population,
        model=model,
        processor=processor,
        device=device,
        dataset_bundle=dataset_bundle,
        split_name="holdout",
        indices=holdout_indices,
        prediction_cache=prediction_cache,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.qwen_max_new_tokens,
        show_progress=True,
    )
    final_holdout_scores = final_holdout_evaluation.scores
    _log_qwen_status(
        f"Holdout evaluation complete: {final_holdout_evaluation.cache_hits} cache hits, "
        f"{final_holdout_evaluation.cache_misses} new evaluations"
    )

    holdout_table = Table(title="Final Holdout Ranking", show_lines=False)
    holdout_table.add_column("Rank", justify="right")
    holdout_table.add_column("Prompt")
    holdout_table.add_column("Accuracy", justify="right")
    holdout_table.add_column("Parse", justify="right")
    holdout_table.add_column("Invalid", justify="right")
    for rank, score in enumerate(final_holdout_scores[: min(8, len(final_holdout_scores))], start=1):
        holdout_table.add_row(
            str(rank),
            score.prompt,
            f"{score.accuracy:.4f}",
            f"{score.parse_rate:.4f}",
            str(score.invalid_count),
        )
    console.print(holdout_table)

    save_qwen_run_summary(
        cfg.cache_dir,
        cfg=cfg,
        split_plan=split_plan,
        per_generation=generation_records,
        final_holdout=final_holdout_evaluation,
    )

    best_prompt = final_holdout_scores[0].prompt if final_holdout_scores else None
    console.print("[bold green]Qwen prompt evolution complete.[/bold green]")
    return {
        "best_prompt": best_prompt,
        "holdout_ranking": [score.to_dict() for score in final_holdout_scores],
        "split_plan": split_plan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve Qwen2.5-VL prompts on MNIST-in-the-Wild.")
    parser.add_argument("--repo", type=str, default=DEFAULT_QWEN_VL_REPO, choices=list_qwen_vl_repos())
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--llm-model", type=str, default=DEFAULT_MUTATION_LLM)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--children-per-generation", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_QWEN_EVOLUTION_CACHE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = QwenEvolutionConfig(
        repo=args.repo,
        device=args.device,
        llm_model=args.llm_model,
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        children_per_generation=args.children_per_generation,
        random_seed=args.random_seed,
        cache_dir=args.cache_dir,
    )
    run_qwen_prompt_evolution(cfg)


if __name__ == "__main__":
    main()
