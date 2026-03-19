from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


log = logging.getLogger(__name__)


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


def save_json_artifact(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=get_llm_torch_dtype(device),
        )
        self.model.to(device)
        self.model.eval()

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
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
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_items(
        self,
        system_prompt: str,
        user_prompt: str,
        extractor: Callable[[str], list[str]],
        *,
        shuffle_rng=None,
    ) -> list[str]:
        text = self.generate_text(system_prompt, user_prompt)
        items = extractor(text)
        if shuffle_rng is not None:
            shuffle_rng.shuffle(items)
        return items
