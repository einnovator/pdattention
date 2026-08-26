"""Deterministic, resumable Ollama ingestion sidecar for Paper 3.1."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


PROMPTS = {
    "generic": (
        "Summarize each source chunk faithfully. Preserve the main subject and central claim. "
        "Do not answer any future question and do not combine chunks."
    ),
    "retrieval": (
        "Create a retrieval address for each source chunk. Preserve names, aliases, entities, "
        "relations, events, dates, numbers, technical terms, and rare literal strings that could "
        "distinguish this chunk for an unknown future query. Do not add facts."
    ),
    "faceted": (
        "Create independent retrieval facets for each source chunk. Preserve entities and aliases, "
        "relations or events, terminology or rare strings, and dates or numbers. Facets are search "
        "addresses, not answer evidence. Do not add facts."
    ),
}
PROMPT_VERSION = "paper3.1-v4-exact-output-budget"


def _clean_generated_text(text: str) -> str:
    """Remove empty thinking wrappers and output labels from one address."""

    value = text.strip()
    if "</think>" in value:
        value = value.split("</think>", 1)[1].strip()
    for prefix in ("SUMMARY:", "Summary:", "summary:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    return value


@dataclass(frozen=True)
class GeneratedAddress:
    """One parsed model output plus measured generation accounting."""

    item_id: str
    summary: str
    facets: tuple[tuple[str, str], ...]
    prompt_eval_tokens: int
    eval_tokens: int
    generation_seconds: float
    raw_response_sha256: str
    generation_mode: str = "structured_json"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "summary": self.summary,
            "facets": [{"label": label, "text": text} for label, text in self.facets],
            "prompt_eval_tokens": self.prompt_eval_tokens,
            "eval_tokens": self.eval_tokens,
            "generation_seconds": self.generation_seconds,
            "raw_response_sha256": self.raw_response_sha256,
            "generation_mode": self.generation_mode,
        }

    @classmethod
    def from_dict(cls, row: Mapping) -> "GeneratedAddress":
        return cls(
            item_id=str(row["item_id"]),
            summary=str(row["summary"]),
            facets=tuple((str(item["label"]), str(item["text"])) for item in row.get("facets", ())),
            prompt_eval_tokens=int(row.get("prompt_eval_tokens", 0)),
            eval_tokens=int(row.get("eval_tokens", 0)),
            generation_seconds=float(row.get("generation_seconds", 0.0)),
            raw_response_sha256=str(row.get("raw_response_sha256", "")),
            generation_mode=str(row.get("generation_mode", "structured_json")),
        )


class JSONLGenerationCache:
    """Append-only cache keyed by model, prompt, geometry, and source hash."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, GeneratedAddress] = {}
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.rows[str(row["cache_key"])] = GeneratedAddress.from_dict(row)

    def get(self, key: str) -> GeneratedAddress | None:
        return self.rows.get(key)

    def put(self, key: str, value: GeneratedAddress) -> None:
        if key in self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"cache_key": key, **value.to_dict()}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.rows[key] = value


def _bind_item_id(value: GeneratedAddress, item_id: str) -> GeneratedAddress:
    """Reuse cached content while preserving the caller's logical identity."""

    return GeneratedAddress(
        item_id=item_id,
        summary=value.summary,
        facets=value.facets,
        prompt_eval_tokens=value.prompt_eval_tokens,
        eval_tokens=value.eval_tokens,
        generation_seconds=value.generation_seconds,
        raw_response_sha256=value.raw_response_sha256,
        generation_mode=value.generation_mode,
    )


def cache_key(
    *, model: str, prompt_id: str, token_budget: int, facet_count: int, source: str
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_id": prompt_id,
            "token_budget": token_budget,
            "facet_count": facet_count,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "prompt_version": PROMPT_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class OllamaClient:
    """Small stdlib client for local deterministic generation and embeddings."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _post(self, endpoint: str, payload: Mapping) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"Ollama request failed for {endpoint}: {error}") from error

    def model_info(self, model: str) -> dict:
        """Return Ollama's immutable model metadata for the experiment manifest."""

        return self._post("/api/show", {"model": model})

    def unload(self, model: str) -> None:
        """Release one resident model before a different runtime needs memory."""

        self._post(
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed a batch of routing addresses or one query with a frozen model."""

        if not texts:
            return []
        response = self._post("/api/embed", {"model": model, "input": texts, "truncate": True})
        return [[float(value) for value in row] for row in response["embeddings"]]

    def generate_json(
        self,
        model: str,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int,
    ) -> tuple[dict, dict]:
        """Generate one JSON object and return Ollama's timing/token counters."""

        started = time.perf_counter()
        response = self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "think": False,
                "options": {
                    "temperature": 0,
                    "seed": int(seed),
                    "num_ctx": 4096,
                    "num_predict": int(max_new_tokens),
                },
                "keep_alive": "10m",
            },
        )
        wall_seconds = time.perf_counter() - started
        raw = str(response.get("response", "")).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Model returned malformed JSON: {raw[:240]}") from error
        accounting = {
            "prompt_eval_tokens": int(response.get("prompt_eval_count", 0)),
            "eval_tokens": int(response.get("eval_count", 0)),
            "generation_seconds": wall_seconds,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }
        return parsed, accounting

    def generate_text(
        self,
        model: str,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int,
    ) -> tuple[str, dict]:
        """Generate unstructured text for a one-item lower-bound fallback."""

        started = time.perf_counter()
        response = self._post(
            "/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0,
                    "seed": int(seed),
                    "num_ctx": 4096,
                    "num_predict": int(max_new_tokens),
                    "stop": ["\nSOURCE:", "\nINPUT="],
                },
                "keep_alive": "10m",
            },
        )
        raw = str(response.get("response", "")).strip()
        accounting = {
            "prompt_eval_tokens": int(response.get("prompt_eval_count", 0)),
            "eval_tokens": int(response.get("eval_count", 0)),
            "generation_seconds": time.perf_counter() - started,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }
        return raw, accounting


def build_prompt(
    items: Iterable[tuple[str, str]],
    *,
    prompt_id: str,
    token_budget: int,
    facet_count: int,
) -> str:
    """Build a bounded batch prompt whose output remains aligned by item ID."""

    if prompt_id not in PROMPTS:
        raise ValueError(f"Unknown summary prompt: {prompt_id}")
    if token_budget <= 0 or facet_count <= 0:
        raise ValueError("Summary token budget and facet count must be positive.")
    item_rows = [{"id": item_id, "source": source} for item_id, source in items]
    per_facet = max(token_budget // facet_count, 1)
    return (
        f"/no_think\n{PROMPTS[prompt_id]}\n"
        f"Use at most {token_budget} output tokens per item in total. "
        f"Return exactly {facet_count} facet(s), each at most {per_facet} tokens. "
        "Return JSON only with this shape: "
        '{"summaries":[{"id":"...","summary":"...","facets":'
        '[{"label":"...","text":"..."}]}]}. '
        "Copy each input id exactly and return one output per input.\n"
        f"INPUT={json.dumps(item_rows, ensure_ascii=False)}"
    )


def parse_generated_batch(
    payload: Mapping,
    expected_ids: Iterable[str],
    accounting: Mapping,
) -> list[GeneratedAddress]:
    """Validate a model batch so summaries cannot drift across chunk identities."""

    expected = tuple(expected_ids)
    rows = payload.get("summaries")
    if not isinstance(rows, list):
        raise ValueError("Generated JSON requires a summaries list.")
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, Mapping)}
    if set(by_id) != set(expected):
        raise ValueError(f"Generated IDs do not match input IDs: expected={expected}, got={tuple(by_id)}")
    per_item_seconds = float(accounting["generation_seconds"]) / max(len(expected), 1)
    outputs = []
    for item_id in expected:
        row = by_id[item_id]
        summary = _clean_generated_text(str(row.get("summary", "")))
        facets = tuple(
            (
                str(facet.get("label", f"facet-{index}")),
                _clean_generated_text(str(facet.get("text", ""))),
            )
            for index, facet in enumerate(row.get("facets", ()))
            if isinstance(facet, Mapping) and str(facet.get("text", "")).strip()
        )
        if not summary and facets:
            summary = " ".join(text for _, text in facets)
        if not summary:
            raise ValueError(f"Generated summary is empty for {item_id}")
        outputs.append(
            GeneratedAddress(
                item_id=item_id,
                summary=summary,
                facets=facets,
                prompt_eval_tokens=int(accounting["prompt_eval_tokens"]) // max(len(expected), 1),
                eval_tokens=int(accounting["eval_tokens"]) // max(len(expected), 1),
                generation_seconds=per_item_seconds,
                raw_response_sha256=str(accounting["raw_response_sha256"]),
            )
        )
    return outputs


def generate_cached(
    client: OllamaClient,
    cache: JSONLGenerationCache,
    items: list[tuple[str, str]],
    *,
    model: str,
    prompt_id: str,
    token_budget: int,
    facet_count: int,
    seed: int,
    batch_size: int,
    structured_batches: bool = True,
) -> list[GeneratedAddress]:
    """Generate only cache misses and preserve caller order across resumptions."""

    keys = {
        item_id: cache_key(
            model=model,
            prompt_id=prompt_id,
            token_budget=token_budget,
            facet_count=facet_count,
            source=source,
        )
        for item_id, source in items
    }
    sources = dict(items)
    missing = []
    pending_keys = set()
    for item_id, source in items:
        key = keys[item_id]
        if cache.get(key) is None and key not in pending_keys:
            missing.append((item_id, source))
            pending_keys.add(key)
    if not structured_batches:
        for index, (item_id, source) in enumerate(missing, start=1):
            raw, accounting = client.generate_text(
                model,
                (
                    f"/no_think\n{PROMPTS[prompt_id]}\n"
                    f"Write only one summary of at most {token_budget} tokens.\n"
                    f"SOURCE:\n{source}\nSUMMARY:"
                ),
                seed=seed,
                max_new_tokens=token_budget,
            )
            raw = _clean_generated_text(raw)
            if not raw:
                raise ValueError(f"Unstructured generation is empty for {item_id}")
            output = GeneratedAddress(
                item_id=item_id,
                summary=raw,
                facets=(),
                prompt_eval_tokens=int(accounting["prompt_eval_tokens"]),
                eval_tokens=int(accounting["eval_tokens"]),
                generation_seconds=float(accounting["generation_seconds"]),
                raw_response_sha256=str(accounting["raw_response_sha256"]),
                generation_mode="plain_text",
            )
            cache.put(keys[item_id], output)
            if index % max(batch_size, 1) == 0 or index == len(missing):
                print(
                    f"[summary {model} {prompt_id} {index}/{len(missing)} misses]",
                    flush=True,
                )
        return [
            _bind_item_id(cache.get(keys[item_id]), item_id)  # type: ignore[arg-type]
            for item_id, _ in items
        ]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        aliased = [(str(index), source) for index, (_, source) in enumerate(batch)]
        aliases = {str(index): item_id for index, (item_id, _) in enumerate(batch)}
        prompt = build_prompt(
            aliased,
            prompt_id=prompt_id,
            token_budget=token_budget,
            facet_count=facet_count,
        )
        try:
            payload, accounting = client.generate_json(
                model,
                prompt,
                seed=seed,
                max_new_tokens=64
                + len(batch) * (token_budget + facet_count * 12 + 20),
            )
            parsed_outputs = parse_generated_batch(payload, aliases, accounting)
            outputs = [
                GeneratedAddress(
                    item_id=aliases[output.item_id],
                    summary=output.summary,
                    facets=output.facets,
                    prompt_eval_tokens=output.prompt_eval_tokens,
                    eval_tokens=output.eval_tokens,
                    generation_seconds=output.generation_seconds,
                    raw_response_sha256=output.raw_response_sha256,
                    generation_mode=output.generation_mode,
                )
                for output in parsed_outputs
            ]
        except (RuntimeError, ValueError):
            # A malformed batch is retried item-wise so one response cannot
            # misalign multiple persistent chunk addresses.
            outputs = []
            for item_id, source in batch:
                try:
                    payload, accounting = client.generate_json(
                        model,
                        build_prompt(
                            [("0", source)],
                            prompt_id=prompt_id,
                            token_budget=token_budget,
                            facet_count=facet_count,
                        ),
                        seed=seed,
                        max_new_tokens=64 + token_budget + facet_count * 12 + 20,
                    )
                    output = parse_generated_batch(payload, ["0"], accounting)[0]
                    outputs.append(
                        GeneratedAddress(
                            item_id=item_id,
                            summary=output.summary,
                            facets=output.facets,
                            prompt_eval_tokens=output.prompt_eval_tokens,
                            eval_tokens=output.eval_tokens,
                            generation_seconds=output.generation_seconds,
                            raw_response_sha256=output.raw_response_sha256,
                            generation_mode=output.generation_mode,
                        )
                    )
                except (RuntimeError, ValueError):
                    raw, accounting = client.generate_text(
                        model,
                        (
                            f"/no_think\n{PROMPTS[prompt_id]}\n"
                            f"Write only one summary of at most {token_budget} tokens.\n"
                            f"SOURCE:\n{source}\nSUMMARY:"
                        ),
                        seed=seed,
                        max_new_tokens=token_budget,
                    )
                    raw = _clean_generated_text(raw)
                    if not raw:
                        raise ValueError(f"Unstructured generation is empty for {item_id}")
                    outputs.append(
                        GeneratedAddress(
                            item_id=item_id,
                            summary=raw,
                            facets=(),
                            prompt_eval_tokens=int(accounting["prompt_eval_tokens"]),
                            eval_tokens=int(accounting["eval_tokens"]),
                            generation_seconds=float(accounting["generation_seconds"]),
                            raw_response_sha256=str(accounting["raw_response_sha256"]),
                            generation_mode="plain_text_fallback",
                        )
                    )
        for output in outputs:
            cache.put(keys[output.item_id], output)
        print(
            f"[summary {model} {prompt_id} {start + len(batch)}/{len(missing)} misses]",
            flush=True,
        )
    return [
        _bind_item_id(cache.get(keys[item_id]), item_id)  # type: ignore[arg-type]
        for item_id in sources
    ]
