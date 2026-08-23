"""Fresh matched-budget retrieval for the natural query-facet benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import write_csv, write_json
from experiments.paper2_7_query_graph.run_natural_facet_validation import (
    _graph,
    _llm_map,
    _syntax_labels,
)
from pra_hf.natural_query_facets import NaturalFacetAnnotation, align_subquestions_to_units, interleaving_statistics
from pra_hf.query_graph_cluster import connected_components, deterministic_kmeans


CONDITIONS = (
    "global_semantic",
    "syntax_semantic",
    "embedding_kmeans_semantic",
    "graph_cc_semantic",
    "llm_semantic",
    "lexical_semantic_hybrid",
    "graph_cc_hybrid",
    "llm_hybrid",
)
_WORD = re.compile(r"\w+", re.UNICODE)


def _load_annotations(path: Path):
    return [
        NaturalFacetAnnotation.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_records(twowiki: Path, musique: Path):
    wiki = {str(row["_id"]): row for row in json.loads(twowiki.read_text(encoding="utf-8"))}
    music = {}
    with musique.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            music[str(row["id"])] = row
    return wiki, music


def _documents(annotation, wiki, music):
    if annotation.dataset == "2wikimultihopqa":
        row = wiki[annotation.example_id]
        positive_titles = {str(value[0]) for value in row["supporting_facts"]}
        documents = [f"{title}. {' '.join(sentences)}" for title, sentences in row["context"]]
        positives = {index for index, (title, _) in enumerate(row["context"]) if str(title) in positive_titles}
    else:
        row = music[annotation.example_id]
        paragraphs = row["paragraphs"]
        documents = [f"{value['title']}. {value['paragraph_text']}" for value in paragraphs]
        positives = {index for index, value in enumerate(paragraphs) if value["is_supporting"]}
    if not positives:
        raise ValueError(f"No supporting document for {annotation.dataset}/{annotation.example_id}")
    return documents, positives


def _facet_texts(annotation, labels):
    output = []
    for label in torch.unique(labels, sorted=True):
        text = " ".join(unit.text for unit, value in zip(annotation.units, labels.tolist()) if value == int(label))
        if text.strip():
            output.append(text)
    return output or [annotation.question]


class SentenceEncoder:
    def __init__(self, model_id: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def encode(self, texts, batch_size=32):
        output = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**encoded, return_dict=True).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            output.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(output)


def _lexical_scores(question, documents):
    query = Counter(_WORD.findall(question.casefold()))
    doc_terms = [set(_WORD.findall(text.casefold())) for text in documents]
    frequencies = Counter(term for terms in doc_terms for term in terms)
    weights = {term: math.log((len(documents) + 1) / (frequencies[term] + 1)) + 1 for term in query}
    denominator = sum(query[term] * weights[term] for term in query) or 1.0
    return torch.tensor([
        sum(query[term] * weights[term] for term in query if term in terms) / denominator
        for terms in doc_terms
    ])


def _semantic_scores(facets, documents):
    return (facets @ documents.T).max(0).values


def _normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-8)


def _metrics(scores, positives, budget):
    ranks = torch.argsort(scores, descending=True, stable=True).tolist()
    selected = ranks[: min(budget, len(ranks))]
    first = min((ranks.index(index) + 1 for index in positives), default=len(ranks) + 1)
    return {
        "evidence_recall": len(set(selected) & positives) / len(positives),
        "precision": len(set(selected) & positives) / len(selected),
        "mrr": 1.0 / first,
        "selected": selected,
    }


def _bootstrap(rows, left, right, dataset, seed=20260823, draws=5000):
    pairs = defaultdict(dict)
    for row in rows:
        if row["dataset"] == dataset:
            pairs[row["example_id"]][row["condition"]] = row["evidence_recall"]
    deltas = [value[left] - value[right] for value in pairs.values() if left in value and right in value]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(draws))
    return {
        "dataset": dataset,
        "left": left,
        "right": right,
        "examples": len(deltas),
        "mean_recall_delta": sum(deltas) / len(deltas),
        "ci95": [boot[int(0.025 * draws)], boot[int(0.975 * draws)]],
        "better": sum(value > 0 for value in deltas),
        "worse": sum(value < 0 for value in deltas),
        "unchanged": sum(value == 0 for value in deltas),
    }


def run(args):
    annotations = [row for row in _load_annotations(args.annotations) if row.split == "test"]
    feature_artifact = torch.load(args.query_features, map_location="cpu", weights_only=False)
    features = feature_artifact["features"]
    graph_policy = json.loads(args.graph_findings.read_text(encoding="utf-8"))["selected_graph_policy"]
    layer = int(graph_policy["layer"])
    llm = _llm_map(args.llm_predictions)
    wiki, music = _source_records(args.twowiki_dev, args.musique_dev)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    encoder = SentenceEncoder(args.retrieval_model, device)
    rows = []
    for index, annotation in enumerate(annotations, 1):
        hidden = features[(annotation.dataset, annotation.example_id)][layer]
        graph = _graph(hidden, annotation, top_k=int(graph_policy["top_k"]), threshold=float(graph_policy["threshold"]))
        labels = {
            "global": torch.zeros(len(annotation.units), dtype=torch.long),
            "syntax": _syntax_labels(annotation),
            "embedding_kmeans": deterministic_kmeans(hidden, max(1, min(6, round(math.sqrt(len(annotation.units) / 2.0))))).labels,
            "graph_cc": connected_components(graph).labels,
        }
        key = (annotation.dataset, annotation.example_id)
        if key in llm:
            labels["llm"] = align_subquestions_to_units(annotation, llm[key])
        documents, positives = _documents(annotation, wiki, music)
        names = list(labels)
        facet_groups = [
            list(llm[key]) if name == "llm" else _facet_texts(annotation, labels[name])
            for name in names
        ]
        flat_facets = [text for group in facet_groups for text in group]
        encoded = encoder.encode(documents + flat_facets)
        document_vectors = encoded[: len(documents)]
        cursor = len(documents)
        semantic = {}
        for name, group in zip(names, facet_groups):
            facet_vectors = encoded[cursor : cursor + len(group)]
            cursor += len(group)
            semantic[name] = _semantic_scores(facet_vectors, document_vectors)
        lexical = _lexical_scores(annotation.question, documents)
        condition_scores = {
            "global_semantic": semantic["global"],
            "syntax_semantic": semantic["syntax"],
            "embedding_kmeans_semantic": semantic["embedding_kmeans"],
            "graph_cc_semantic": semantic["graph_cc"],
            "lexical_semantic_hybrid": 0.5 * _normalize(lexical) + 0.5 * _normalize(semantic["global"]),
            "graph_cc_hybrid": 0.5 * _normalize(lexical) + 0.5 * _normalize(semantic["graph_cc"]),
        }
        if "llm" in semantic:
            condition_scores["llm_semantic"] = semantic["llm"]
            condition_scores["llm_hybrid"] = 0.5 * _normalize(lexical) + 0.5 * _normalize(semantic["llm"])
        strata = interleaving_statistics(annotation)
        for condition, scores in condition_scores.items():
            metric = _metrics(scores, positives, args.budget)
            rows.append({
                "query_graph_model": feature_artifact["model_id"],
                "retrieval_model": args.retrieval_model,
                "dataset": annotation.dataset,
                "example_id": annotation.example_id,
                "condition": condition,
                "candidate_documents": len(documents),
                "positive_documents": len(positives),
                "requested_document_budget": args.budget,
                "facet_count": len(annotation.source_facets),
                **strata,
                "evidence_recall": metric["evidence_recall"],
                "precision": metric["precision"],
                "mrr": metric["mrr"],
                "selected_document_ids": json.dumps(metric["selected"]),
            })
        if index % 20 == 0:
            print(f"retrieved {index}/{len(annotations)}", flush=True)
    summary = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for condition in CONDITIONS:
            group = [row for row in rows if row["dataset"] == dataset and row["condition"] == condition]
            if group:
                summary.append({"query_graph_model": feature_artifact["model_id"], "dataset": dataset, "condition": condition, "examples": len(group), **{key: sum(row[key] for row in group) / len(group) for key in ("evidence_recall", "precision", "mrr")}})
    comparisons = []
    for dataset in sorted({row["dataset"] for row in rows}):
        comparisons.extend([
            _bootstrap(rows, "graph_cc_semantic", "global_semantic", dataset),
            _bootstrap(rows, "graph_cc_semantic", "embedding_kmeans_semantic", dataset),
            _bootstrap(rows, "graph_cc_hybrid", "lexical_semantic_hybrid", dataset),
        ])
        if any(row["condition"] == "llm_semantic" and row["dataset"] == dataset for row in rows):
            comparisons.extend([
                _bootstrap(rows, "graph_cc_semantic", "llm_semantic", dataset),
                _bootstrap(rows, "graph_cc_hybrid", "llm_hybrid", dataset),
            ])
    gate = any(row["left"] == "graph_cc_semantic" and row["right"] == "global_semantic" and row["ci95"][0] > 0 for row in comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fresh_retrieval_rows.csv", rows)
    write_csv(args.output_dir / "fresh_retrieval_summary.csv", summary)
    findings = {
        "schema_version": "1.0",
        "query_graph_model": feature_artifact["model_id"],
        "retrieval_model": args.retrieval_model,
        "retrieval_model_revision": encoder.revision,
        "test_examples": len(annotations),
        "requested_document_budget": args.budget,
        "summary": summary,
        "paired_recall": comparisons,
        "native_kv_gate_pass": gate,
        "native_kv_gate_rule": "graph semantic recall minus global semantic recall has a paired bootstrap CI strictly above zero on at least one dataset",
    }
    write_json(args.output_dir / "fresh_retrieval_findings.json", findings)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, sorted({row["dataset"] for row in summary})):
        group = [row for row in summary if row["dataset"] == dataset]
        axis.bar(range(len(group)), [row["evidence_recall"] for row in group], color="#2878b5")
        axis.set_xticks(range(len(group)), [row["condition"].replace("_", "\n") for row in group], fontsize=7)
        axis.set_title(dataset)
        axis.set_ylabel("Evidence-document recall@4")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(args.output_dir / "fresh_retrieval.pdf")
    fig.savefig(args.output_dir / "fresh_retrieval.png", dpi=180)
    plt.close(fig)
    return findings


def parse_args():
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--graph-findings", type=Path, required=True)
    parser.add_argument("--llm-predictions", type=Path)
    parser.add_argument("--retrieval-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json")
    parser.add_argument("--musique-dev", type=Path, default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
