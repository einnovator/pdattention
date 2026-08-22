"""Run the four-dataset Paper 2.6 retrieval-channel geometry extension.

The study reuses frozen Qwen3-0.6B attention-input states.  QASPER and
HotpotQA use Paper 2's feature bundle; 2Wiki and MuSiQue use Paper 2.5's
identity-frozen natural graph bundle.  All primary channels request four
32-token chunks and stop before native-K/V materialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_study import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    _records,
    _route_case,
    _synthetic_study,
    load_split_examples,
)
from pra_hf.channel_geometry import (  # noqa: E402
    jaccard,
    new_address_tokens,
    oracle_channel,
    select_observable_channel,
)
from pra_hf.hybrid_discovery import (  # noqa: E402
    HybridDiscoveryPolicy,
    TokenNativeIndex,
    _normalize_piece,
)
from pra_hf.iterative import IterativeGistRouter  # noqa: E402
from pra_hf.natural_reasoning_graph import load_2wiki, load_musique  # noqa: E402
from pra_torch.hf import QUERY_QUESTION_EXPONENTIAL, aggregate_query_states  # noqa: E402


CHANNELS = {
    "gist": ("B0_gist", "gist_only"),
    "exact": ("B2_exact", "token_exact"),
    "bm25": ("B1_bm25", "bm25"),
    "approx": ("B4_approx", "token_approx"),
    "hybrid": ("H2_token_semantic", "token_semantic_rerank"),
    "iterative_hybrid": ("H5_iterative_hybrid", "iterative_hybrid"),
}
DATASETS = ("qasper", "hotpotqa", "2wikimultihopqa", "musique")
LABELS = {
    "qasper": "QASPER",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki",
    "musique": "MuSiQue",
}
WEIGHTS = (0.6, 0.1)  # Frozen by the preceding validation-only Paper 2.6 study.


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _natural_feature(row: dict) -> dict:
    """Adapt Paper 2.5 token states to Paper 2.6's one-mean-gist schema."""
    spans = [tuple(map(int, span)) for span in row["local_spans"]]
    hidden = row["token_hidden"].float()
    gists = torch.stack([hidden[start:end].mean(0) for start, end in spans])
    gold_spans = [tuple(map(int, span)) for span in row["node_token_spans"].values()]
    positive = torch.tensor(
        [any(_overlap(span, gold) for gold in gold_spans) for span in spans],
        dtype=torch.bool,
    )
    query = aggregate_query_states(
        row["query_hidden_states"].float().unsqueeze(0),
        QUERY_QUESTION_EXPONENTIAL,
        half_life=2.0,
        token_spans=[tuple(map(int, row["question_span"]))],
    )[0]
    if not bool(positive.any()):
        raise ValueError(f"Natural example has no mapped evidence: {row['example_id']}")
    return {
        "split": row["partition"],
        "dataset": row["dataset"],
        "example_id": row["example_id"],
        "chunk_spans": spans,
        "memory_gists": gists,
        "positive_mask": positive,
        "queries": {"question_exp_h2.0": query},
        "evidence_spans": gold_spans,
        "source_tokens": int(row["source_tokens"]),
        "annotated_hops": int(row["annotated_hops"]),
        "graph_type": row["graph_type"],
        "nodes": row["nodes"],
    }


def _load_cases(args) -> list[tuple[dict, dict]]:
    cases: list[tuple[dict, dict]] = []
    for split, (offset, count) in {"validation": (0, 8), "test": (8, 16)}.items():
        examples = {
            (row["dataset"], row["id"]): row
            for row in load_split_examples(args.cache_dir, count, offset, args.seed)
        }
        features = torch.load(
            args.paper2_feature_dir / f"router_features_{split}.pt",
            map_location="cpu",
            weights_only=False,
        )
        cases.extend((feature, examples[(feature["dataset"], feature["example_id"])]) for feature in features)

    selected = torch.load(
        args.natural_features, map_location="cpu", weights_only=False, mmap=True
    )
    selected_ids = {row["example_id"] for row in selected}
    natural = load_musique(args.musique_dev) + load_2wiki(args.twowiki_dev)
    natural_by_id = {row.example_id: row for row in natural if row.example_id in selected_ids}
    if set(natural_by_id) != selected_ids:
        raise ValueError("Natural feature identities do not match local labelled datasets.")
    for row in selected:
        example = natural_by_id[row["example_id"]]
        cases.append(
            (
                _natural_feature(row),
                {
                    "dataset": example.dataset,
                    "id": example.example_id,
                    "source": example.source,
                    "question": example.question,
                    "answer": example.answer,
                    "annotated_hops": example.annotated_hops,
                    "graph_type": example.graph_type,
                    "nodes": example.nodes,
                },
            )
        )
    return cases


def _pieces(tokenizer, ids) -> set[str]:
    return {
        value
        for value in (_normalize_piece(str(piece)) for piece in tokenizer.convert_ids_to_tokens(list(ids)))
        if value
    }


def _gap(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else (ordered[0] if ordered else 0.0)


def _root_geometry(tokenizer, feature: dict, example: dict) -> tuple[dict, dict[str, list]]:
    index, positives = _records(feature, example)
    token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
    query = feature["queries"]["question_exp_h2.0"].float()
    semantic, _ = IterativeGistRouter(index)._scores(query.unsqueeze(0))
    query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
    modes = {
        "gist": "gist_only",
        "exact": "token_exact",
        "bm25": "bm25",
        "approx": "token_approx",
        "hybrid": "token_semantic_rerank",
    }
    candidates = {
        channel: token_index.score(
            query_ids,
            semantic[0],
            tokenizer,
            HybridDiscoveryPolicy(mode=mode, semantic_weight=WEIGHTS[0], token_weight=1-WEIGHTS[0]),
            hop=1,
            parent_id="__root__",
        )
        for channel, mode in modes.items()
    }
    positive_rows = {index.chunk_ids.index(identity) for identity in positives}
    ranks = {
        channel: min((rows[row].rank for row in positive_rows), default=len(rows) + 1)
        for channel, rows in candidates.items()
    }
    top_ids, top_scores, score_gaps = {}, {}, {}
    for channel, rows in candidates.items():
        ordered = sorted(range(len(rows)), key=lambda value: (-rows[value].selected_score, value))
        top_ids[channel] = index.chunk_ids[ordered[0]]
        top_scores[channel] = rows[ordered[0]].selected_score
        score_gaps[channel] = _gap([row.selected_score for row in rows])

    source_ids = tokenizer(example["source"], add_special_tokens=False).input_ids
    query_tokens = _pieces(tokenizer, query_ids)
    evidence_tokens = set().union(
        *(
            _pieces(tokenizer, source_ids[start:end])
            for (start, end), flag in zip(feature["chunk_spans"], feature["positive_mask"])
            if bool(flag)
        )
    )
    rare_cutoff = 0.75 * max(token_index.idf.values(), default=0.0)
    rare_query = {token for token in query_tokens if token_index.idf.get(token, 0.0) >= rare_cutoff}
    idf_total = sum(token_index.idf.get(token, 0.0) for token in query_tokens)
    idf_hit = sum(token_index.idf.get(token, 0.0) for token in query_tokens & evidence_tokens)
    capitalized = {
        word.casefold() for word in re.findall(r"\b[A-Z][\w-]+\b", example["question"])[1:]
    }
    evidence_text = tokenizer.decode(
        [token for (start, end), flag in zip(feature["chunk_spans"], feature["positive_mask"])
         if bool(flag) for token in source_ids[start:end]]
    ).casefold()
    positive_spans = [
        tuple(map(int, span))
        for span, flag in zip(feature["chunk_spans"], feature["positive_mask"])
        if bool(flag)
    ]
    gaps = [max(0, right[0] - left[1]) for left, right in zip(positive_spans, positive_spans[1:])]
    total_tokens = sum(end - start for start, end in positive_spans)
    extent = max(end for _, end in positive_spans) - min(start for start, _ in positive_spans)
    answer_tokens = _pieces(tokenizer, tokenizer(str(example.get("answer", "")), add_special_tokens=False).input_ids)
    observable = {
        "query_rare_fraction": len(rare_query) / max(len(query_tokens), 1),
        "exact_top_score": top_scores["exact"],
        "bm25_score_gap": score_gaps["bm25"],
        "semantic_score_gap": score_gaps["gist"],
        "channel_disagreement": len(set(top_ids.values())),
    }
    explanatory = {
        "query_evidence_overlap": len(query_tokens & evidence_tokens) / max(len(query_tokens), 1),
        "rare_token_overlap": len(rare_query & evidence_tokens) / max(len(rare_query), 1),
        "idf_weighted_overlap": idf_hit / max(idf_total, 1e-12),
        "named_entity_overlap": sum(name in evidence_text for name in capitalized) / max(len(capitalized), 1),
        "answer_overlap": len(answer_tokens & evidence_tokens) / max(len(answer_tokens), 1),
        "evidence_regions": len(positive_spans),
        "evidence_documents": len(example.get("nodes", example.get("evidence", positive_spans))),
        "evidence_tokens": total_tokens,
        "maximum_evidence_span": max((end - start for start, end in positive_spans), default=0),
        "evidence_gap": sum(gaps) / max(len(gaps), 1),
        "chain_depth": int(example.get("annotated_hops", feature.get("annotated_hops", 1))),
        "evidence_compactness": total_tokens / max(extent, 1),
    }
    rank_features = {f"{channel}_root_rank": rank for channel, rank in ranks.items()}
    score_features = {
        **{f"{channel}_top_score": value for channel, value in top_scores.items()},
        **{f"{channel}_score_gap": value for channel, value in score_gaps.items()},
    }
    return {**observable, **explanatory, **rank_features, **score_features}, {
        "index": index,
        "token_index": token_index,
        "source_ids": source_ids,
        "query_tokens": query_tokens,
        "evidence_tokens": evidence_tokens,
        "positives": positives,
        "top_ids": top_ids,
    }


def _bootstrap_ci(values: list[float], rng: random.Random, draws: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def _summaries(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for dataset in DATASETS:
        for channel in CHANNELS:
            group = [row for row in rows if row["split"] == "test" and row["dataset"] == dataset and row["channel"] == channel]
            recalls = [float(row["evidence_recall"]) for row in group]
            precisions = [float(row["precision"]) for row in group]
            rlo, rhi = _bootstrap_ci(recalls, rng)
            plo, phi = _bootstrap_ci(precisions, rng)
            output.append({
                "dataset": dataset, "channel": channel, "examples": len(group),
                "evidence_recall": sum(recalls) / max(len(recalls), 1),
                "recall_ci95_low": rlo, "recall_ci95_high": rhi,
                "precision": sum(precisions) / max(len(precisions), 1),
                "precision_ci95_low": plo, "precision_ci95_high": phi,
                "mrr": sum(float(row["mrr"]) for row in group) / max(len(group), 1),
                "complete_path": sum(float(row["path_completion"]) for row in group) / max(len(group), 1),
                "mean_requested_chunks": sum(float(row["requested_chunks"]) for row in group) / max(len(group), 1),
                "mean_search_comparisons": sum(float(row["semantic_comparisons"])+float(row["token_comparisons"]) for row in group) / max(len(group), 1),
            })
    return output


def _plots(summary, geometry, advantages, oracle_rows, overlap_rows, new_rows, selector_rows, wrong_rows, output):
    colors = ["#345995", "#d1495b", "#00798c", "#edae49", "#5f6f52", "#7a5195"]
    for metric, stem, ylabel in (("evidence_recall", "channel_recall", "Evidence recall"), ("precision", "channel_precision", "Evidence precision")):
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        width = 0.12
        for ci, channel in enumerate(CHANNELS):
            vals = [next(row[metric] for row in summary if row["dataset"] == ds and row["channel"] == channel) for ds in DATASETS]
            ax.bar([i+(ci-2.5)*width for i in range(4)], vals, width, label=channel.replace("_", " "), color=colors[ci])
        ax.set(xticks=range(4), xticklabels=[LABELS[d] for d in DATASETS], ylabel=ylabel, ylim=(0, 1.02))
        ax.grid(axis="y", alpha=.25); ax.legend(ncol=3, fontsize=8)
        fig.savefig(output/f"{stem}.png", dpi=180); fig.savefig(output/f"{stem}.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    for dataset in DATASETS:
        group=[r for r in summary if r["dataset"]==dataset]
        ax.scatter([r["evidence_recall"] for r in group],[r["precision"] for r in group],label=LABELS[dataset],s=55)
    ax.set(xlabel="Evidence recall",ylabel="Evidence precision",xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.25); ax.legend()
    fig.savefig(output/"channel_precision_recall.png",dpi=180); fig.savefig(output/"channel_precision_recall.pdf"); plt.close(fig)
    fig, ax=plt.subplots(figsize=(7.5,4.5),constrained_layout=True)
    means=[sum(r["headroom"] for r in oracle_rows if r["dataset"]==d)/max(sum(r["dataset"]==d for r in oracle_rows),1) for d in DATASETS]
    ax.bar([LABELS[d] for d in DATASETS],means,color="#00798c"); ax.set(ylabel="Oracle-channel recall headroom",ylim=(0,1)); ax.grid(axis="y",alpha=.25)
    fig.savefig(output/"channel_oracle_headroom.png",dpi=180); fig.savefig(output/"channel_oracle_headroom.pdf"); plt.close(fig)
    scatter_specs=[("query_evidence_overlap","delta_exact_gist","Lexical overlap","Exact - semantic recall","lexical_advantage"),("evidence_gap","delta_hybrid_gist","Mean evidence gap (tokens)","Hybrid - semantic recall","dispersion_advantage"),("channel_disagreement","oracle_headroom","Distinct root choices","Oracle headroom","disagreement_headroom")]
    for x,y,xlabel,ylabel,stem in scatter_specs:
        source=advantages
        fig,ax=plt.subplots(figsize=(6.5,4.8),constrained_layout=True)
        for d in DATASETS:
            group=[r for r in source if r["dataset"]==d and r["split"]=="test"]
            ax.scatter([r[x] for r in group],[r[y] for r in group],label=LABELS[d],alpha=.72)
        ax.set(xlabel=xlabel,ylabel=ylabel); ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.savefig(output/f"{stem}.png",dpi=180); fig.savefig(output/f"{stem}.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.6),constrained_layout=True)
    labels=[]; vals=[]
    for d in DATASETS:
        for flag in (0,1):
            group=[r for r in new_rows if r["dataset"]==d and r["split"]=="test" and r["new_address_observed"]==flag]
            labels.append(f"{LABELS[d]}\n{'address' if flag else 'none'}"); vals.append(sum(r["iterative_gain"] for r in group)/max(len(group),1))
    ax.bar(range(len(vals)),vals,color=["#9aa0a6" if i%2==0 else "#edae49" for i in range(len(vals))])
    ax.set(xticks=range(len(vals)), ylabel="Iterative - static hybrid recall")
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(0,color="black",lw=.8)
    fig.savefig(output/"new_address_iterative_gain.png",dpi=180); fig.savefig(output/"new_address_iterative_gain.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    perturb=sorted({r["perturbation"] for r in wrong_rows}); channels=["exact","approx","iterative_hybrid"]
    for i,ch in enumerate(channels):
        vals=[next((r["evidence_recall"] for r in wrong_rows if r["perturbation"]==p and r["channel"]==ch),0) for p in perturb]
        ax.plot(perturb,vals,marker="o",label=ch.replace("_"," "))
    ax.set(ylabel="Target recall",ylim=(0,1.03)); ax.grid(alpha=.25); ax.legend()
    fig.savefig(output/"wrong_reference_robustness.png",dpi=180); fig.savefig(output/"wrong_reference_robustness.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,5),constrained_layout=True)
    for policy,marker in (("best_fixed","o"),("hybrid","s"),("adaptive","^"),("oracle","*")):
        group=[r for r in selector_rows if r["policy"]==policy]
        ax.scatter([r["recall"] for r in group],[r["precision"] for r in group],label=policy.replace("_"," "),marker=marker,s=90)
    ax.set(xlabel="Evidence recall",ylabel="Evidence precision",xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.25); ax.legend()
    fig.savefig(output/"selector_frontier.png",dpi=180); fig.savefig(output/"selector_frontier.pdf"); plt.close(fig)


def run(args) -> dict:
    random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only)
    cases = _load_cases(args)
    rows, geometry_rows = [], []
    traces = {}
    for case_index, (feature, example) in enumerate(cases, 1):
        base_geometry, trace = _root_geometry(tokenizer, feature, example)
        key=(feature["split"],feature["dataset"],feature["example_id"]); traces[key]=trace
        geometry_rows.append({"split":feature["split"],"dataset":feature["dataset"],"example_id":feature["example_id"],**base_geometry})
        for channel,(condition,mode) in CHANNELS.items():
            row,_ = _route_case(tokenizer,feature,example,condition,mode,WEIGHTS,args.budget)
            row["channel"]=channel; rows.append(row)
        print(f"[channel {case_index}/{len(cases)}] {feature['dataset']} {feature['example_id']}",flush=True)

    by_case=defaultdict(dict)
    for row in rows: by_case[(row["split"],row["dataset"],row["example_id"])][row["channel"]]=row
    geometry_by_case={(r["split"],r["dataset"],r["example_id"]):r for r in geometry_rows}
    advantages=[]; overlap_rows=[]; new_rows=[]
    for key, outcomes in by_case.items():
        recalls={c:float(outcomes[c]["evidence_recall"]) for c in CHANNELS}
        _,oracle_recall=oracle_channel(recalls)
        g=geometry_by_case[key]
        advantages.append({"split":key[0],"dataset":key[1],"example_id":key[2],**{k:v for k,v in g.items() if k not in {"split","dataset","example_id"}},"delta_exact_gist":recalls["exact"]-recalls["gist"],"delta_bm25_gist":recalls["bm25"]-recalls["gist"],"delta_approx_gist":recalls["approx"]-recalls["gist"],"delta_hybrid_gist":recalls["hybrid"]-recalls["gist"],"delta_hybrid_best_single":recalls["hybrid"]-max(recalls[c] for c in ("gist","exact","bm25","approx")),"oracle_recall":oracle_recall})
        for left_i,left in enumerate(CHANNELS):
            ls=outcomes[left]["selected_chunk_ids"].split("|")
            for right in list(CHANNELS)[left_i+1:]:
                rs=outcomes[right]["selected_chunk_ids"].split("|")
                overlap_rows.append({"split":key[0],"dataset":key[1],"example_id":key[2],"left_channel":left,"right_channel":right,"selected_jaccard":jaccard(ls,rs),"shared_gold_chunks":len((set(ls)&set(rs))&set(outcomes[left]["positive_chunk_ids"].split("|")))})
        iterative=outcomes["iterative_hybrid"]
        pairs=[value.rsplit("@",1) for value in iterative["selected_hop_pairs"].split("|") if "@" in value]
        hop1={identity for identity,hop in pairs if hop=="1"}
        trace=traces[key]; idx={identity:i for i,identity in enumerate(trace["index"].chunk_ids)}
        first=set().union(*(_pieces(tokenizer,trace["source_ids"][trace["index"].records[idx[i]][1].token_start:trace["index"].records[idx[i]][1].token_end]) for i in hop1),set())
        remaining=set(trace["positives"])-hop1
        later=set().union(*(_pieces(tokenizer,trace["source_ids"][trace["index"].records[idx[i]][1].token_start:trace["index"].records[idx[i]][1].token_end]) for i in remaining),set())
        addresses=new_address_tokens(trace["query_tokens"],first,later)
        rare_cutoff = 0.75 * max(trace["token_index"].idf.values(), default=0.0)
        addresses = {
            token
            for token in addresses
            if trace["token_index"].idf.get(token, 0.0) >= rare_cutoff
        }
        flag=int(bool(addresses))
        g["new_address_observed"]=flag
        new_rows.append({"split":key[0],"dataset":key[1],"example_id":key[2],"new_address_observed":flag,"new_address_count":len(addresses),"new_address_tokens":"|".join(sorted(addresses)),"iterative_gain":recalls["iterative_hybrid"]-recalls["hybrid"],"hop0_retrieved_chunks":len(hop1),"total_retrieved_chunks":int(iterative["requested_chunks"]),"total_search_comparisons":int(iterative["semantic_comparisons"])+int(iterative["token_comparisons"])})

    requested = {
        int(float(row["requested_chunks"]))
        for row in rows
        if len(row["positive_chunk_ids"].split("|")) > 0
    }
    if requested != {args.budget}:
        raise AssertionError(f"Primary channel budgets differ: {sorted(requested)}")
    validation_best={}
    for dataset in DATASETS:
        validation_best[dataset]=max(CHANNELS,key=lambda c:sum(float(r["evidence_recall"]) for r in rows if r["split"]=="validation" and r["dataset"]==dataset and r["channel"]==c)/max(sum(r["split"]=="validation" and r["dataset"]==dataset and r["channel"]==c for r in rows),1))
    for row in advantages:
        fixed = validation_best[row["dataset"]]
        outcome = by_case[(row["split"], row["dataset"], row["example_id"])][fixed]
        row["oracle_headroom"] = row["oracle_recall"] - float(outcome["evidence_recall"])
    oracle_rows=[]; selector_examples=[]
    for key,outcomes in by_case.items():
        if key[0]!="test": continue
        recalls={c:float(outcomes[c]["evidence_recall"]) for c in CHANNELS}; oracle_name,oracle_recall=oracle_channel(recalls)
        selected=select_observable_channel({k:geometry_by_case[key][k] for k in ("query_rare_fraction","exact_top_score","bm25_score_gap","semantic_score_gap","channel_disagreement","new_address_observed")})
        fixed=validation_best[key[1]]
        oracle_rows.append({"dataset":key[1],"example_id":key[2],"best_fixed_channel":fixed,"best_fixed_recall":recalls[fixed],"oracle_channel":oracle_name,"oracle_recall":oracle_recall,"headroom":oracle_recall-recalls[fixed],"static_hybrid_recall":recalls["hybrid"],"adaptive_channel":selected,"adaptive_recall":recalls[selected]})
        selector_examples.append((key,outcomes,fixed,selected,oracle_name))
    selector_rows=[]
    for dataset in DATASETS:
        group=[r for r in selector_examples if r[0][1]==dataset]
        for policy,chooser in (("best_fixed",lambda r:r[2]),("hybrid",lambda r:"hybrid"),("adaptive",lambda r:r[3]),("oracle",lambda r:r[4])):
            chosen=[r[1][chooser(r)] for r in group]
            selector_rows.append({"dataset":dataset,"policy":policy,"examples":len(chosen),"recall":sum(float(r["evidence_recall"]) for r in chosen)/max(len(chosen),1),"precision":sum(float(r["precision"]) for r in chosen)/max(len(chosen),1)})

    summary=_summaries(rows,args.seed)
    for dataset in DATASETS: _write_csv(args.output_dir/f"channel_results_{'2wiki' if dataset=='2wikimultihopqa' else dataset}.csv",[r for r in rows if r["dataset"]==dataset])
    _write_csv(args.output_dir/"channel_geometry_features.csv",geometry_rows)
    _write_csv(args.output_dir/"channel_advantage_rows.csv",advantages)
    _write_csv(args.output_dir/"channel_precision_recall.csv",summary)
    _write_csv(args.output_dir/"channel_oracle_headroom.csv",oracle_rows)
    _write_csv(args.output_dir/"channel_overlap.csv",overlap_rows)
    _write_csv(args.output_dir/"iterative_new_address.csv",new_rows)
    _write_csv(args.output_dir/"channel_selector_baselines.csv",selector_rows)
    synthetic,_wrong=_synthetic_study(tokenizer,WEIGHTS,budget=2)
    channel_map={"B0_gist":"gist","B1_bm25":"bm25","B2_exact":"exact","B3_weighted":"weighted","B4_approx":"approx","H5_iterative_hybrid":"iterative_hybrid"}
    wrong_rows=[{**row,"channel":channel_map[row["condition"]]} for row in synthetic]
    _write_csv(args.output_dir/"wrong_reference_robustness.csv",wrong_rows)
    findings={"protocol":{"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"routing_layer":27,"chunk_tokens":32,"requested_chunks":args.budget,"iterative_depth":2,"iterative_per_hop_branch":args.budget//2,"materialization_performed":False,"hybrid_weights":{"entry_semantic":WEIGHTS[0],"later_semantic":WEIGHTS[1]},"selector_gold_features":False,"seed":args.seed},"cohorts":{d:{s:sum(f[0]["dataset"]==d and f[0]["split"]==s for f in cases) for s in ("validation","test")} for d in DATASETS},"summary":summary,"validation_selected_fixed_channel":validation_best,"oracle_headroom":{d:sum(r["headroom"] for r in oracle_rows if r["dataset"]==d)/max(sum(r["dataset"]==d for r in oracle_rows),1) for d in DATASETS},"selector":selector_rows,"wrong_reference":_wrong}
    (args.output_dir/"paper2_6_findings.json").write_text(json.dumps(findings,indent=2,sort_keys=True),encoding="utf-8")
    audit=["# Paper 2.6 claim audit","","- Frozen Qwen3-0.6B features; no model weights are loaded by this analysis.","- Every primary channel requests at most four aligned 32-token chunks.","- Iterative routing reports total unique chunks and all semantic/token comparisons.","- Natural dataset identities use deterministic validation/test partitions inherited from Paper 2.5.","- The heuristic selector receives only query/routing observables; its API rejects gold geometry fields.","- Gold evidence geometry is explanatory only.","- No native K/V is materialized and no generation metric is reported.","- Cohorts are below the requested 50 held-out examples per dataset; conclusions are cohort-bounded.","- Approximate matching is normalized tokenizer-piece sequence similarity, not character edit distance."]
    (args.output_dir/"claim_audit.md").write_text("\n".join(audit)+"\n",encoding="utf-8")
    _plots(summary,geometry_rows,advantages,oracle_rows,overlap_rows,new_rows,selector_rows,wrong_rows,args.output_dir)
    return findings


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--budget",type=int,default=4); parser.add_argument("--seed",type=int,default=20260811); parser.add_argument("--local-files-only",action="store_true")
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"data/.hf_cache")
    parser.add_argument("--paper2-feature-dir",type=Path,default=ROOT/"docs/papers/shared/results/paper2_hf/routing/learned_adapter")
    natural=ROOT/"docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt"
    data=ROOT/"data/.paper2_5_datasets"
    parser.add_argument("--natural-features",type=Path,default=natural)
    parser.add_argument("--musique-dev",type=Path,default=data/"musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev",type=Path,default=data/"2wiki/dev.json")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"docs/papers/shared/results/paper2_6_hybrid_pra/channel_geometry")
    return parser.parse_args()


if __name__=="__main__":
    print(json.dumps(run(parse_args())["protocol"],indent=2))
