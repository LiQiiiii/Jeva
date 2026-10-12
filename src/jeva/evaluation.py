from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
import math
from pathlib import Path
import random
import statistics
import time


FORMAT_VERSION = 1
VISIBLE_FIELDS = ("state", "question", "decision_type", "options", "allow_null", "values", "positive_index")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def implementation_identity():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): file_sha256(path)
            for path in sorted(root.rglob("*.py"))}


def read_jsonl(path):
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank records are not permitted")
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty evaluation input: {path}")
    return rows


def validate_record(row):
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("Every record needs a nonempty string id")
    if row.get("accuracy_rule", "first_argmax") not in ("first_argmax", "tied_top_set"):
        raise ValueError("accuracy_rule must be first_argmax or tied_top_set")
    for key in ("state", "question"):
        if not isinstance(row.get(key), str) or (key == "question" and not row[key].strip()):
            raise ValueError(f"{row['id']}: {key} must be text; supply inline text or resolve it with --corpus")
    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2 or any(not isinstance(v, str) or not v.strip() for v in options):
        raise ValueError(f"{row['id']}: at least two nonempty option strings required")
    kind = row.get("decision_type")
    if kind not in ("choice", "noul", "score"):
        raise ValueError(f"{row['id']}: unsupported decision_type")
    if type(row.get("allow_null")) is not bool:
        raise ValueError(f"{row['id']}: allow_null must be boolean")
    target = row.get("target_probs")
    if not isinstance(target, list) or len(target) != len(options) + 1:
        raise ValueError(f"{row['id']}: target_probs must be [null, candidate_0, ...]")
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in target):
        raise ValueError(f"{row['id']}: invalid target probabilities")
    if not math.isclose(sum(target), 1, abs_tol=1e-5):
        raise ValueError(f"{row['id']}: target probabilities must sum to one")
    if not row["allow_null"] and target[0] != 0:
        raise ValueError(f"{row['id']}: closed-set target assigns null mass")
    if kind == "noul" and (len(options) != 2 or type(row.get("positive_index")) is not int or row["positive_index"] not in (0, 1)):
        raise ValueError(f"{row['id']}: Noul needs two options and positive_index 0 or 1")
    if kind == "score":
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(options) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"{row['id']}: Score needs one finite value per candidate")
    if "qrels" in row:
        ids = row.get("candidate_ids")
        if not isinstance(ids, list) or len(ids) != len(options) or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{row['id']}: retrieval requires unique string candidate_ids matching options")
        qrels = row["qrels"]
        if not isinstance(qrels, dict) or any(not isinstance(key, str) or type(grade) not in (int, float) or not math.isfinite(grade) or grade < 0 for key, grade in qrels.items()):
            raise ValueError(f"{row['id']}: invalid qrels")
        threshold = row.get("eligible_grade", 1)
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"{row['id']}: invalid eligible_grade")
    return row


def resolve_corpus(rows, corpus):

    bank = {}
    for entry in read_jsonl(corpus):
        key = entry["id"]
        if key in bank:
            raise ValueError(f"Duplicate corpus id: {key}")
        bank[key] = entry["text"]
    resolved = []
    for row in rows:
        item = dict(row)
        for field in ("state", "question"):
            if field not in item and field + "_id" in item:
                item[field] = bank[item[field + "_id"]]
        if "options" not in item:
            item["options"] = [bank[key] for key in item["option_ids"]]
        resolved.append(item)
    return resolved


def permutations(k, random_count=4, seed=2026):

    if k < 2 or random_count < 0:
        raise ValueError("Invalid permutation arguments")
    identity = tuple(range(k))
    orders = [identity, tuple(reversed(identity))]
    seen = set(orders)
    seed_bytes = f"das1-option-permutation-v1\0{seed}\0{k}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big"))
    target = min(random_count + 2, math.factorial(k) if k < 10 else random_count + 2)
    attempts = 0
    while len(orders) < target:
        order = list(identity)
        rng.shuffle(order)
        order = tuple(order)
        if order not in seen:
            orders.append(order)
            seen.add(order)
        attempts += 1
        if attempts > max(100, target * 100):
            raise RuntimeError("Unable to generate distinct permutation orders")
    return orders


def permute_record(row, order):
    if sorted(order) != list(range(len(row["options"]))):
        raise ValueError("Not a candidate permutation")
    result = dict(row, options=[row["options"][i] for i in order],
                  target_probs=[row["target_probs"][0], *[row["target_probs"][i + 1] for i in order]])
    if row.get("values") is not None:
        result["values"] = [row["values"][i] for i in order]
    if row.get("positive_index") is not None:
        result["positive_index"] = order.index(row["positive_index"])
    for field in ("candidate_ids", "option_ids", "upstream_option_ids"):
        if field in row:
            result[field] = [row[field][i] for i in order]
    return result


def correct_index(target, selected, rule="first_argmax"):
    if rule == "first_argmax":
        return float(selected == max(range(len(target)), key=target.__getitem__))
    return float(target[selected] >= max(target) - 1e-8)


def score_distribution(row, conditional, null_probability):
    q = list(conditional)
    if len(q) != len(row["options"]) or any(not math.isfinite(p) or p < 0 for p in q) or not math.isclose(sum(q), 1, abs_tol=1e-5):
        raise ValueError(f"{row['id']}: invalid model candidate distribution")
    if not math.isfinite(null_probability) or not 0 <= null_probability <= 1:
        raise ValueError(f"{row['id']}: invalid null probability")
    null = null_probability if row["allow_null"] else 0.0
    p = [null, *[(1 - null) * v for v in q]]
    target = row["target_probs"]
    if not row["allow_null"]:
        mass = sum(target[1:])
        target = [0.0, *[v / mass for v in target[1:]]]
    best = max(range(len(p)), key=p.__getitem__)
    rule = row.get("accuracy_rule", "first_argmax")
    result = dict(probabilities=p, candidate_probabilities=q, predicted_index=best,
                  accuracy=correct_index(target, best, rule), accuracy_rule=rule,
                  nll=-sum(t * math.log(max(v, 1e-8)) for t, v in zip(target, p)),
                  brier=sum((v - t) ** 2 for t, v in zip(target, p)), confidence=max(p))
    result["raw_ce"] = -sum(t * math.log(max(v, 1e-12)) for t, v in zip(row["target_probs"], p))
    result["brier_vs_target_distribution"] = sum((v-t)**2 for t,v in zip(row["target_probs"],p))
    if row["allow_null"]:
        result.update(null_probability=null, null_target=target[0],
                      null_mae=abs(null-target[0]), null_brier=(null-target[0])**2)
    mass = sum(target[1:])
    if mass > 0:
        ct = [v / mass for v in target[1:]]
        result.update(candidate_accuracy=correct_index(ct, max(range(len(q)), key=q.__getitem__), rule),
                      candidate_nll=-sum(t * math.log(max(v, 1e-8)) for t, v in zip(ct, q)),
                      candidate_brier=sum((v - t) ** 2 for t, v in zip(ct, q)))
        if row["decision_type"] == "score":
            error = sum(v * (a - b) for v, a, b in zip(row["values"], q, ct))
            result.update(score_mae=abs(error), score_squared_error=error * error)
        if row["decision_type"] == "noul":
            error = q[row["positive_index"]] - ct[row["positive_index"]]
            result.update(noul_mae=abs(error), noul_brier=error * error,
                          noul_probability=q[row["positive_index"]], noul_target=ct[row["positive_index"]])
    if "qrels" in row:
        ids = row["candidate_ids"]
        qrels = row["qrels"]
        ranking = sorted(range(len(q)), key=lambda i: (-q[i], i))
        threshold = row.get("eligible_grade", 1)
        relevant = {key for key, grade in qrels.items() if grade >= threshold}
        hits = [qrels.get(ids[i], 0) >= threshold for i in ranking]
        result["hit_at_1"] = float(hits[0])
        result["mrr"] = next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0)
        result["recall_at_10"] = sum(hits[:10]) / len(relevant) if relevant else 0.0

        gains = [qrels.get(ids[i], 0) for i in ranking[:10]]
        ideal = sorted(qrels.values(), reverse=True)[:10]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        result["ndcg_at_10"] = dcg / idcg if idcg else 0.0
    return result


def calibration_error(probabilities, targets, bins=15):
    groups = defaultdict(list)
    for p, target in zip(probabilities, targets):
        groups[min(int(p*bins), bins-1)].append((p,target))
    return sum(len(group)/len(probabilities)*abs(statistics.fmean(p for p,_ in group)-statistics.fmean(t for _,t in group)) for group in groups.values())


def binary_ranking_metrics(probabilities, labels):
    positive = sum(labels)
    negative = len(labels)-positive
    if not positive or not negative:
        return dict(auroc=None, auprc=None, reason="both_binary_classes_required")
    groups = defaultdict(list)
    for p,label in zip(probabilities,labels):
        groups[p].append(label)
    smaller_negative = wins = 0
    for p in sorted(groups):
        pos = sum(groups[p])
        neg = len(groups[p])-pos
        wins += pos*(smaller_negative+neg/2)
        smaller_negative += neg
    tp = fp = 0
    ap = 0.0
    for p in sorted(groups,reverse=True):
        pos = sum(groups[p])
        tp += pos
        fp += len(groups[p])-pos
        ap += pos/positive*tp/(tp+fp)
    return dict(auroc=wins/(positive*negative),auprc=ap)


def summarize(predictions):
    metrics = ("accuracy", "nll", "brier", "raw_ce", "brier_vs_target_distribution", "candidate_accuracy", "candidate_nll", "candidate_brier", "score_mae", "noul_mae", "noul_brier", "null_mae", "null_brier", "request_ms", "hit_at_1", "mrr", "recall_at_10", "ndcg_at_10")
    report = {"n": len(predictions)}
    for key in metrics:
        values = [row[key] for row in predictions if key in row]
        if values:
            report[key] = statistics.fmean(values)
            report[key + "_n"] = len(values)
    squared = [row["score_squared_error"] for row in predictions if "score_squared_error" in row]
    if squared:
        report["score_rmse"] = math.sqrt(statistics.fmean(squared))
    for count in (10, 15):
        if predictions:
            report[f"ece_{count}_bins"] = calibration_error([r["confidence"] for r in predictions],[r["accuracy"] for r in predictions],count)
    for kind in ("null", "noul"):
        subset = [r for r in predictions if kind+"_probability" in r]
        if subset:
            probabilities = [r[kind+"_probability"] for r in subset]
            targets = [r[kind+"_target"] for r in subset]
            report[kind+"_ece"] = calibration_error(probabilities,targets)
            if kind == "null":
                report.update({"null_"+k:v for k,v in binary_ranking_metrics(probabilities,[t>=.5 for t in targets]).items()})
                for name, condition in (("withheld",True),("present",False)):
                    group = [r for r in subset if (r["null_target"]>=.5)==condition]
                    report[name+"_n"] = len(group)
                    if group:
                        report[name+"_accuracy"] = statistics.fmean(r["accuracy"] for r in group)
    groups = defaultdict(list)
    for row in predictions:
        if row.get("group_id") is not None:
            groups[row["group_id"]].append(row["accuracy"])
    if groups:
        report["macro_group_accuracy"] = statistics.fmean(statistics.fmean(v) for v in groups.values())
    candidate_groups = defaultdict(list)
    for row in predictions:
        if row.get("group_id") is not None and "candidate_accuracy" in row:
            candidate_groups[row["group_id"]].append(row["candidate_accuracy"])
    if candidate_groups:
        report["candidate_macro_group_accuracy"] = statistics.fmean(statistics.fmean(v) for v in candidate_groups.values())
    return report


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def hydrate_cache_episodes(episodes, source_records):

    source_records = list(source_records)
    sources = {row["id"]: row for row in source_records}
    texts = {}
    for row in source_records:
        ids = row.get("upstream_option_ids", row.get("option_ids"))
        if ids is None:
            ids = [fingerprint(["text", text]) for text in row["options"]]
        for key, text in zip(ids, row["options"], strict=True):


            for alias in (key, fingerprint(["text", text])):
                if alias in texts and texts[alias] != text:
                    raise ValueError("Option id aliases different source text")
                texts[alias] = text
    result = []
    for episode in episodes:
        def restore(reference):
            if reference is None:
                return None
            source = sources[reference["source_id"]]
            item = dict(reference, state=source["state"], question=source["question"],
                        options=[texts[key] for key in reference["option_ids"]])
            for field in ("state", "question"):
                if fingerprint(item[field]) != reference[field + "_sha256"]:
                    raise ValueError("Cache source text identity mismatch")
            if [fingerprint(text) for text in item["options"]] != reference["option_text_sha256"]:
                raise ValueError("Cache option text identity mismatch")
            return item
        result.append(dict(episode, prime=restore(episode["prime"]), requests=[restore(r) for r in episode["requests"]]))
    return result


def _cache_counters(model):

    cache_info = getattr(model, "cache_info", None)
    if not callable(cache_info):
        return None
    info = cache_info()
    if info is None:
        return None
    counters = {name: {key: info[name][key] for key in ("hits", "misses")}
                for name in ("sets", "options")}
    if any(type(value) is not int or value < 0
           for group in counters.values() for value in group.values()):
        raise ValueError("Cache counters must be nonnegative integers")
    return counters


def _cache_counter_delta(before, after):
    if before is None or after is None:
        return None
    delta = {name: {key: after[name][key] - before[name][key]
                    for key in ("hits", "misses")}
             for name in ("sets", "options")}
    if any(value < 0 for group in delta.values() for value in group.values()):
        raise ValueError("Cache counters reset during a request")
    return delta


def evaluate_cache_episodes(model, episodes, output, *, model_identity):

    if not model_identity:
        raise ValueError("Explicit immutable model identity required")
    if not episodes or len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Empty or duplicated cache episodes")
    for episode in episodes:
        if not episode.get("requests"):
            raise ValueError("Empty cache episode")
        for request in episode["requests"]:
            validate_record(request)
        if episode.get("prime") and "target_probs" in episode["prime"]:
            raise ValueError("Cache priming must not contain target labels")
    protocol = dict(format_version=FORMAT_VERSION, model=model_identity, episodes_sha256=fingerprint(episodes),
                    implementation_sha256=implementation_identity(), modes=["none", "set", "hybrid"],
                    resume="whole_episode", timing="API_wall_time_no_historical_warmup")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Cache resume identity differs")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output without cache protocol")
    else:
        atomic_json(protocol_path, protocol)
    completed = []
    previous_mode = model.cache
    try:
        for index, episode in enumerate(episodes):
            path = output / f"episode-{index:06d}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                checksum = saved.pop("result_sha256", None)
                if checksum != fingerprint(saved):
                    raise ValueError("Cache episode result checksum mismatch")
                saved["result_sha256"] = checksum
                if saved.get("id") != episode["id"] or saved.get("input_sha256") != fingerprint(episode):
                    raise ValueError("Cache episode identity mismatch")
                completed.append(saved)
                continue
            mode_results = {}
            prime_ms_by_mode = {}
            prime_cache_stats_by_mode = {}
            modes = protocol["modes"][index % 3:] + protocol["modes"][:index % 3]
            for mode in modes:
                model.cache = mode
                model.clear_cache()
                prime_ms_by_mode[mode] = 0.0
                prime_cache_stats_by_mode[mode] = None
                if episode.get("prime"):
                    before = _cache_counters(model)
                    started = time.perf_counter()
                    model.predict(**{k:v for k,v in episode["prime"].items() if k in VISIBLE_FIELDS and v is not None})
                    prime_ms_by_mode[mode] = (time.perf_counter()-started)*1000
                    prime_cache_stats_by_mode[mode] = _cache_counter_delta(before, _cache_counters(model))
                measured = []
                for request in episode["requests"]:
                    before = _cache_counters(model)
                    started = time.perf_counter()
                    answer = model.predict(**{k:v for k,v in request.items() if k in VISIBLE_FIELDS and v is not None})
                    score = score_distribution(request, answer.candidate_probabilities, answer.null_probability)
                    score.update(id=request["id"], request_ms=(time.perf_counter()-started)*1000)
                    score["cache_stats"] = _cache_counter_delta(before, _cache_counters(model))
                    if getattr(answer, "usage", None) is not None:
                        score["usage"] = answer.usage.to_dict()
                    measured.append(score)
                mode_results[mode] = measured
            drift = max(abs(a-b) for mode in ("set", "hybrid") for first, second in zip(mode_results["none"], mode_results[mode]) for a,b in zip(first["probabilities"],second["probabilities"]))
            saved = dict(id=episode["id"], input_sha256=fingerprint(episode), modes=mode_results, max_probability_drift=drift,
                         prime_ms_by_mode=prime_ms_by_mode, prime_cache_stats_by_mode=prime_cache_stats_by_mode)
            saved["result_sha256"] = fingerprint(saved)
            atomic_json(path, saved)
            completed.append(saved)
    finally:
        model.cache = previous_mode
        model.clear_cache()
    report = dict(episodes=len(completed), max_probability_drift=max(e["max_probability_drift"] for e in completed),
                  by_mode={mode:summarize([r for e in completed for r in e["modes"][mode]]) for mode in protocol["modes"]})
    atomic_json(output / "summary.json", report)
    return report


def evaluate_records(model, rows, output, *, model_identity, random_permutations=None, seed=2026):

    rows = [validate_record(row) for row in rows]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty input or duplicate evaluation ids")
    if not model_identity:
        raise ValueError("Explicit immutable model identity required")
    protocol = dict(format_version=FORMAT_VERSION, data_sha256=fingerprint(rows), model=model_identity,
                    implementation_sha256=implementation_identity(), random_permutations=random_permutations, seed=seed,
                    cache_policy="reset_before_each_request", score_permutations="excluded_ordinal_semantics")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    saved_path = output / "predictions.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol:
            raise ValueError("Resume identity differs; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Refusing to reuse an unmanifested nonempty output directory")
    else:
        atomic_json(manifest, protocol)
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else []
    expected = []
    for row in rows:
        orders = [tuple(range(len(row["options"])))]
        if random_permutations is not None and row["decision_type"] != "score":
            orders = permutations(len(row["options"]), random_permutations, seed)
        expected.extend((row, order, index) for index, order in enumerate(orders))
    if len(saved) > len(expected):
        raise ValueError("Resume predictions exceed expected count")
    for i, prediction in enumerate(saved):
        checksum = prediction.pop("result_sha256", None)
        if checksum != fingerprint(prediction):
            raise ValueError("Resume prediction checksum mismatch")
        prediction["result_sha256"] = checksum
        row, order, variant = expected[i]
        if prediction.get("id") != row["id"] or prediction.get("variant") != variant or prediction.get("order") != list(order):
            raise ValueError("Resume predictions are not the exact expected prefix")
        if prediction.get("row_sha256") != fingerprint(row):
            raise ValueError("Resume row fingerprint mismatch")
    for row, order, variant in expected[len(saved):]:
        item = permute_record(row, order)
        model.clear_cache()
        started = time.perf_counter()
        answer = model.predict(**{k: item[k] for k in VISIBLE_FIELDS if k in item and item[k] is not None})
        elapsed = (time.perf_counter() - started) * 1000
        canonical = [0.0] * len(order)
        for display, original in enumerate(order):
            canonical[original] = answer.candidate_probabilities[display]
        result = score_distribution(row, canonical, answer.null_probability)


        null = answer.null_probability if row["allow_null"] else 0.0
        displayed = [null, *[(1 - null) * v for v in answer.candidate_probabilities]]
        selected = max(range(len(displayed)), key=displayed.__getitem__)
        canonical_selected = 0 if selected == 0 else order[selected - 1] + 1
        result["predicted_index"] = canonical_selected
        result["accuracy"] = correct_index(row["target_probs"], canonical_selected, row.get("accuracy_rule", "first_argmax"))
        if "candidate_accuracy" in result:
            candidate_selected = order[max(range(len(order)), key=answer.candidate_probabilities.__getitem__)]
            result["candidate_accuracy"] = correct_index(row["target_probs"][1:], candidate_selected, row.get("accuracy_rule", "first_argmax"))
        result.update(id=row["id"], group_id=row.get("group_id"), decision_type=row["decision_type"],
                      variant=variant, order=list(order), row_sha256=fingerprint(row), request_ms=elapsed)
        result["result_sha256"] = fingerprint(result)
        saved.append(result)
        atomic_json(saved_path, saved)
    originals = [row for row in saved if row["variant"] == 0]
    report = dict(protocol=protocol, expected_rows=len(rows), completed_rows=len(originals),
                  overall=summarize(originals), by_type={kind: summarize([r for r in originals if r["decision_type"] == kind]) for kind in sorted({r["decision_type"] for r in originals})},
                  timing_note="Cold request wall time; includes Python API overhead. Historical GPU latency protocols differ.")
    if random_permutations is not None:
        bases = {r["id"]: r for r in originals}
        changed = [r for r in saved if r["variant"] > 0]
        report["permutation"] = dict(eligible_rows=sum(r["decision_type"] != "score" for r in rows),
            excluded_score_rows=sum(r["decision_type"] == "score" for r in rows), measured_variants=len(changed),
            prediction_agreement=statistics.fmean(float(r["predicted_index"] == bases[r["id"]]["predicted_index"]) for r in changed) if changed else None,
            max_probability_drift=max((max(abs(a-b) for a,b in zip(r["probabilities"], bases[r["id"]]["probabilities"])) for r in changed), default=0.0))
    atomic_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Jeva on local JSONL decisions")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="Immutable Hub commit, required for remote models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--permutations", type=int, help="Additional random orders; also measures original and reverse")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-state-length", type=int, default=4096)
    parser.add_argument("--max-option-length", type=int, default=512)
    args = parser.parse_args()
    rows = read_jsonl(args.data)
    if args.corpus:
        rows = resolve_corpus(rows, args.corpus)
    model_path = Path(args.model)
    if model_path.is_dir():
        files = sorted(p for p in model_path.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".pt", ".json", ".yaml", ".model", ".jinja", ".txt"))
        identity = {"files": {str(p.relative_to(model_path)): file_sha256(p) for p in files}}
    else:
        if not args.revision or len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
            parser.error("Remote evaluation requires --revision as an immutable 40-character commit")
        identity = {"repo_id": args.model, "revision": args.revision}
    from jeva import Jeva
    settings = dict(device=args.device, max_state_length=args.max_state_length, max_option_length=args.max_option_length)
    identity["settings"] = settings
    model = Jeva.from_pretrained(args.model, revision=args.revision, **settings)
    report = evaluate_records(model, rows, args.output, model_identity=identity, random_permutations=args.permutations, seed=args.seed)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
