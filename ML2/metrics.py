"""Query-level ranking metrics, averaging tied scores instead of lucky ordering."""

from collections import defaultdict
import math


def expected_dcg(labels, scores, k=5):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    result, start = 0., 0
    while start < len(order) and start < k:
        end = start + 1
        while end < len(order) and math.isclose(scores[order[start]], scores[order[end]], abs_tol=1e-12, rel_tol=0.):
            end += 1
        gain = sum(labels[i] for i in order[start:end]) / (end - start)
        result += gain * sum(1 / math.log2(rank + 2) for rank in range(start, min(end, k)))
        start = end
    return result


def evaluate(rows, scores, k=5, group_by="cv_id"):
    if group_by not in {"cv_id", "vacancy_id"}:
        raise ValueError("Ranking queries must be CVs or vacancies")
    if len(rows) != len(scores) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("One finite score is required per row")
    input_count = len(rows)
    if group_by == "vacancy_id":
        # One person must not occupy multiple slots in a HR ranking metric.
        # Pick a stable source ID without consulting labels or scores.
        canonical = {}
        for row, score in sorted(zip(rows, scores), key=lambda item: str(item[0]["cv_id"])):
            key = (row["vacancy_id"], row.get("candidate_group_id", row["cv_id"]))
            canonical.setdefault(key, (row, score))
        rows, scores = zip(*canonical.values()) if canonical else ([], [])
    queries = defaultdict(list)
    for row, score in zip(rows, scores):
        if row.get("relevance") is None:
            raise ValueError("Unknown relevance is excluded from ranking evaluation")
        queries[row[group_by]].append((int(row["relevance"]), float(score)))
    ndcgs, informative_ndcgs, accuracies, top1s = [], [], [], []
    details = []
    for cid, values in sorted(queries.items()):
        labels, prediction = zip(*values)
        ideal = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(labels, reverse=True)[:k]))
        ndcg = expected_dcg(labels, prediction, k) / ideal if ideal > 0 else None
        informative = len(set(labels)) > 1
        if ndcg is not None:
            ndcgs.append(ndcg)
            if informative:
                informative_ndcgs.append(ndcg)
        pair_credit = [1. if prediction[a] > prediction[b] + 1e-12
                       else .5 if math.isclose(prediction[a], prediction[b], abs_tol=1e-12, rel_tol=0.) else 0.
                       for a in range(len(values)) for b in range(len(values)) if labels[a] > labels[b]]
        accuracy = sum(pair_credit) / len(pair_credit) if pair_credit else None
        if accuracy is not None:
            accuracies.append(accuracy)
        top_indices = [i for i, value in enumerate(prediction)
                       if math.isclose(value, max(prediction), abs_tol=1e-12, rel_tol=0.)]
        top1 = sum(labels[i] == max(labels) for i in top_indices) / len(top_indices)
        if informative:
            top1s.append(top1)
        details.append({group_by: cid, "ndcg": ndcg, "informative": informative,
                        "pairwise_accuracy": accuracy, "top1_best_label_credit": top1})
    mean = lambda values: sum(values) / len(values) if values else None
    result = {f"ndcg_at_{k}": mean(ndcgs), f"ndcg_at_{k}_informative": mean(informative_ndcgs),
            "pairwise_accuracy": mean(accuracies), "top1_best_label_credit_informative": mean(top1s),
            "queries": len(queries), "informative_queries": len(informative_ndcgs),
            "all_weak_queries_excluded_from_ndcg": sum(d["ndcg"] is None for d in details),
            "by_query": details}
    if group_by == "vacancy_id":
        result.update({"input_pairs": input_count, "evaluated_pairs": len(rows),
                       "duplicate_candidate_versions_excluded": input_count - len(rows),
                       "multi_candidate_queries": sum(len(values) > 1 for values in queries.values()),
                       "version_policy": "One candidate group per vacancy; stable minimum CV source ID, independent of labels and scores"})
    return result
