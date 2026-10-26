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


def evaluate(rows, scores, k=5):
    if len(rows) != len(scores) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("One finite score is required per row")
    queries = defaultdict(list)
    for row, score in zip(rows, scores):
        if row.get("relevance") is None:
            raise ValueError("Unknown relevance is excluded from ranking evaluation")
        queries[row["cv_id"]].append((int(row["relevance"]), float(score)))
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
        details.append({"cv_id": cid, "ndcg": ndcg, "informative": informative,
                        "pairwise_accuracy": accuracy, "top1_best_label_credit": top1})
    mean = lambda values: sum(values) / len(values) if values else None
    return {f"ndcg_at_{k}": mean(ndcgs), f"ndcg_at_{k}_informative": mean(informative_ndcgs),
            "pairwise_accuracy": mean(accuracies), "top1_best_label_credit_informative": mean(top1s),
            "queries": len(queries), "informative_queries": len(informative_ndcgs),
            "all_weak_queries_excluded_from_ndcg": sum(d["ndcg"] is None for d in details),
            "by_query": details}
