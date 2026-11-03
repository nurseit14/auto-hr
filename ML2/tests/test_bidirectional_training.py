"""Meaningful checks for candidate-ranking supervision and grouped selection."""
import numpy as np
import pytest

from ML2.features import FEATURE_NAMES
from ML2.metrics import evaluate
from ML2.ranker import LinearRanker, preference_pairs
from ML2.train import candidate_folds


def rows_and_matrix():
    rows = [
        {"cv_id": "a", "candidate_group_id": "a", "vacancy_id": "one", "relevance": 2},
        {"cv_id": "a", "candidate_group_id": "a", "vacancy_id": "two", "relevance": 0},
        {"cv_id": "b", "candidate_group_id": "b", "vacancy_id": "one", "relevance": 0},
        {"cv_id": "b", "candidate_group_id": "b", "vacancy_id": "two", "relevance": 1},
    ]
    matrix = np.zeros((len(rows), len(FEATURE_NAMES)))
    matrix[:, 0] = [1, 0, 0, .5]
    return rows, matrix


def test_both_directions_have_balanced_mass_and_learn_candidate_order():
    rows, matrix = rows_and_matrix()
    differences, weights = preference_pairs(rows, matrix, hr_weight=.5)
    assert differences.shape == (4, len(FEATURE_NAMES))
    assert weights.sum() == pytest.approx(1)
    assert weights[:2].sum() == pytest.approx(.5)
    assert weights[2:].sum() == pytest.approx(.5)
    model = LinearRanker.fit(rows, matrix, hr_weight=.5)
    scores = model.predict(matrix)
    assert scores[0] > scores[2]  # Candidates A and B for job one.
    assert scores[3] > scores[1]  # Candidates B and A for job two.
    assert evaluate(rows, scores, group_by="vacancy_id")["ndcg_at_5_informative"] == 1


def test_hr_preferences_do_not_compare_versions_of_the_same_candidate():
    rows = [{"cv_id": "old", "candidate_group_id": "same", "vacancy_id": "one", "relevance": 0},
            {"cv_id": "new", "candidate_group_id": "same", "vacancy_id": "one", "relevance": 2}]
    with pytest.raises(ValueError, match="different relevance"):
        preference_pairs(rows, np.zeros((2, len(FEATURE_NAMES))), hr_weight=1.)


def test_grade_gap_changes_preference_importance_without_fabricating_pairs():
    rows = [{"cv_id": "one", "relevance": label} for label in [0, 1, 2]]
    matrix = np.zeros((3, len(FEATURE_NAMES)))
    matrix[:, 0] = [0, .5, 1]
    _, weights = preference_pairs(rows, matrix, grade_gap=True)
    assert sorted(weights.tolist()) == [.25, .25, .5]


def test_candidate_versions_stay_in_one_reproducible_fold_without_label_use():
    rows = [{"cv_id": f"{group}-{version}", "candidate_group_id": group, "relevance": version}
            for group in ["a", "b", "c", "d"] for version in [0, 1]]
    original = candidate_folds(rows)
    relabelled = candidate_folds([{**row, "relevance": 100 - row["relevance"]} for row in rows])
    for (train, heldout), (again_train, again_heldout) in zip(original, relabelled):
        assert np.array_equal(train, again_train)
        assert np.array_equal(heldout, again_heldout)
        assert {rows[i]["candidate_group_id"] for i in train}.isdisjoint(
            {rows[i]["candidate_group_id"] for i in heldout})


def test_unknown_labels_cannot_enter_either_training_direction():
    rows, matrix = rows_and_matrix()
    rows[0]["relevance"] = None
    with pytest.raises(ValueError, match="Unknown"):
        preference_pairs(rows, matrix, hr_weight=.5)


def test_hr_metric_counts_one_candidate_despite_multiple_resume_versions():
    rows = [{"cv_id": "a-old", "candidate_group_id": "a", "vacancy_id": "job", "relevance": 2},
            {"cv_id": "a-new", "candidate_group_id": "a", "vacancy_id": "job", "relevance": 2},
            {"cv_id": "b", "candidate_group_id": "b", "vacancy_id": "job", "relevance": 0}]
    result = evaluate(rows, [3, 2, 1], group_by="vacancy_id")
    assert result["input_pairs"] == 3
    assert result["evaluated_pairs"] == 2
    assert result["duplicate_candidate_versions_excluded"] == 1
    assert result["multi_candidate_queries"] == result["informative_queries"] == 1
    assert result["ndcg_at_5_informative"] == 1


def test_correctness_refit_rechecks_only_previously_locked_configuration(tmp_path, monkeypatch):
    import json
    import ML2.train as training
    from ML2.prepare_dataset import build_dataset
    from ML2.tests.test_prepare_dataset import export_fixture

    tasks = export_fixture()
    for task in tasks:
        task["data"]["vacancy_fields"] = {"title": "Java Developer", "experience": "1–3 года", "requirements": "Java, Python"}
    source, dataset, output = tmp_path / "source.json", tmp_path / "dataset", tmp_path / "model"
    source.write_text(json.dumps(tasks), encoding="utf-8")
    build_dataset(source, dataset)
    original, calls = training.select_configuration, []

    def bounded(rows, matrix, l2_grid, hr_grid, grade_grid):
        calls.append((l2_grid, hr_grid, grade_grid))
        assert l2_grid == (1.,) and hr_grid == (.5,) and grade_grid == (False,)
        return original(rows, matrix, l2_grid, hr_grid, grade_grid)

    monkeypatch.setattr(training, "select_configuration", bounded)
    provenance = {"reason": "heldout-informed extraction correctness fix", "original_model_sha256": "frozen"}
    report = training.train_dataset(dataset, output,
                                    locked_configuration={"l2": 1., "hr_weight": .5, "grade_gap": False},
                                    development_provenance=provenance)
    assert len(calls) == 1
    assert report["cross_validation_search"] == []
    assert report["development_provenance"] == provenance
    assert report["fixed_configuration_cross_validation_recheck"]["l2"] == 1.
    model = json.loads((output / "model.json").read_text())
    assert model["metadata"]["model_selection"]["configuration_count"] == 0
    assert model["metadata"]["model_selection"]["fixed_configuration_recheck"] is True
