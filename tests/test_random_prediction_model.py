from src.model.random import RandomPredictionModel


def test_is_deterministic():
    seed = 42
    random_model_1 = RandomPredictionModel(0, 1, -1, 1, seed=seed)
    predictions_1 = [random_model_1._do_predict(None) for _ in range(100)]

    random_model_2 = RandomPredictionModel(0, 1, -1, 1, seed=seed)
    predictions_2 = [random_model_2._do_predict(None) for _ in range(100)]

    assert predictions_1 == predictions_2, "Random model should produce the same predictions with the same seed."