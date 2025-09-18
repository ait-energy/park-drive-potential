import math

from intro_matsim.behavior import (
    MODEL_DOMINO,
    MODE_DRIVER,
    MODE_PASSENGER,
    MODEL_PNM,
    BehaviorModel,
)

model = BehaviorModel()


def test_get_choice_probability__real_data():
    prob = model.get_choice_probability(
        MODEL_DOMINO, MODE_DRIVER, True, True, False, 18
    )
    assert round(prob, 4) == 0.5335

    prob = model.get_choice_probability(
        MODEL_DOMINO, MODE_DRIVER, True, True, False, 37
    )
    assert round(prob, 4) == 0.4767

    prob = model.get_choice_probability(
        MODEL_DOMINO, MODE_PASSENGER, True, False, False, 37
    )
    assert round(prob, 4) == 0.3043


def test_get_choice_probability__old():
    prob = model.get_choice_probability(
        MODEL_DOMINO, MODE_DRIVER, True, True, False, 105
    )
    assert round(prob, 5) == 0.46089


def test_get_choice_probability__young():
    prob = model.get_choice_probability(MODEL_DOMINO, MODE_DRIVER, True, True, False, 5)
    assert math.isnan(prob)


def test_get_choice_probability__valid_values_for_all_scenarios():
    for model_name in [MODEL_DOMINO, MODEL_PNM]:
        for mode in [MODE_DRIVER, MODE_PASSENGER]:
            for employed in [True, False]:
                for full_time in [True, False]:
                    for in_education in [True, False]:
                        for age in range(18, 80):
                            prob = model.get_choice_probability(
                                model_name, mode, employed, full_time, in_education, age
                            )
                            desc = f"Model: {model_name}, Mode: {mode}, Employed: {employed}, Full-time: {full_time}, In education: {in_education}, Age: {age}"
                            assert isinstance(prob, float), (
                                f"{type(prob)} instead of float for {desc}"
                            )
                            assert prob >= 0, f"Negative value for {desc}"
                            assert prob <= 1, f"Value larger than 1 for {desc}"
