import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def get_reliability_score(rating, length, age):

    # -------------------------------------------------
    # INPUTS
    # -------------------------------------------------
    rating_input = ctrl.Antecedent(
        np.arange(1, 6.1, 1),
        'rating'
    )

    length_input = ctrl.Antecedent(
        np.arange(0, 151, 1),
        'length'
    )

    age_input = ctrl.Antecedent(
        np.arange(0, 2001, 1),
        'age'
    )

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    reliability_output = ctrl.Consequent(
        np.arange(0, 101, 1),
        'reliability'
    )

    # -------------------------------------------------
    # MEMBERSHIP FUNCTIONS
    # -------------------------------------------------

    # Rating
    rating_input['low'] = fuzz.trimf(
        rating_input.universe,
        [1, 1, 3]
    )

    rating_input['medium'] = fuzz.trimf(
        rating_input.universe,
        [2, 3, 4]
    )

    rating_input['high'] = fuzz.trimf(
        rating_input.universe,
        [3, 5, 5]
    )

    # Review Length
    length_input['short'] = fuzz.trimf(
        length_input.universe,
        [0, 0, 40]
    )

    length_input['medium'] = fuzz.trimf(
        length_input.universe,
        [20, 75, 130]
    )

    length_input['long'] = fuzz.trimf(
        length_input.universe,
        [80, 150, 150]
    )

    # Review Age
    age_input['new'] = fuzz.trimf(
        age_input.universe,
        [0, 0, 180]
    )

    age_input['moderate'] = fuzz.trimf(
        age_input.universe,
        [90, 500, 1000]
    )

    age_input['old'] = fuzz.trimf(
        age_input.universe,
        [700, 2000, 2000]
    )

    # Reliability
    reliability_output['low'] = fuzz.trimf(
        reliability_output.universe,
        [0, 0, 50]
    )

    reliability_output['medium'] = fuzz.trimf(
        reliability_output.universe,
        [25, 50, 75]
    )

    reliability_output['high'] = fuzz.trimf(
        reliability_output.universe,
        [50, 100, 100]
    )

    # -------------------------------------------------
    # RULES
    # -------------------------------------------------

    rule1 = ctrl.Rule(
        rating_input['high']
        & length_input['long']
        & age_input['new'],
        reliability_output['high']
    )

    rule2 = ctrl.Rule(
        rating_input['low']
        & length_input['short'],
        reliability_output['low']
    )

    rule3 = ctrl.Rule(
        rating_input['medium'],
        reliability_output['medium']
    )

    rule4 = ctrl.Rule(
        age_input['new']
        & length_input['long'],
        reliability_output['high']
    )

    rule5 = ctrl.Rule(
        age_input['old']
        & length_input['short'],
        reliability_output['low']
    )

    rule6 = ctrl.Rule(
        rating_input['high']
        & age_input['old'],
        reliability_output['medium']
    )

    rule7 = ctrl.Rule(
        rating_input['low']
        & age_input['new'],
        reliability_output['medium']
    )

    # -------------------------------------------------
    # CONTROL SYSTEM
    # -------------------------------------------------

    system = ctrl.ControlSystem([
        rule1,
        rule2,
        rule3,
        rule4,
        rule5,
        rule6,
        rule7
    ])

    sim = ctrl.ControlSystemSimulation(system)

    sim.input['rating'] = float(rating)
    sim.input['length'] = float(length)
    sim.input['age'] = float(age)

    try:
        sim.compute()

        reliability = sim.output.get(
            'reliability',
            50
        )

        return round(reliability, 2)

    except Exception as e:
        print("Fuzzy Error:", e)
        return 50.0