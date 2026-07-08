import numpy as np

from clustomer.monitoring.drift import population_stability_index


def test_psi_is_zero_for_identical_population_and_increases_for_shift():
    reference = np.arange(1, 101)
    assert population_stability_index(reference, reference) == 0
    assert population_stability_index(reference, reference + 100) > 0.2
