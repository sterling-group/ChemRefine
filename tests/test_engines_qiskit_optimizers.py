"""SPSA calculations own their random streams without changing ambient Qiskit state."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import sleep
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chemrefine.engines.qiskit.components.optimizers import SPSAOptions, build_spsa


@pytest.fixture
def fake_spsa(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Expose the same shared-generator behavior as upstream SPSA without dependencies."""
    ambient = SimpleNamespace(random_seed=101, random=np.random.default_rng(101))

    class SPSA:
        """Small upstream stand-in drawing during minimization, including before failures."""

        def __init__(self, **options: Any) -> None:
            self.options = options

        def minimize(self, fun: Any, x0: Any, jac: Any = None, bounds: Any = None) -> Any:
            self.arguments = (x0, jac, bounds)
            return fun(ambient.random.random(3))

    for name, attributes in (
        ("qiskit_algorithms.optimizers", {"SPSA": SPSA}),
        ("qiskit_algorithms.utils", {"algorithm_globals": ambient}),
    ):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return ambient


@pytest.mark.parametrize("seed", [None, 17])
def test_constructing_spsa_preserves_global_seed_and_generator(fake_spsa: Any, seed: Any) -> None:
    """Creating an optimizer does not seed or consume the caller's random stream."""
    generator = fake_spsa.random
    state = deepcopy(generator.bit_generator.state)
    optimizer: Any = build_spsa(options=SPSAOptions(seed=seed))
    assert fake_spsa.random_seed == 101
    assert fake_spsa.random is generator
    assert fake_spsa.random.bit_generator.state == state
    assert optimizer.options["maxiter"] == 100


def test_spsa_instances_advance_independently_and_restore_ambient_state(fake_spsa: Any) -> None:
    """Interleaving optimizers cannot change their seeded trajectories."""
    generator = fake_spsa.random
    state = deepcopy(generator.bit_generator.state)
    first: Any = build_spsa(options=SPSAOptions(seed=17))
    second: Any = build_spsa(options=SPSAOptions(seed=23))
    first_expected = np.random.default_rng(17)
    second_expected = np.random.default_rng(23)
    jacobian, bounds = object(), [(0, 1)]
    for optimizer, expected in [
        (first, first_expected),
        (second, second_expected),
        (first, first_expected),
    ]:
        actual = optimizer.minimize(lambda draws: draws, [0], jac=jacobian, bounds=bounds)
        np.testing.assert_array_equal(actual, expected.random(3))
        assert optimizer.arguments == ([0], jacobian, bounds)
        assert fake_spsa.random_seed == 101
        assert fake_spsa.random is generator
        assert generator.bit_generator.state == state


def test_spsa_restores_ambient_state_after_objective_failure(fake_spsa: Any) -> None:
    """Exceptions restore the outside stream while preserving work already sampled."""
    state = deepcopy(fake_spsa.random.bit_generator.state)
    optimizer: Any = build_spsa(options=SPSAOptions(seed=17))

    def failing_objective(draws: Any) -> Any:
        raise RuntimeError("failed objective")

    with pytest.raises(RuntimeError, match="failed objective"):
        optimizer.minimize(failing_objective, [0])
    assert fake_spsa.random.bit_generator.state == state
    expected = np.random.default_rng(17)
    expected.random(3)
    np.testing.assert_array_equal(optimizer.minimize(lambda draws: draws, [0]), expected.random(3))


def test_nested_spsa_calls_restore_the_outer_stream(fake_spsa: Any) -> None:
    """Nested optimizer use in the same thread neither deadlocks nor changes its parent."""
    state = deepcopy(fake_spsa.random.bit_generator.state)
    outer: Any = build_spsa(options=SPSAOptions(seed=17))
    inner: Any = build_spsa(options=SPSAOptions(seed=23))

    def objective(draws: Any) -> Any:
        before = deepcopy(fake_spsa.random.bit_generator.state)
        inner.minimize(lambda other: other, [0])
        assert fake_spsa.random.bit_generator.state == before
        return draws

    np.testing.assert_array_equal(
        outer.minimize(objective, [0]), np.random.default_rng(17).random(3)
    )
    assert fake_spsa.random.bit_generator.state == state


def test_spsa_calls_from_threads_serialize_the_shared_generator(fake_spsa: Any) -> None:
    """Engine instances cannot overlap their temporary access to upstream random state."""
    state = deepcopy(fake_spsa.random.bit_generator.state)
    active = 0
    maximum_active = 0

    def objective(draws: Any) -> Any:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        sleep(0.005)
        active -= 1
        return draws

    first: Any = build_spsa(options=SPSAOptions(seed=17))
    second: Any = build_spsa(options=SPSAOptions(seed=23))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.minimize, objective, [0])
        second_result = pool.submit(second.minimize, objective, [0])
        np.testing.assert_array_equal(first_result.result(), np.random.default_rng(17).random(3))
        np.testing.assert_array_equal(second_result.result(), np.random.default_rng(23).random(3))
    assert maximum_active == 1
    assert fake_spsa.random.bit_generator.state == state


def test_real_spsa_seed_is_independent_of_another_optimizer_construction() -> None:
    """The supported Qiskit implementation preserves both isolation and reproducibility."""
    utils = pytest.importorskip("qiskit_algorithms.utils")
    generator = utils.algorithm_globals.random
    state = deepcopy(generator.bit_generator.state)
    seed = utils.algorithm_globals.random_seed
    options = SPSAOptions(seed=1, maxiter=2, learning_rate=0.1, perturbation=0.1)
    initial_point = np.arange(8) / 8

    def objective(point: Any) -> float:
        return float(np.dot(point, point))

    first: Any = build_spsa(options=options)
    expected = first.minimize(objective, initial_point).x
    second: Any = build_spsa(options=options)
    other: Any = build_spsa(
        options=SPSAOptions(seed=12345, maxiter=2, learning_rate=0.1, perturbation=0.1)
    )
    other.minimize(objective, initial_point)
    actual = second.minimize(objective, initial_point).x
    np.testing.assert_array_equal(actual, expected)
    assert utils.algorithm_globals.random_seed == seed
    assert utils.algorithm_globals.random is generator
    assert generator.bit_generator.state == state
