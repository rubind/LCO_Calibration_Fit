import numpy as np
import pytest


pytest.importorskip("jax")
pytest.importorskip("numpyro")

from passband_fit.numpyro_sampler import NumpyroSettings, run_numpyro_nuts


def gaussian_potential(theta):
    import jax.numpy as jnp

    return 0.5 * jnp.dot(theta, theta)


def test_parallel_chains_have_legacy_shape_and_diagnostics():
    settings = NumpyroSettings(
        num_warmup=15,
        num_samples=18,
        num_chains=2,
        seed=21,
        target_accept_prob=0.85,
        max_tree_depth=6,
        chain_method="parallel",
        progress_bar=False,
    )
    result = run_numpyro_nuts(
        gaussian_potential,
        [0.0, 0.0],
        settings,
        parameter_names=("x", "y"),
    )
    assert result.chain.shape == (18, 2, 2)
    assert result.log_probability.shape == (18, 2)
    np.testing.assert_allclose(
        result.log_probability,
        -0.5 * np.sum(result.chain**2, axis=-1),
        rtol=2e-5,
        atol=2e-5,
    )
    assert result.diagnostics.rhat.shape == (2,)
    assert result.diagnostics.effective_sample_size.shape == (2,)
    assert set(result.diagnostics.rhat_by_parameter) == {"x", "y"}
    assert result.diagnostics.effective_chain_method in {"parallel", "sequential"}
    if result.diagnostics.effective_chain_method == "sequential":
        assert any("using sequential chains" in item for item in result.diagnostics.warnings)
    assert set(result.extra_fields) == {
        "accept_prob",
        "diverging",
        "num_steps",
        "potential_energy",
    }
    assert all(value.shape == (18, 2) for value in result.extra_fields.values())


def test_thinning_retains_requested_draw_count_and_single_chain_rhat_is_nan():
    settings = NumpyroSettings(
        num_warmup=10,
        num_samples=9,
        num_chains=1,
        thinning=3,
        seed=2,
        chain_method="sequential",
        progress_bar=False,
    )
    result = run_numpyro_nuts(gaussian_potential, [0.1], settings)
    assert result.chain.shape == (9, 1, 1)
    assert result.parameter_names == ("theta[0]",)
    assert np.isnan(result.diagnostics.rhat[0])
    assert np.isfinite(result.diagnostics.effective_sample_size[0])
    assert any("at least two chains" in item for item in result.diagnostics.warnings)
    flat_chain, flat_log_probability = result.flattened(discard=2)
    assert flat_chain.shape == (7, 1)
    assert flat_log_probability.shape == (7,)


def test_seed_is_reproducible_independent_of_numpy_global_rng():
    settings = NumpyroSettings(
        num_warmup=8,
        num_samples=8,
        num_chains=1,
        seed=99,
        chain_method="sequential",
        progress_bar=False,
    )
    np.random.seed(1)
    first = run_numpyro_nuts(gaussian_potential, [0.2, -0.1], settings)
    np.random.seed(987654)
    second = run_numpyro_nuts(gaussian_potential, [0.2, -0.1], settings)
    np.testing.assert_array_equal(first.chain, second.chain)
    np.testing.assert_array_equal(first.log_probability, second.log_probability)


@pytest.mark.parametrize(
    "settings",
    [
        NumpyroSettings(num_warmup=-1),
        NumpyroSettings(num_samples=0),
        NumpyroSettings(num_chains=0),
        NumpyroSettings(thinning=0),
        NumpyroSettings(seed=-1),
        NumpyroSettings(target_accept_prob=1.0),
        NumpyroSettings(max_tree_depth=0),
        NumpyroSettings(chain_method="bad"),
    ],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_distinct_initial_positions_and_names_are_validated():
    settings = NumpyroSettings(
        num_warmup=5,
        num_samples=5,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )
    result = run_numpyro_nuts(
        gaussian_potential,
        [[-0.2, 0.1], [0.2, -0.1]],
        settings,
        parameter_names=("left", "right"),
    )
    assert result.chain.shape == (5, 2, 2)
    with pytest.raises(ValueError, match="parameter_names"):
        run_numpyro_nuts(
            gaussian_potential,
            [0.0, 0.0],
            settings,
            parameter_names=("only-one",),
        )


def test_nonfinite_initial_potential_is_rejected():
    def invalid_potential(theta):
        import jax.numpy as jnp

        return jnp.asarray(np.nan)

    with pytest.raises(ValueError, match="finite scalar"):
        run_numpyro_nuts(
            invalid_potential,
            [0.0],
            NumpyroSettings(
                num_warmup=1,
                num_samples=1,
                num_chains=1,
                chain_method="sequential",
                progress_bar=False,
            ),
        )

