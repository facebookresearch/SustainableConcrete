"""Copy-on-access providers for the expensive shared GP fits.

Four tests across three modules each need the production strength GP, and
each was fitting its own copy:

    108.6s  setup  test_models.py::TestPredictiveQualityRegression
    108.3s  setup  test_models.py::TestGetModelListWithCost
    106.9s  call   test_lengthscale_identifiability
    102.9s  call   test_strength_curve_monotonicity

They are the *same* fit. ``SustainableConcreteModel.fit_strength_model``
calls ``fit_strength_gp(X, Y, Yvar, X_bounds)`` on ``data.strength_data``,
which is exactly what the other two call directly; ``fit_strength_gp``
defaults to ``seed=0`` and applies it via ``torch.manual_seed``, and every
caller uses that default. ``DATA_PATH`` is the default path for
``load_concrete_strength``, so the inputs match too. One fit therefore
serves all four.

Isolation:

* The fitted objects live in a closure. No module-level name is bound to
  them, so there is no reference a caller reaches for by accident, and
  every accessor returns ``copy.deepcopy``. This is a strong convention,
  not a hard guarantee -- the cache is still reachable in two hops via
  ``get_fitted_strength_gp.__closure__``, which takes deliberate effort.
* Measured, not assumed: mutating a copy's parameters leaves the original
  untouched, and the copy costs ~0.00s against a ~100s fit. Independence
  was also checked structurally -- zero shared tensor identities and zero
  shared storage between a copy and the original.
* Note ``fit_strength_model`` returns the model in EVAL mode (it computes
  a diagnostic MLL under ``torch.no_grad`` at the end of the fit), so a
  consumer calling ``.eval()`` is a no-op on state the original already
  has. Mode changes on a copy do not reach the original either way.
* ``copy.deepcopy`` drops GPyTorch's ``prediction_strategy`` (its
  ``__deepcopy__`` deliberately returns ``None``), so each copy rebuilds
  its posterior caches lazily. That is why copies cannot inherit or
  corrupt cached solves; posteriors were verified bit-identical to the
  original's.

The cache is per process, which is all that is needed: the suite runs
serially. Parallelism was evaluated and dropped -- see the note above
``test-py`` in the Makefile.
"""

from __future__ import annotations

import copy

from boxcrete.concrete_model import SustainableConcreteModel
from boxcrete.utils import load_concrete_strength

__all__ = ["get_fitted_concrete_model", "get_fitted_strength_gp"]


def _make_providers():
    """Build the accessors over a closure-private cache.

    Deliberately a factory: the fitted model is reachable only from inside
    this scope, so callers physically cannot obtain the shared instance.
    """
    cache: dict = {}

    def _ensure() -> dict:
        if not cache:
            data = load_concrete_strength()
            model = SustainableConcreteModel(strength_days=[1, 28])
            # Free -- constructed from coefficients, no optimisation.
            model.fit_gwp_model(data)
            # The single expensive fit. Called through the public method
            # rather than fit_strength_gp directly, so the production code
            # path under test is still the one exercised.
            model.fit_strength_model(data)
            cache["data"] = data
            cache["model"] = model
        return cache

    def get_fitted_concrete_model():
        """A fitted ``SustainableConcreteModel`` and its dataset.

        Both are fresh deepcopies; mutate them freely.
        """
        c = _ensure()
        return copy.deepcopy(c["model"]), copy.deepcopy(c["data"])

    def get_fitted_strength_gp():
        """``(gp, X, Y, Yvar, X_bounds)`` for the production strength GP.

        Equivalent to calling ``fit_strength_gp`` directly with the default
        seed; the GP is the one that method produced. Fresh deepcopy.
        """
        c = _ensure()
        X, Y, Yvar, X_bounds = c["data"].strength_data
        return (
            copy.deepcopy(c["model"].strength_model),
            X.clone(),
            Y.clone(),
            Yvar.clone() if Yvar is not None else None,
            X_bounds.clone() if X_bounds is not None else None,
        )

    return get_fitted_concrete_model, get_fitted_strength_gp


get_fitted_concrete_model, get_fitted_strength_gp = _make_providers()
