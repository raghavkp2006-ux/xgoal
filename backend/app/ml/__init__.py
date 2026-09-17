"""Phase 2 — prediction models.

* :mod:`app.ml.dixon_coles` — primary Dixon-Coles Poisson scoreline model
* :mod:`app.ml.baselines` — Elo, base-rate, uniform and de-vigged closing odds
* :mod:`app.ml.features` — point-in-time features (stream form and the
  ``build_features(match_id, as_of)`` database contract)
* :mod:`app.ml.xgb_model` — XGBoost comparison model
* :mod:`app.ml.evaluation` — walk-forward harness
* :mod:`app.ml.store` — model version / artifact / prediction persistence

The package is deliberately DB-agnostic: jobs convert ORM rows into
``MatchInput`` objects, so every model is unit-testable without a database.
"""
