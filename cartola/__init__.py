from __future__ import annotations

from .config import (
    POSICOES, STATUS_ORD, SHEET_BY_POS,
    ALL_SCOUTS, POSITIVE_SCOUTS, NEGATIVE_SCOUTS,
    RANDOM_SEED, MIN_GAMES, BACKTEST_RODADAS, TOP_K_VALUES,
    CALIB_ROUNDS, CONFORMAL_ALPHA, HALF_LIFE_ROUNDS,
    OPTUNA_TRIALS, EARLY_STOPPING_ROUNDS, MAX_ESTIMATORS_TUNE,
    ENSEMBLE_SEEDS, BACKTEST_ENSEMBLE_SEEDS,  # NOVO
    get_default_paths,
    SEASON_BASE_YEAR, ROUNDS_PER_SEASON,
    calculate_temporal_id, decompose_temporal_id,
)

from .io import (
    safe_int, parse_bool, parse_scout_json, ensure_probability_simplex,
    PoissonModel, MatchOdds, OddsCache,
    CartolaAPI, load_historical_data, load_odds,
    save_models, load_models,
    load_multi_year_data,
)

from .validation import DataValidator
from .features import TemporalFeatureEngineer

from .models import (
    split_train_cal_by_round, _quantile_higher,
    temporal_folds_by_round, sort_by_group,
    make_relevance_labels_by_group, make_time_decay_weights,
    mean_lift_at_k_temporal,
    RankingModel, PositionModels, RankingMetrics,
    build_and_train, generate_predictions,
)

from .evaluation import RankingBacktester

from .report import (
    PlayerPrediction,
    print_backtest, print_importance, print_matches,
    preds_to_df, matches_to_df, importance_to_df,
    save_excel, _autosize_and_format_sheet,
)

__version__ = "3.1-optimized"

__all__ = [
    # Config
    "POSICOES", "STATUS_ORD", "SHEET_BY_POS", "get_default_paths",
    "ALL_SCOUTS", "POSITIVE_SCOUTS", "NEGATIVE_SCOUTS",
    "RANDOM_SEED", "MIN_GAMES", "BACKTEST_RODADAS", "TOP_K_VALUES",
    "CALIB_ROUNDS", "CONFORMAL_ALPHA", "HALF_LIFE_ROUNDS",
    "OPTUNA_TRIALS", "EARLY_STOPPING_ROUNDS", "MAX_ESTIMATORS_TUNE",
    "ENSEMBLE_SEEDS", "BACKTEST_ENSEMBLE_SEEDS",

    # Temporal
    "SEASON_BASE_YEAR", "ROUNDS_PER_SEASON",
    "calculate_temporal_id", "decompose_temporal_id",

    # IO
    "safe_int", "parse_bool", "parse_scout_json", "ensure_probability_simplex",
    "PoissonModel", "MatchOdds", "OddsCache",
    "CartolaAPI", "load_historical_data", "load_odds",
    "save_models", "load_models",
    "load_multi_year_data",

    # Validation
    "DataValidator",

    # Features
    "TemporalFeatureEngineer",

    # Models
    "split_train_cal_by_round", "_quantile_higher",
    "temporal_folds_by_round", "sort_by_group",
    "make_relevance_labels_by_group", "make_time_decay_weights",
    "mean_lift_at_k_temporal",
    "RankingModel", "PositionModels", "RankingMetrics",
    "build_and_train", "generate_predictions",

    # Evaluation
    "RankingBacktester",

    # Report
    "PlayerPrediction",
    "print_backtest", "print_importance", "print_matches",
    "preds_to_df", "matches_to_df", "importance_to_df",
    "save_excel", "_autosize_and_format_sheet",
]
