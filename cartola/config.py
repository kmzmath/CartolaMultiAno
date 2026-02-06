"""
CARTOLA CONFIG - Configurações CORRIGIDAS v3
=============================================
Correções:
1. Parâmetros default com regularização mais forte
2. Ranges de Optuna mais conservadores
3. K consistente (5) para tuning e backtest
"""

from typing import Tuple

# =============================================================================
# TEMPORAL ID - Multi-temporada
# =============================================================================

SEASON_BASE_YEAR = 2023
ROUNDS_PER_SEASON = 38


def calculate_temporal_id(temporada: int, rodada_id: int) -> int:
    """
    Calcula o temporal_id único para uma combinação temporada/rodada.
    temporal_id = (temporada - SEASON_BASE_YEAR) * ROUNDS_PER_SEASON + rodada_id
    """
    return (temporada - SEASON_BASE_YEAR) * ROUNDS_PER_SEASON + rodada_id


def decompose_temporal_id(temporal_id: int) -> Tuple[int, int]:
    """Decompõe o temporal_id de volta em temporada e rodada_id."""
    temporal_id_zero_based = temporal_id - 1
    season_offset = temporal_id_zero_based // ROUNDS_PER_SEASON
    rodada_id = (temporal_id_zero_based % ROUNDS_PER_SEASON) + 1
    temporada = SEASON_BASE_YEAR + season_offset
    return temporada, rodada_id


# =============================================================================
# COLUNAS ESPERADAS NO CSV
# =============================================================================

REQUIRED_CSV_COLUMNS = [
    "rodada_id", "atleta_id", "pontuacao", "entrou_em_campo", "scout_json",
    "posicao_id", "clube_id", "opponent_id", "is_home",
    "p_team_win", "p_draw", "p_team_lose",
]

GENERATED_COLUMNS = [
    "temporada",
    "temporal_id",
]


# =============================================================================
# MAPEAMENTOS
# =============================================================================

STATUS_ORD = {"Provável": 0, "Dúvida": 1}

POSICOES = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}

SHEET_BY_POS = {
    1: "Goleiros",
    2: "Laterais",
    3: "Zagueiros",
    4: "Meias",
    5: "Atacantes",
    6: "Tecnicos"
}


# =============================================================================
# SCOUTS
# =============================================================================

ALL_SCOUTS = [
    "G", "A", "FT", "FD", "FF", "FS", "PS", "I", "SG", "DP", "DE", "DS",
    "FC", "CA", "CV", "GC", "GS", "PC"
]

POSITIVE_SCOUTS = ["G", "A", "FT", "FD", "FF", "FS", "PS", "SG", "DP", "DE", "DS"]
NEGATIVE_SCOUTS = ["I", "FC", "CA", "CV", "GC", "GS", "PC"]


# =============================================================================
# HIPERPARÂMETROS DE TREINO - REGULARIZAÇÃO FORTE
# =============================================================================

RANDOM_SEED = 42
MIN_GAMES = 3

# Backtest
BACKTEST_RODADAS = 19
TOP_K_VALUES = [5, 10, 15, 20, 30]

# K padrão para métricas (CONSISTENTE entre tuning e backtest)
DEFAULT_K = 5

# Calibração Conformal
CALIB_ROUNDS = 6
CONFORMAL_ALPHA = 0.20

# Time decay
HALF_LIFE_ROUNDS = 35

# Optuna + Early stopping
OPTUNA_TRIALS = 500
EARLY_STOPPING_ROUNDS = 50  # Reduzido de 100 para 50 (early stopping mais agressivo)
MAX_ESTIMATORS_TUNE = 50000  # Reduzido de 70000 para 50000


# =============================================================================
# PARÂMETROS DEFAULT - REGULARIZAÇÃO MAIS FORTE
# =============================================================================

# Valores default mais conservadores
DEFAULT_MAX_DEPTH = 4          # Era 8 -> 4 (REDUZIDO PARA EVITAR OVERFITTING)
DEFAULT_NUM_LEAVES = 31        # Era 63 -> 31 (REDUZIDO)
DEFAULT_MIN_CHILD_SAMPLES = 50 # Era 30 -> 50 (AUMENTADO)

# Ranges de otimização mais restritos
OPTUNA_MAX_DEPTH_RANGE = (3, 6)           # Era (3, 12) -> (3, 6)
OPTUNA_NUM_LEAVES_RANGE = (16, 63)        # Era (16, 255) -> (16, 63)
OPTUNA_MIN_CHILD_SAMPLES_RANGE = (30, 200) # Era (10, 200) -> (30, 200)


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

# Seeds para treino final (9 seeds = ensemble robusto)
ENSEMBLE_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122]

# Seeds para backtest (3 seeds = mais rápido, ainda representativo)
BACKTEST_ENSEMBLE_SEEDS = [42, 72, 102]


# =============================================================================
# POISSON
# =============================================================================

POISSON_MAX_GOALS = 10


# =============================================================================
# PATHS
# =============================================================================

def get_default_paths(base_dir: str | None = None):
    """Retorna caminhos padrão para os arquivos."""
    import os

    if base_dir is None:
        base_dir = os.getcwd()

    base_dir = os.path.abspath(base_dir)

    return {
        "csv": os.path.join(base_dir, "player_games_with_odds.csv"),
        "csv_pattern": os.path.join(base_dir, "player_games_with_odds_*.csv"),
        "odds": os.path.join(base_dir, "OddsCasas.txt"),
        "output": os.path.join(base_dir, "output", "cartola_predictions.xlsx"),
        "models": os.path.join(base_dir, "models", "cartola_models.pkl"),
        "logs_dir": os.path.join(base_dir, "logs"),
    }
