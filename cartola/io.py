import os
import re
import glob
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scipy.optimize import minimize
from scipy.stats import poisson

from training_logger import LOGGER

from .config import (
    REQUIRED_CSV_COLUMNS,
    POISSON_MAX_GOALS,
    SEASON_BASE_YEAR,
    ROUNDS_PER_SEASON,
    calculate_temporal_id,
)
# -----------------------------------------------------------------------------
# Temporal ID defaults (para suportar multi-ano mesmo antes de migrar config.py)
# -----------------------------------------------------------------------------
try:
    from .config import SEASON_BASE_YEAR, ROUNDS_PER_SEASON
except Exception:
    SEASON_BASE_YEAR = 2023
    ROUNDS_PER_SEASON = 38


def _infer_season_from_path(path: str) -> Optional[int]:
    """Tenta inferir a temporada (ano) a partir do nome do arquivo."""
    if not path:
        return None
    name = os.path.basename(str(path))
    hits = re.findall(r"(20\d{2})", name)
    if not hits:
        return None
    try:
        return int(hits[-1])  # usa o último match (ex.: *_2025.csv)
    except Exception:
        return None


def _calculate_temporal_id(
    temporada: int,
    rodada_id: int,
    base_year: int = SEASON_BASE_YEAR,
    rounds_per_season: int = ROUNDS_PER_SEASON,
) -> int:
    temporada = int(temporada)
    rodada_id = int(rodada_id)
    if temporada < base_year:
        raise ValueError(f"temporada inválida: {temporada} (< {base_year})")
    if not (1 <= rodada_id <= rounds_per_season):
        raise ValueError(f"rodada_id inválida: {rodada_id} (esperado 1..{rounds_per_season})")
    return (temporada - base_year) * rounds_per_season + rodada_id


def _decompose_temporal_id(
    temporal_id: int,
    base_year: int = SEASON_BASE_YEAR,
    rounds_per_season: int = ROUNDS_PER_SEASON,
) -> Tuple[int, int]:
    temporal_id = int(temporal_id)
    if temporal_id < 1:
        raise ValueError(f"temporal_id inválido: {temporal_id} (>= 1)")
    temporada = base_year + (temporal_id - 1) // rounds_per_season
    rodada_id = 1 + (temporal_id - 1) % rounds_per_season
    return temporada, rodada_id


# =============================================================================
# UTIL
# =============================================================================

def safe_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        return int(x)
    except Exception:
        return None


def parse_bool(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer, float, np.floating)):
        try:
            return float(x) != 0.0
        except Exception:
            return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "sim", "s")


def parse_scout_json(x: Any) -> Dict[str, float]:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "{}":
        return {}
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return {}


def ensure_probability_simplex(p_home: float, p_draw: float, p_away: float) -> Tuple[float, float, float]:
    p_home = float(p_home) if p_home is not None else 0.0
    p_draw = float(p_draw) if p_draw is not None else 0.0
    p_away = float(p_away) if p_away is not None else 0.0
    total = p_home + p_draw + p_away
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (p_home / total, p_draw / total, p_away / total)


def _try_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def extract_season_from_partida_id(partida_id: Any) -> Optional[int]:
    """
    Extrai a temporada do partida_id.
    
    Formatos suportados:
    - BR_2023_2_5 -> 2023
    - BR_2024_5_5 -> 2024
    - 333874.0 (sem info) -> None
    """
    if partida_id is None:
        return None
    
    s = str(partida_id).strip()
    
    # Formato BR_YYYY_X_Y
    match = re.search(r'BR_(\d{4})_', s)
    if match:
        return int(match.group(1))
    
    # Outros formatos que contenham ano
    match = re.search(r'(20\d{2})', s)
    if match:
        year = int(match.group(1))
        if 2020 <= year <= 2030:  # Sanity check
            return year
    
    return None


def extract_season_from_filename(filepath: str) -> Optional[int]:
    """
    Extrai a temporada do nome do arquivo.
    
    Formatos suportados:
    - player_games_with_odds_2023.csv -> 2023
    - player_games_2024.csv -> 2024
    - dados_2025_completo.csv -> 2025
    """
    filename = os.path.basename(filepath)
    match = re.search(r'(20\d{2})', filename)
    if match:
        year = int(match.group(1))
        if 2020 <= year <= 2030:
            return year
    return None


# =============================================================================
# POISSON / xG
# =============================================================================

class PoissonModel:
    MAX_GOALS = POISSON_MAX_GOALS

    @staticmethod
    def poisson_prob(lam: float, k: int) -> float:
        return poisson.pmf(k, lam)

    @classmethod
    def match_probs(cls, lh: float, la: float) -> Tuple[float, float, float]:
        p_home = p_draw = p_away = 0.0
        for i in range(cls.MAX_GOALS + 1):
            pi = cls.poisson_prob(lh, i)
            for j in range(cls.MAX_GOALS + 1):
                pj = cls.poisson_prob(la, j)
                pij = pi * pj
                if i > j:
                    p_home += pij
                elif i == j:
                    p_draw += pij
                else:
                    p_away += pij
        return p_home, p_draw, p_away

    @classmethod
    def fit_lambdas(cls, p_home: float, p_draw: float, p_away: float) -> Tuple[float, float]:
        p_home, p_draw, p_away = ensure_probability_simplex(p_home, p_draw, p_away)

        def objective(params):
            lh, la = params
            if lh <= 0 or la <= 0:
                return 1e10
            ch, cd, ca = cls.match_probs(lh, la)
            return (ch - p_home) ** 2 + (cd - p_draw) ** 2 + (ca - p_away) ** 2

        x0 = [1.3, 1.0]
        if p_home > 0.55:
            x0 = [1.8, 0.8]
        elif p_away > 0.55:
            x0 = [0.8, 1.8]

        res = minimize(objective, x0, method="L-BFGS-B", bounds=[(0.1, 4.0), (0.1, 4.0)])

        if not getattr(res, "success", False) or res.x is None or len(res.x) != 2:
            return float(x0[0]), float(x0[1])

        lh, la = float(res.x[0]), float(res.x[1])
        lh = min(max(lh, 0.1), 4.0)
        la = min(max(la, 0.1), 4.0)
        return lh, la

    @classmethod
    def derived(cls, team_xg: float, opp_xg: float) -> Dict[str, float]:
        team_xg = max(float(team_xg), 0.0)
        opp_xg = max(float(opp_xg), 0.0)

        p_team_0 = float(cls.poisson_prob(team_xg, 0))
        p_team_1 = float(cls.poisson_prob(team_xg, 1))

        p_team_scores_1plus = 1.0 - p_team_0
        p_team_scores_2plus = 1.0 - (p_team_0 + p_team_1)

        p_clean_sheet = float(cls.poisson_prob(opp_xg, 0))
        p_concede_1plus = 1.0 - p_clean_sheet

        p_btts = 1.0 - p_team_0 - p_clean_sheet + (p_team_0 * p_clean_sheet)

        def clip01(x: float) -> float:
            return float(min(max(x, 0.0), 1.0))

        return {
            "team_xG": float(team_xg),
            "opp_xG": float(opp_xg),
            "p_clean_sheet": clip01(p_clean_sheet),
            "p_concede_1plus": clip01(p_concede_1plus),
            "p_team_scores_1plus": clip01(p_team_scores_1plus),
            "p_team_scores_2plus": clip01(p_team_scores_2plus),
            "p_btts": clip01(p_btts),
        }


@dataclass
class MatchOdds:
    home_team: str
    away_team: str
    home_id: int
    away_id: int
    odd_home: float
    odd_draw: float
    odd_away: float
    p_home: float = 0.0
    p_draw: float = 0.0
    p_away: float = 0.0
    lambda_home: float = 0.0
    lambda_away: float = 0.0

    def __post_init__(self):
        inv_sum = (1 / self.odd_home) + (1 / self.odd_draw) + (1 / self.odd_away)
        self.p_home = (1 / self.odd_home) / inv_sum
        self.p_draw = (1 / self.odd_draw) / inv_sum
        self.p_away = (1 / self.odd_away) / inv_sum

        s = self.p_home + self.p_draw + self.p_away
        if abs(s - 1.0) > 1e-6:
            self.p_home, self.p_draw, self.p_away = ensure_probability_simplex(self.p_home, self.p_draw, self.p_away)

        self.lambda_home, self.lambda_away = PoissonModel.fit_lambdas(self.p_home, self.p_draw, self.p_away)

    def get_xg_features(self, is_home: bool) -> Dict[str, float]:
        if is_home:
            team_xg, opp_xg = self.lambda_home, self.lambda_away
            p_win, p_lose = self.p_home, self.p_away
        else:
            team_xg, opp_xg = self.lambda_away, self.lambda_home
            p_win, p_lose = self.p_away, self.p_home

        f = PoissonModel.derived(team_xg, opp_xg)
        f["p_team_win"] = float(p_win)
        f["p_draw"] = float(self.p_draw)
        f["p_team_lose"] = float(p_lose)

        s = f["p_team_win"] + f["p_draw"] + f["p_team_lose"]
        if abs(s - 1.0) > 1e-3:
            f["p_team_win"], f["p_draw"], f["p_team_lose"] = ensure_probability_simplex(
                f["p_team_win"], f["p_draw"], f["p_team_lose"]
            )
        return f


class OddsCache:
    _cache: Dict[str, MatchOdds] = {}

    @staticmethod
    def key(p_home: float, p_draw: float, p_away: float) -> str:
        p_home, p_draw, p_away = ensure_probability_simplex(p_home, p_draw, p_away)
        return f"{p_home:.4f}_{p_draw:.4f}_{p_away:.4f}"

    @classmethod
    def get_or_create(cls, p_home: float, p_draw: float, p_away: float) -> MatchOdds:
        k = cls.key(p_home, p_draw, p_away)
        if k not in cls._cache:
            p_home, p_draw, p_away = ensure_probability_simplex(p_home, p_draw, p_away)
            oh = 1.0 / max(p_home, 1e-6)
            od = 1.0 / max(p_draw, 1e-6)
            oa = 1.0 / max(p_away, 1e-6)
            cls._cache[k] = MatchOdds("H", "A", 0, 0, oh, od, oa)
        return cls._cache[k]

    @classmethod
    def clear(cls):
        cls._cache = {}

    @classmethod
    def size(cls) -> int:
        return len(cls._cache)


# =============================================================================
# API
# =============================================================================

class CartolaAPI:
    BASE_URL = "https://api.cartola.globo.com"

    def __init__(self):
        self.clubes = {}
        self.clubes_by_abbr = {}
        self.partidas = []
        self.partidas_rodada = None
        self.status = None

    def _get(self, endpoint: str) -> Optional[Dict]:
        try:
            r = requests.get(f"{self.BASE_URL}/{endpoint}", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            LOGGER.error(f"API erro ({endpoint}): {e}")
            return None

    def get_status(self) -> Optional[Dict]:
        self.status = self._get("mercado/status")
        return self.status

    def get_mercado(self) -> Tuple[List[Dict], Dict, Dict]:
        data = self._get("atletas/mercado")
        if not data:
            return [], {}, {}
        self.clubes = data.get("clubes", {})
        self.clubes_by_abbr = {}
        for cid, info in self.clubes.items():
            ab = (info.get("abreviacao") or "").upper()
            if ab:
                self.clubes_by_abbr[ab] = int(cid)
        return data.get("atletas", []), self.clubes, data.get("posicoes", {})

    def get_partidas(self) -> List[Dict]:
        data = self._get("partidas")
        if data:
            self.partidas_rodada = data.get("rodada")
            self.partidas = data.get("partidas", [])
            for p in self.partidas:
                p["rodada"] = self.partidas_rodada
        return self.partidas

    def get_clube_abbr(self, clube_id: int) -> str:
        return (self.clubes.get(str(clube_id), {}).get("abreviacao") or str(clube_id)).upper()

    def get_clube_id(self, abbr: str) -> Optional[int]:
        if not abbr:
            return None
        return self.clubes_by_abbr.get(str(abbr).upper())

    def get_match_for_clube(self, clube_id: int) -> Optional[Dict]:
        hits = []
        for p in self.partidas:
            if p.get("clube_casa_id") == clube_id:
                hits.append({"opponent_id": p.get("clube_visitante_id"), "is_home": True})
            elif p.get("clube_visitante_id") == clube_id:
                hits.append({"opponent_id": p.get("clube_casa_id"), "is_home": False})
        if len(hits) == 1:
            return hits[0]
        return None


# =============================================================================
# CARREGAMENTO DE DADOS - MULTI-TEMPORADA
# =============================================================================

def _process_dataframe(df: pd.DataFrame, source_season: Optional[int] = None) -> pd.DataFrame:
    """
    Processa um DataFrame adicionando colunas temporada e temporal_id.
    
    Args:
        df: DataFrame com dados de jogadores
        source_season: Temporada conhecida (do nome do arquivo). 
                       Se None, tenta extrair do partida_id.
    
    Returns:
        DataFrame processado com colunas temporada e temporal_id
    """
    df = df.copy()
    
    # Converter colunas básicas
    df["rodada_id"] = df["rodada_id"].apply(safe_int)
    df["atleta_id"] = df["atleta_id"].apply(safe_int)
    df["posicao_id"] = df["posicao_id"].apply(safe_int)
    df["clube_id"] = df["clube_id"].apply(safe_int)
    df["opponent_id"] = df["opponent_id"].apply(safe_int)
    
    df["pontuacao"] = pd.to_numeric(df["pontuacao"], errors="coerce").fillna(0.0)
    df["entrou_em_campo"] = df["entrou_em_campo"].apply(parse_bool)
    df["is_home"] = df["is_home"].apply(parse_bool)
    
    df["p_team_win"] = pd.to_numeric(df["p_team_win"], errors="coerce")
    df["p_draw"] = pd.to_numeric(df["p_draw"], errors="coerce")
    df["p_team_lose"] = pd.to_numeric(df["p_team_lose"], errors="coerce")
    
    df["scout_dict"] = df["scout_json"].apply(parse_scout_json)
    
    # Extrair temporada
    if source_season is not None:
        # Temporada conhecida pelo nome do arquivo
        df["temporada"] = source_season
    elif "partida_id" in df.columns:
        # Tentar extrair do partida_id
        df["temporada"] = df["partida_id"].apply(extract_season_from_partida_id)
        # Preencher valores faltantes com fallback
        if df["temporada"].isna().any():
            # Tentar inferir do contexto ou usar valor padrão
            valid_seasons = df["temporada"].dropna().unique()
            if len(valid_seasons) == 1:
                df["temporada"] = df["temporada"].fillna(valid_seasons[0])
            else:
                LOGGER.warning("Algumas linhas não têm temporada identificável no partida_id")
    else:
        raise ValueError("Não foi possível identificar a temporada. "
                        "Forneça source_season ou inclua partida_id no CSV.")
    
    df["temporada"] = df["temporada"].apply(safe_int)
    
    # Calcular temporal_id
    def _calc_temporal_id(row):
        temporada = safe_int(row.get("temporada"))
        rodada_id = safe_int(row.get("rodada_id"))
        if temporada is None or rodada_id is None:
            return None
        return calculate_temporal_id(temporada, rodada_id)
    
    df["temporal_id"] = df.apply(_calc_temporal_id, axis=1)
    
    return df


def load_historical_data(csv_path: str, source_season: Optional[int] = None) -> pd.DataFrame:
    """
    Carrega dados históricos de um único arquivo CSV.
    
    Mantém compatibilidade com código existente, mas agora adiciona
    temporada e temporal_id.
    
    Args:
        csv_path: Caminho para o arquivo CSV
        source_season: Temporada opcional (se não especificada, tenta extrair)
        
    Returns:
        DataFrame com colunas temporada e temporal_id adicionadas
    """
    LOGGER.info(f"Carregando histórico: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV inválido. Faltam colunas: {missing}")

    # Se não foi passada temporada, tenta extrair do nome do arquivo
    if source_season is None:
        source_season = extract_season_from_filename(csv_path)
    
    df = _process_dataframe(df, source_season)
    
    df = df.sort_values(["temporal_id", "atleta_id"]).reset_index(drop=True)

    LOGGER.info(
        f"   ✓ {len(df)} registros | {df['atleta_id'].nunique()} jogadores | "
        f"Temporada(s): {sorted(df['temporada'].dropna().unique().tolist())} | "
        f"Temporal IDs: {df['temporal_id'].min()}-{df['temporal_id'].max()}"
    )
    return df


def load_multi_year_data(
    csv_paths: Union[str, List[str]],
    seasons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Carrega e combina dados de múltiplas temporadas.
    
    Args:
        csv_paths: Pode ser:
            - String com glob pattern (ex: "player_games_with_odds_*.csv")
            - Lista de caminhos de arquivos
        seasons: Lista opcional de temporadas correspondentes aos arquivos.
                 Se não fornecida, tenta extrair do nome/conteúdo dos arquivos.
    
    Returns:
        DataFrame combinado com todas as temporadas, ordenado por temporal_id
        
    Exemplos:
        # Via glob pattern
        df = load_multi_year_data("data/player_games_with_odds_*.csv")
        
        # Via lista de arquivos
        df = load_multi_year_data([
            "player_games_with_odds_2023.csv",
            "player_games_with_odds_2024.csv",
            "player_games_with_odds_2025.csv"
        ])
        
        # Com temporadas explícitas
        df = load_multi_year_data(
            ["dados_a.csv", "dados_b.csv"],
            seasons=[2023, 2024]
        )
    """
    # Resolver lista de arquivos
    if isinstance(csv_paths, str):
        if '*' in csv_paths or '?' in csv_paths:
            # É um glob pattern
            file_list = sorted(glob.glob(csv_paths))
        else:
            # É um único arquivo
            file_list = [csv_paths]
    else:
        file_list = list(csv_paths)
    
    if not file_list:
        raise ValueError(f"Nenhum arquivo encontrado: {csv_paths}")
    
    LOGGER.info(f"Carregando {len(file_list)} arquivo(s) de múltiplas temporadas...")
    
    # Verificar se seasons foi fornecida corretamente
    if seasons is not None and len(seasons) != len(file_list):
        raise ValueError(
            f"Número de temporadas ({len(seasons)}) não corresponde "
            f"ao número de arquivos ({len(file_list)})"
        )
    
    dfs = []
    for i, filepath in enumerate(file_list):
        if not os.path.exists(filepath):
            LOGGER.warning(f"Arquivo não encontrado: {filepath}")
            continue
            
        # Determinar temporada
        if seasons is not None:
            season = seasons[i]
        else:
            season = extract_season_from_filename(filepath)
        
        LOGGER.info(f"   → {filepath} (temporada={season})")
        
        # Carregar e validar
        df = pd.read_csv(filepath)
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
        if missing:
            LOGGER.error(f"Arquivo {filepath} inválido. Faltam colunas: {missing}")
            continue
        
        # Processar
        df = _process_dataframe(df, season)
        dfs.append(df)
    
    if not dfs:
        raise ValueError("Nenhum arquivo válido carregado")
    
    # Combinar todos os DataFrames
    combined = pd.concat(dfs, ignore_index=True)
    
    # Ordenar por temporal_id e atleta_id
    combined = combined.sort_values(["temporal_id", "atleta_id"]).reset_index(drop=True)
    
    # Log do resultado
    temporadas = sorted(combined["temporada"].dropna().unique().tolist())
    min_tid = combined["temporal_id"].min()
    max_tid = combined["temporal_id"].max()
    
    LOGGER.info(f"   ✓ Total: {len(combined)} registros | {combined['atleta_id'].nunique()} jogadores")
    LOGGER.info(f"   ✓ Temporadas: {temporadas}")
    LOGGER.info(f"   ✓ Temporal IDs: {min_tid} → {max_tid} ({max_tid - min_tid + 1} rodadas)")
    
    # Estatísticas por temporada
    for temp in temporadas:
        df_temp = combined[combined["temporada"] == temp]
        n_rows = len(df_temp)
        n_players = df_temp["atleta_id"].nunique()
        n_rounds = df_temp["rodada_id"].nunique()
        LOGGER.info(f"     Temporada {temp}: {n_rows} registros, {n_players} jogadores, {n_rounds} rodadas")
    
    return combined


def load_odds(
    txt_path: str,
    api: CartolaAPI
) -> Tuple[Dict[int, MatchOdds], Dict[str, Tuple[float, float, float]], Dict[str, int], List[str]]:
    LOGGER.info(f"Carregando odds: {txt_path}")

    odds_by_clube: Dict[int, MatchOdds] = {}
    odds_pairs: Dict[str, Tuple[float, float, float]] = {}
    odds_pair_counts: Dict[str, int] = defaultdict(int)
    unknown_abbrs: List[str] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for line in lines:
        parts = line.replace(";", " ").replace("\t", " ").split()
        if len(parts) < 5:
            continue

        h_abbr = parts[0].upper()
        a_abbr = parts[1].upper()
        oh = _try_float(parts[2])
        od = _try_float(parts[3])
        oa = _try_float(parts[4])

        if oh is None or od is None or oa is None:
            continue

        h_id = api.get_clube_id(h_abbr)
        a_id = api.get_clube_id(a_abbr)

        if h_id is None:
            unknown_abbrs.append(h_abbr)
        if a_id is None:
            unknown_abbrs.append(a_abbr)

        if h_id is None or a_id is None:
            continue

        key = f"{h_abbr}_{a_abbr}"
        odds_pair_counts[key] += 1

        match = MatchOdds(h_abbr, a_abbr, h_id, a_id, float(oh), float(od), float(oa))

        odds_by_clube[h_id] = match
        odds_by_clube[a_id] = match
        odds_pairs[key] = (float(oh), float(od), float(oa))

    LOGGER.info(f"   ✓ {len(odds_pairs)} partidas | odds_by_clube={len(odds_by_clube)}")
    return odds_by_clube, odds_pairs, dict(odds_pair_counts), unknown_abbrs


# =============================================================================
# PERSISTÊNCIA DE MODELOS
# =============================================================================

def save_models(models, fe, filepath: str = "cartola_models.pkl"):
    """
    Salva os modelos treinados e o feature engineer em um arquivo pickle.
    
    Args:
        models: PositionModels treinado
        fe: TemporalFeatureEngineer com dados históricos
        filepath: caminho do arquivo para salvar
    """
    save_data = {
        "models": models,
        "fe": fe,
        "timestamp": datetime.now().isoformat(),
        "version": "3.1-multiyear"
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)
    
    LOGGER.info(f"✓ Modelos salvos em: {filepath}")
    print(f"✓ Modelos salvos em: {filepath}")
    

def load_models(filepath: str = "cartola_models.pkl"):
    """
    Carrega os modelos treinados de um arquivo pickle.
    
    Args:
        filepath: caminho do arquivo
        
    Returns:
        tuple: (models, fe) - PositionModels e TemporalFeatureEngineer
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo de modelos não encontrado: {filepath}")
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    models = data["models"]
    fe = data["fe"]
    timestamp = data.get("timestamp", "unknown")
    version = data.get("version", "unknown")
    
    LOGGER.info(f"✓ Modelos carregados de: {filepath} (versão {version}, salvo em {timestamp})")
    print(f"✓ Modelos carregados de: {filepath}")
    print(f"   Versão: {version} | Salvo em: {timestamp}")
    
    return models, fe
