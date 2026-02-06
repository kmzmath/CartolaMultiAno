"""
CARTOLA FEATURES - Feature Engineering Temporal CORRIGIDO v3
============================================================
Correções:
1. Estruturas indexadas por (atleta_id, temporal_id) usando set_index e .loc
2. Eliminação completa de iterrows() - usa operações vetorizadas
3. Checagem apropriada de None/NaN usando pd.isna()
4. Lookup eficiente com dict/defaultdict para features de jogador
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd

from training_logger import LOGGER

from .config import ALL_SCOUTS, POSITIVE_SCOUTS, NEGATIVE_SCOUTS, MIN_GAMES, HALF_LIFE_ROUNDS
from .io import safe_int, parse_bool, MatchOdds


class TemporalFeatureEngineer:
    """
    Feature engineering temporal sem vazamento de dados.
    
    CORREÇÕES v3:
    1. DataFrames indexados por (atleta_id, temporal_id) para lookup O(1)
    2. Pré-cálculo vetorizado em vez de iterrows()
    3. Checagem apropriada de None/NaN
    """
    
    def __init__(self, df: pd.DataFrame):
        LOGGER.info("Inicializando TemporalFeatureEngineer (otimizado v3)...")

        self.df = df[df["entrou_em_campo"] == True].copy()

        if "temporal_id" not in self.df.columns:
            raise ValueError(
                "DataFrame precisa de coluna 'temporal_id'. "
                "Use load_multi_year_data() ou load_historical_data() para carregar dados."
            )

        # Converter tipos uma vez
        self.df["temporal_id"] = pd.to_numeric(self.df["temporal_id"], errors="coerce").fillna(0).astype(int)
        self.df["rodada_id"] = pd.to_numeric(self.df["rodada_id"], errors="coerce").fillna(0).astype(int)
        self.df["atleta_id"] = pd.to_numeric(self.df["atleta_id"], errors="coerce").fillna(0).astype(int)
        self.df["opponent_id"] = pd.to_numeric(self.df["opponent_id"], errors="coerce").fillna(0).astype(int)
        self.df["posicao_id"] = pd.to_numeric(self.df["posicao_id"], errors="coerce").fillna(0).astype(int)
        self.df["clube_id"] = pd.to_numeric(self.df["clube_id"], errors="coerce").fillna(0).astype(int)

        if "temporada" in self.df.columns:
            self.df["temporada"] = pd.to_numeric(self.df["temporada"], errors="coerce").fillna(0).astype(int)

        if "opponent_id" not in self.df.columns:
            raise ValueError("Histórico precisa de opponent_id.")

        # Ordenar por atleta e temporal_id para operações de janela
        self.df = self.df.sort_values(["atleta_id", "temporal_id"]).reset_index(drop=True)

        # =====================================================================
        # ESTRUTURAS INDEXADAS PARA LOOKUP RÁPIDO
        # =====================================================================

        self._player_games_indexed = self.df.set_index(["atleta_id", "temporal_id"])

        LOGGER.info("   Pré-calculando features de jogador...")
        self._player_features_df = self._build_player_features_vectorized()

        if self._has_scout_columns():
            self._scout_features_df = self._build_scout_features_vectorized()

            if self._scout_features_df is not None and not self._scout_features_df.empty:
                self._player_features_df = (
                    self._player_features_df
                    .join(self._scout_features_df, how="left")
                    .fillna(0.0)
                )

            self._opp_concedes_df = self._build_opp_concedes_features_vectorized()
        else:
            self._scout_features_df = None
            self._opp_concedes_df = None

        LOGGER.info("   Pré-calculando agregados por time/posição...")
        self._team_aggs = self._build_team_aggregates()
        self._pos_aggs = self._build_position_aggregates()
        self._pos_vs_opp_aggs = self._build_pos_vs_opp_aggregates()

        self._player_cache: Dict[Tuple[int, int], Dict] = {}

        LOGGER.info(f"   ✓ TemporalFeatureEngineer inicializado: {len(self.df)} registros")


    def _has_scout_columns(self) -> bool:
        """Verifica se há colunas de scout no DataFrame."""
        return any(col in self.df.columns for col in ALL_SCOUTS)
    
    
    def _build_player_features_vectorized(self) -> pd.DataFrame:
        """
        Features do jogador (sem leakage: tudo usa shift(1)).

        Volta a gerar:
        - player_std, player_cv, player_last10_std
        - player_form, player_consistency
        - player_home_advantage
        """
        player_groups = self.df.groupby("atleta_id", sort=False)
        features_list = []

        for atleta_id, group in player_groups:
            group = group.sort_values("temporal_id")

            pts = pd.to_numeric(group["pontuacao"], errors="coerce").fillna(0.0).astype(float)
            pts_prev = pts.shift(1)

            last5_mean = pts_prev.rolling(window=5, min_periods=1).mean()
            last3_mean = pts_prev.rolling(window=3, min_periods=1).mean()
            prev3_mean = pts.shift(4).rolling(window=3, min_periods=1).mean()

            player_mean = pts_prev.expanding(min_periods=1).mean()

            player_std = pts_prev.expanding(min_periods=2).std(ddof=0).fillna(0.0)
            last5_std = pts_prev.rolling(window=5, min_periods=2).std(ddof=0).fillna(0.0)
            last10_std = pts_prev.rolling(window=10, min_periods=2).std(ddof=0).fillna(0.0)

            denom = player_mean.abs().replace(0.0, np.nan)
            player_cv = (player_std / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            games_count = pts_prev.notna().cumsum().fillna(0).astype(int)
            trend = (last3_mean - prev3_mean).fillna(0.0)

            max_pts = pts_prev.expanding(min_periods=1).max().fillna(0.0)
            min_pts = pts_prev.expanding(min_periods=1).min().fillna(0.0)

            pts_positive = (pts_prev.fillna(0.0) > 0).astype(int)
            streak = pts_positive.groupby((pts_positive != pts_positive.shift()).cumsum()).cumcount() + 1
            streak = (streak * pts_positive).fillna(0).astype(int)

            is_home = group["is_home"].astype(bool)

            home_pts_prev = pts.where(is_home).shift(1)
            away_pts_prev = pts.where(~is_home).shift(1)

            home_sum = home_pts_prev.fillna(0.0).cumsum()
            away_sum = away_pts_prev.fillna(0.0).cumsum()

            # garanta dtype boolean (nullable) antes de qualquer coisa
            is_home = is_home.astype("boolean")

            # evita NaN no shift sem precisar fillna em object
            home_cnt = is_home.shift(1, fill_value=False).astype("int32").cumsum()
            away_cnt = (~is_home).shift(1, fill_value=False).astype("int32").cumsum()

            home_mean = (home_sum / home_cnt.replace(0, np.nan)).fillna(0.0)
            away_mean = (away_sum / away_cnt.replace(0, np.nan)).fillna(0.0)
            home_adv = (home_mean - away_mean).fillna(0.0)

            player_form = (last5_mean - player_mean).fillna(0.0)
            player_consistency = (1.0 / (1.0 + player_std)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

            player_df = pd.DataFrame({
                "atleta_id": atleta_id,
                "temporal_id": group["temporal_id"].values,

                "player_mean": player_mean.values,
                "player_std": player_std.values,
                "player_cv": player_cv.values,

                "player_last5_mean": last5_mean.values,
                "player_last5_std": last5_std.values,
                "player_last10_std": last10_std.values,

                "player_games": games_count.values,
                "player_trend": trend.values,

                "player_form": player_form.values,
                "player_consistency": player_consistency.values,
                "player_home_advantage": home_adv.values,

                "player_max": max_pts.values,
                "player_min": min_pts.values,
                "player_streak": streak.values,
            })

            features_list.append(player_df)

        if not features_list:
            return pd.DataFrame()

        result = pd.concat(features_list, ignore_index=True)
        return result.set_index(["atleta_id", "temporal_id"])


    def _build_scout_features_vectorized(self) -> pd.DataFrame:
        """
        Features de scout no formato do modelo antigo:
        - scout_<SCOUT>_avg: média last5 com shift(1)
        - scout_positive_ratio: (positivos last5) / (positivos+negativos last5)
        """
        scout_cols = [c for c in ALL_SCOUTS if c in self.df.columns]
        if not scout_cols:
            return pd.DataFrame()

        player_groups = self.df.groupby("atleta_id", sort=False)
        features_list = []

        pos_scouts = [s for s in POSITIVE_SCOUTS if s in self.df.columns]
        neg_scouts = [s for s in NEGATIVE_SCOUTS if s in self.df.columns]

        for atleta_id, group in player_groups:
            group = group.sort_values("temporal_id")
            out = {"atleta_id": atleta_id, "temporal_id": group["temporal_id"].values}

            for scout in scout_cols:
                vals = pd.to_numeric(group[scout], errors="coerce").fillna(0.0).astype(float)
                out[f"scout_{scout}_avg"] = vals.shift(1).rolling(window=5, min_periods=1).mean().values

            if pos_scouts:
                pos_sum = group[pos_scouts].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).astype(float)
                pos_last5 = pos_sum.shift(1).rolling(window=5, min_periods=1).sum()
            else:
                pos_last5 = pd.Series(0.0, index=group.index)

            if neg_scouts:
                neg_sum = group[neg_scouts].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).astype(float)
                neg_last5 = neg_sum.shift(1).rolling(window=5, min_periods=1).sum()
            else:
                neg_last5 = pd.Series(0.0, index=group.index)

            denom = (pos_last5 + neg_last5).replace(0.0, np.nan)
            ratio = (pos_last5 / denom).fillna(0.0).clip(0.0, 1.0)

            out["scout_positive_ratio"] = ratio.values
            features_list.append(pd.DataFrame(out))

        if not features_list:
            return pd.DataFrame()

        result = pd.concat(features_list, ignore_index=True)
        return result.set_index(["atleta_id", "temporal_id"])

    def _build_opp_concedes_features_vectorized(self) -> Optional[pd.DataFrame]:
        """
        Constrói (opponent_id, temporal_id) -> opp_concedes_<SCOUT>
        Usando a produção média de scouts dos jogadores que enfrentaram o oponente,
        com rolling last5 e shift(1) (sem leakage).
        """
        scout_cols = [c for c in ALL_SCOUTS if c in self.df.columns]
        if not scout_cols:
            return None

        opp_round = (
            self.df.groupby(["opponent_id", "temporal_id"], sort=True)[scout_cols]
            .mean()
            .sort_index()
        )

        def _roll(g: pd.DataFrame) -> pd.DataFrame:
            return g.shift(1).rolling(window=5, min_periods=1).mean()

        rolled = opp_round.groupby(level=0, group_keys=False).apply(_roll).fillna(0.0)
        rolled = rolled.rename(columns={c: f"opp_concedes_{c}" for c in scout_cols})
        return rolled


    def _build_team_aggregates(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Constrói agregados por time usando operações vetorizadas.
        
        Retorna dict[(clube_id, temporal_id)] -> {team_avg, team_std, ...}
        """
        team_aggs: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        team_groups = self.df.groupby("clube_id")
        
        for clube_id, group in team_groups:
            # Agregados por temporal_id (média do time na rodada ANTERIOR)
            round_groups = group.groupby("temporal_id")
            
            tids = sorted(round_groups.groups.keys())
            
            # Média móvel das últimas 5 rodadas do time
            round_means = round_groups["pontuacao"].mean()
            
            for i, tid in enumerate(tids):
                # Usar apenas dados de rodadas anteriores
                prev_tids = [t for t in tids[:i]]
                if not prev_tids:
                    team_aggs[(clube_id, tid)] = {
                        "team_avg": 0.0,
                        "team_std": 0.0,
                        "team_last5_avg": 0.0,
                    }
                    continue
                
                # Últimas 5 rodadas anteriores
                last5_tids = prev_tids[-5:]
                last5_vals = [round_means[t] for t in last5_tids if t in round_means.index]
                
                team_aggs[(clube_id, tid)] = {
                    "team_avg": float(np.mean(last5_vals)) if last5_vals else 0.0,
                    "team_std": float(np.std(last5_vals)) if len(last5_vals) > 1 else 0.0,
                    "team_last5_avg": float(np.mean(last5_vals[-5:])) if last5_vals else 0.0,
                }
        
        return team_aggs
    
    
    def _build_position_aggregates(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        Constrói agregados por posição usando operações vetorizadas.
        """
        pos_aggs: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        pos_groups = self.df.groupby("posicao_id")
        
        for pos_id, group in pos_groups:
            round_groups = group.groupby("temporal_id")
            tids = sorted(round_groups.groups.keys())
            round_means = round_groups["pontuacao"].mean()
            
            for i, tid in enumerate(tids):
                prev_tids = [t for t in tids[:i]]
                if not prev_tids:
                    pos_aggs[(pos_id, tid)] = {"pos_avg": 0.0}
                    continue
                
                last5_tids = prev_tids[-5:]
                last5_vals = [round_means[t] for t in last5_tids if t in round_means.index]
                
                pos_aggs[(pos_id, tid)] = {
                    "pos_avg": float(np.mean(last5_vals)) if last5_vals else 0.0,
                }
        
        return pos_aggs
    
    def _build_pos_vs_opp_aggregates(self) -> Dict[Tuple[int, int, int], Dict[str, float]]:
        """
        Constrói agregados de posição vs oponente.
        """
        pos_vs_opp: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        
        # Agrupar por (posicao_id, opponent_id)
        groups = self.df.groupby(["posicao_id", "opponent_id"])
        
        for (pos_id, opp_id), group in groups:
            round_groups = group.groupby("temporal_id")
            tids = sorted(round_groups.groups.keys())
            round_means = round_groups["pontuacao"].mean()
            
            for i, tid in enumerate(tids):
                prev_tids = [t for t in tids[:i]]
                if not prev_tids:
                    pos_vs_opp[(pos_id, opp_id, tid)] = {"pos_vs_opp_mean": 0.0}
                    continue
                
                last5_tids = prev_tids[-5:]
                last5_vals = [round_means[t] for t in last5_tids if t in round_means.index]
                
                pos_vs_opp[(pos_id, opp_id, tid)] = {
                    "pos_vs_opp_mean": float(np.mean(last5_vals)) if last5_vals else 0.0,
                }
        
        return pos_vs_opp
    
    def _get_player_features(self, atleta_id: int, temporal_id: int) -> Dict[str, float]:
        """
        Obtém features de jogador por lookup (atleta_id, temporal_id).

        Correção: se não existir o temporal_id (ex.: rodada futura sem histórico),
        faz fallback para o último temporal_id anterior disponível para esse atleta.
        """
        cache_key = (atleta_id, temporal_id)
        if cache_key in self._player_cache:
            return self._player_cache[cache_key]

        def _row_to_dict(row) -> Dict[str, float]:
            out = {}
            for col in row.index:
                val = row[col]
                out[col] = 0.0 if pd.isna(val) else float(val)
            return out

        # 1) tenta exato
        try:
            row = self._player_features_df.loc[(atleta_id, temporal_id)]
            result = _row_to_dict(row)
            self._player_cache[cache_key] = result
            return result
        except KeyError:
            pass

        # 2) fallback: procura para trás
        t = temporal_id - 1
        while t > 0:
            try:
                row = self._player_features_df.loc[(atleta_id, t)]
                result = _row_to_dict(row)
                self._player_cache[cache_key] = result
                return result
            except KeyError:
                t -= 1

        # 3) nada encontrado
        result = {
            "player_mean": 0.0,
            "player_last5_mean": 0.0,
            "player_last5_std": 0.0,
            "player_games": 0.0,
            "player_trend": 0.0,
            "player_max": 0.0,
            "player_min": 0.0,
            "player_streak": 0.0,
        }
        self._player_cache[cache_key] = result
        return result


    def _get_team_features(self, clube_id: int, temporal_id: int) -> Dict[str, float]:
        """Features de time com fallback para o último temporal_id disponível."""
        if not hasattr(self, "_team_cache"):
            self._team_cache = {}

        key = (clube_id, temporal_id)
        if key in self._team_cache:
            return self._team_cache[key]

        v = self._team_aggs.get(key)
        if v is not None:
            self._team_cache[key] = v
            return v

        t = temporal_id - 1
        while t > 0:
            k2 = (clube_id, t)
            v2 = self._team_aggs.get(k2)
            if v2 is not None:
                self._team_cache[key] = v2
                return v2
            t -= 1

        d = {"team_avg": 0.0, "team_std": 0.0, "team_last5_avg": 0.0}
        self._team_cache[key] = d
        return d


    def _get_position_features(self, posicao_id: int, temporal_id: int) -> Dict[str, float]:
        """Features de posição com fallback para o último temporal_id disponível."""
        if not hasattr(self, "_pos_cache"):
            self._pos_cache = {}

        key = (posicao_id, temporal_id)
        if key in self._pos_cache:
            return self._pos_cache[key]

        v = self._pos_aggs.get(key)
        if v is not None:
            self._pos_cache[key] = v
            return v

        t = temporal_id - 1
        while t > 0:
            k2 = (posicao_id, t)
            v2 = self._pos_aggs.get(k2)
            if v2 is not None:
                self._pos_cache[key] = v2
                return v2
            t -= 1

        d = {"pos_avg": 0.0}
        self._pos_cache[key] = d
        return d


    def _get_pos_vs_opp_features(self, posicao_id: int, opponent_id: int, temporal_id: int) -> Dict[str, float]:
        """Features posição vs oponente com fallback para o último temporal_id disponível."""
        if not hasattr(self, "_pos_opp_cache"):
            self._pos_opp_cache = {}

        key = (posicao_id, opponent_id, temporal_id)
        if key in self._pos_opp_cache:
            return self._pos_opp_cache[key]

        v = self._pos_vs_opp_aggs.get(key)
        if v is not None:
            self._pos_opp_cache[key] = v
            return v

        t = temporal_id - 1
        while t > 0:
            k2 = (posicao_id, opponent_id, t)
            v2 = self._pos_vs_opp_aggs.get(k2)
            if v2 is not None:
                self._pos_opp_cache[key] = v2
                return v2
            t -= 1

        d = {"pos_vs_opp_mean": 0.0}
        self._pos_opp_cache[key] = d
        return d


    def _get_match_features(self, match_odds: MatchOdds, is_home: bool) -> Dict[str, float]:
        """Extrai features do confronto (odds)."""
        if match_odds is None:
            return {
                "team_xG": 0.0,
                "opp_xG": 0.0,
                "p_team_win": 0.0,
                "p_draw": 0.0,
                "p_team_lose": 0.0,
                "p_clean_sheet": 0.0,
                "p_team_scores": 0.0,
                "p_team_scores_2plus": 0.0,
                "is_home": float(is_home),
            }
        
        if is_home:
            team_xG = match_odds.home_xG
            opp_xG = match_odds.away_xG
            p_win = match_odds.p_home
            p_lose = match_odds.p_away
            p_cs = match_odds.p_away_scores(0)
            p_scores = 1.0 - match_odds.p_home_scores(0)
            p_scores_2 = 1.0 - match_odds.p_home_scores(0) - match_odds.p_home_scores(1)
        else:
            team_xG = match_odds.away_xG
            opp_xG = match_odds.home_xG
            p_win = match_odds.p_away
            p_lose = match_odds.p_home
            p_cs = match_odds.p_home_scores(0)
            p_scores = 1.0 - match_odds.p_away_scores(0)
            p_scores_2 = 1.0 - match_odds.p_away_scores(0) - match_odds.p_away_scores(1)
        
        return {
            "team_xG": float(team_xG),
            "opp_xG": float(opp_xG),
            "p_team_win": float(p_win),
            "p_draw": float(match_odds.p_draw),
            "p_team_lose": float(p_lose),
            "p_clean_sheet": float(max(0, min(1, p_cs))),
            "p_team_scores": float(max(0, min(1, p_scores))),
            "p_team_scores_2plus": float(max(0, min(1, p_scores_2))),
            "is_home": float(is_home),
        }
    
    def calculate_all_features(
        self,
        atleta_id: int,
        posicao_id: int,
        clube_id: int,
        opponent_id: int,
        match_odds: MatchOdds,
        is_home: bool,
        temporal_id: int,
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}

        # jogador + scouts (já entram em _get_player_features)
        features.update(self._get_player_features(atleta_id, temporal_id))

        # time/posição/pos_vs_opp
        features.update(self._get_team_features(clube_id, temporal_id))
        features.update(self._get_position_features(posicao_id, temporal_id))
        features.update(self._get_pos_vs_opp_features(posicao_id, opponent_id, temporal_id))

        # confronto
        match_feats = self._get_match_features(match_odds, is_home)
        features.update(match_feats)

        # nome antigo (OLDMODEL)
        if "p_team_scores" in match_feats:
            features["p_team_scores_1plus"] = float(match_feats["p_team_scores"])

        # opp_concedes_*
        if getattr(self, "_opp_concedes_df", None) is not None:
            try:
                row = self._opp_concedes_df.loc[(opponent_id, temporal_id)]
                for col in row.index:
                    v = row[col]
                    features[col] = 0.0 if pd.isna(v) else float(v)
            except KeyError:
                pass

        # === Perfis/interações do modelo antigo ===
        g_avg = float(features.get("scout_G_avg", 0.0))
        shots_avg = float(features.get("scout_FD_avg", 0.0)) + float(features.get("scout_FF_avg", 0.0)) + float(features.get("scout_FT_avg", 0.0))

        if posicao_id in (4, 5):  # meia/atacante
            profile_finisher = float(np.clip(g_avg / (shots_avg + 0.20), 0.0, 1.0))
        else:
            profile_finisher = 0.0

        profile_defender = 1.0 if posicao_id in (1, 2, 3, 6) else 0.0

        team_xg = float(features.get("team_xG", 0.0))
        opp_xg = float(features.get("opp_xG", 0.0))
        opp_concedes_g = float(features.get("opp_concedes_G", opp_xg))

        features["profile_finisher"] = profile_finisher
        features["profile_defender"] = profile_defender
        features["player_x_xG"] = float(profile_finisher * team_xg)
        features["finisher_x_opp_goals"] = float(profile_finisher * opp_concedes_g)
        features["home_x_player_home"] = float((1.0 if is_home else 0.0) * float(features.get("player_home_advantage", 0.0)))

        features["posicao_id"] = float(posicao_id)

        # sanitização
        for k, v in list(features.items()):
            if v is None or pd.isna(v) or np.isinf(v):
                features[k] = 0.0
            else:
                try:
                    features[k] = float(v)
                except Exception:
                    features[k] = 0.0

        return features


    def get_historical_features(
        self,
        atleta_id: int,
        temporal_id: int,
        n_games: int = 5,
    ) -> List[Dict[str, float]]:
        """
        Retorna features históricas dos últimos n_games.
        
        Usa lookup indexado em vez de filtro sequencial.
        """
        try:
            # Filtrar jogos do atleta antes do temporal_id atual
            player_data = self._player_games_indexed.loc[atleta_id]
            
            if isinstance(player_data, pd.Series):
                # Apenas um registro
                return []
            
            # Filtrar temporal_ids anteriores
            historical = player_data[player_data.index < temporal_id]
            
            if len(historical) == 0:
                return []
            
            # Pegar últimos n_games
            last_n = historical.tail(n_games)
            
            result = []
            for tid in last_n.index:
                feats = self._get_player_features(atleta_id, tid)
                feats["temporal_id"] = tid
                result.append(feats)
            
            return result
            
        except KeyError:
            return []
