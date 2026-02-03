"""
CARTOLA FEATURES - Feature Engineering Temporal OTIMIZADO v2
============================================================
Versão corrigida com:
- Lookup "as-of" para temporal_id futuro (previsões)
- Sem iterrows - usa apply/vectorized
- reindex seguro (sem method="ffill" problemático)
- Streaks/trend vetorizados
"""

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from training_logger import LOGGER

from .config import ALL_SCOUTS, POSITIVE_SCOUTS, NEGATIVE_SCOUTS, MIN_GAMES, HALF_LIFE_ROUNDS
from .io import safe_int, parse_bool, MatchOdds


class TemporalFeatureEngineer:
    """
    Feature engineering temporal sem vazamento de dados.
    Todas as features são PRÉ-CALCULADAS no __init__ usando operações vetorizadas.
    
    Suporta lookup "as-of" para temporal_id que ainda não existe nos dados
    (necessário para gerar previsões da rodada atual).
    """
    
    def __init__(self, df: pd.DataFrame):
        LOGGER.info("Inicializando TemporalFeatureEngineer (vetorizado v2)...")
        
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

        self.df["is_home"] = self.df["is_home"].apply(parse_bool)

        self.max_temporal_id = int(self.df["temporal_id"].max())
        self.max_round = int(self.df["rodada_id"].max())
        
        # Ordenar uma vez
        self.df = self.df.sort_values(["atleta_id", "temporal_id"]).reset_index(drop=True)
        
        # PRÉ-CALCULAR TUDO
        self._build_temporal_tables()
        self._build_player_tables()
        self._build_scout_tables()
        
        LOGGER.info(f"   ✓ FeatureEngineer pronto (max_temporal_id={self.max_temporal_id})")

    # =========================================================================
    # HELPER: reindex seguro
    # =========================================================================
    
    def _safe_reindex_ffill(self, s: pd.Series, new_index: range) -> pd.Series:
        """Reindex com ffill de forma segura."""
        s = s.copy()
        s.index = s.index.astype(int)
        # Criar série com novo índice
        new_s = pd.Series(index=list(new_index), dtype=float)
        # Copiar valores existentes
        for idx in s.index:
            if idx in new_s.index:
                new_s.loc[idx] = s.loc[idx]
        # Forward fill
        new_s = new_s.ffill()
        return new_s

    # =========================================================================
    # TABELAS TEMPORAIS (pos/opp/team)
    # =========================================================================
    
    def _build_temporal_tables(self):
        """Constrói tabelas temporais vetorizadas."""
        LOGGER.debug("Construindo tabelas temporais...")

        idx_range = range(1, self.max_temporal_id + 2)

        # Média por posição vs oponente
        pos_opp_round = (
            self.df.groupby(["posicao_id", "opponent_id", "temporal_id"], dropna=True)["pontuacao"]
            .mean()
            .reset_index()
            .sort_values(["posicao_id", "opponent_id", "temporal_id"])
        )
        pos_opp_round["pos_opp_mean_asof"] = (
            pos_opp_round.groupby(["posicao_id", "opponent_id"])["pontuacao"]
            .transform(lambda s: s.expanding().mean().shift(1))
        )

        self.pos_opp_series: Dict[Tuple[int, int], pd.Series] = {}
        for (pos_id, opp_id), g in pos_opp_round.groupby(["posicao_id", "opponent_id"]):
            s = pd.Series(g["pos_opp_mean_asof"].values, index=g["temporal_id"].values.astype(int)).sort_index()
            s = self._safe_reindex_ffill(s, idx_range)
            self.pos_opp_series[(int(pos_id), int(opp_id))] = s

        # Média por posição
        pos_round = (
            self.df.groupby(["posicao_id", "temporal_id"], dropna=True)["pontuacao"]
            .mean()
            .reset_index()
            .sort_values(["posicao_id", "temporal_id"])
        )
        pos_round["pos_mean_asof"] = (
            pos_round.groupby("posicao_id")["pontuacao"]
            .transform(lambda s: s.expanding().mean().shift(1))
        )

        self.pos_series: Dict[int, pd.Series] = {}
        for pos_id, g in pos_round.groupby("posicao_id"):
            s = pd.Series(g["pos_mean_asof"].values, index=g["temporal_id"].values.astype(int)).sort_index()
            s = self._safe_reindex_ffill(s, idx_range)
            self.pos_series[int(pos_id)] = s

        # Média por time
        team_round = (
            self.df.groupby(["clube_id", "temporal_id"], dropna=True)["pontuacao"]
            .mean()
            .reset_index()
            .sort_values(["clube_id", "temporal_id"])
        )
        team_round["team_mean_asof"] = (
            team_round.groupby("clube_id")["pontuacao"]
            .transform(lambda s: s.expanding().mean().shift(1))
        )

        self.team_series: Dict[int, pd.Series] = {}
        for clube_id, g in team_round.groupby("clube_id"):
            s = pd.Series(g["team_mean_asof"].values, index=g["temporal_id"].values.astype(int)).sort_index()
            s = self._safe_reindex_ffill(s, idx_range)
            self.team_series[int(clube_id)] = s

        # Scouts concedidos por oponente
        self._build_opp_scout_tables()

    def _build_opp_scout_tables(self):
        """Constrói tabelas de scouts concedidos por oponente."""
        
        if "scout_dict" not in self.df.columns:
            self.opp_scout_series = {}
            return
        
        # Expandir scout_dict usando apply
        def extract_scouts(scout_dict):
            if scout_dict is None or not isinstance(scout_dict, dict):
                return {sc: 0.0 for sc in ALL_SCOUTS}
            result = {}
            for sc in ALL_SCOUTS:
                try:
                    result[sc] = float(scout_dict.get(sc, 0))
                except (TypeError, ValueError):
                    result[sc] = 0.0
            return result
        
        scout_expanded = self.df["scout_dict"].apply(extract_scouts).apply(pd.Series)
        scout_expanded["opponent_id"] = self.df["opponent_id"].values
        scout_expanded["temporal_id"] = self.df["temporal_id"].values
        
        idx_range = range(1, self.max_temporal_id + 2)
        self.opp_scout_series: Dict[Tuple[int, str], pd.Series] = {}
        
        for sc in ALL_SCOUTS:
            if sc not in scout_expanded.columns:
                continue
            
            agg = (
                scout_expanded.groupby(["opponent_id", "temporal_id"])[sc]
                .mean()
                .reset_index()
                .sort_values(["opponent_id", "temporal_id"])
            )
            
            agg["asof"] = (
                agg.groupby("opponent_id")[sc]
                .transform(lambda s: s.expanding().mean().shift(1))
            )
            
            for opp_id, g in agg.groupby("opponent_id"):
                s = pd.Series(g["asof"].values, index=g["temporal_id"].values.astype(int)).sort_index()
                s = self._safe_reindex_ffill(s, idx_range)
                self.opp_scout_series[(int(opp_id), sc)] = s

    # =========================================================================
    # TABELAS DE JOGADOR
    # =========================================================================
    
    def _build_player_tables(self):
        """Pré-calcula features de jogador."""
        LOGGER.debug("Construindo tabelas de jogador...")
        
        df = self.df.copy()
        g = df.groupby("atleta_id")
        
        # Features básicas
        df["_player_mean"] = g["pontuacao"].transform(lambda s: s.expanding().mean().shift(1))
        df["_player_std"] = g["pontuacao"].transform(lambda s: s.expanding().std().shift(1))
        df["_player_count"] = g["pontuacao"].transform(lambda s: s.expanding().count().shift(1))
        df["_player_max"] = g["pontuacao"].transform(lambda s: s.expanding().max().shift(1))
        df["_player_min"] = g["pontuacao"].transform(lambda s: s.expanding().min().shift(1))
        df["_player_median"] = g["pontuacao"].transform(lambda s: s.expanding().median().shift(1))
        
        # Rolling
        df["_player_last3_mean"] = g["pontuacao"].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        df["_player_last3_std"] = g["pontuacao"].transform(lambda s: s.rolling(3, min_periods=1).std().shift(1))
        df["_player_last5_mean"] = g["pontuacao"].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
        df["_player_last5_std"] = g["pontuacao"].transform(lambda s: s.rolling(5, min_periods=1).std().shift(1))
        df["_player_last10_mean"] = g["pontuacao"].transform(lambda s: s.rolling(10, min_periods=1).mean().shift(1))
        df["_player_last10_std"] = g["pontuacao"].transform(lambda s: s.rolling(10, min_periods=1).std().shift(1))
        
        df["_player_last_score"] = g["pontuacao"].shift(1)
        
        # EWM
        df["_player_ewm"] = g["pontuacao"].transform(
            lambda s: s.ewm(halflife=HALF_LIFE_ROUNDS, min_periods=1).mean().shift(1)
        )
        
        # Trend (rolling apply)
        def calc_slope(window):
            if len(window) < 3:
                return 0.0
            x = np.arange(len(window))
            try:
                return np.polyfit(x, window, 1)[0]
            except:
                return 0.0
        
        df["_player_trend"] = g["pontuacao"].transform(
            lambda s: s.rolling(10, min_periods=3).apply(calc_slope, raw=True).shift(1)
        )
        
        # Home/Away (precisa loop por jogador infelizmente)
        df["_player_home_avg"] = np.nan
        df["_player_away_avg"] = np.nan
        
        for atleta_id, gdf in g:
            gdf_sorted = gdf.sort_values("temporal_id")
            idx_list = gdf_sorted.index.tolist()
            
            home_cumsum = 0.0
            home_count = 0
            away_cumsum = 0.0
            away_count = 0
            
            for i, idx in enumerate(idx_list):
                # Valor AS-OF (antes deste jogo)
                if home_count > 0:
                    df.loc[idx, "_player_home_avg"] = home_cumsum / home_count
                if away_count > 0:
                    df.loc[idx, "_player_away_avg"] = away_cumsum / away_count
                
                # Atualizar para próxima iteração
                pts = gdf_sorted.loc[idx, "pontuacao"]
                is_home = gdf_sorted.loc[idx, "is_home"]
                if is_home:
                    home_cumsum += pts
                    home_count += 1
                else:
                    away_cumsum += pts
                    away_count += 1
        
        # Derivadas
        df["_player_cv"] = df["_player_std"] / (df["_player_mean"].abs() + 0.1)
        df["_player_form"] = df["_player_last5_mean"] - df["_player_mean"]
        
        home_filled = df["_player_home_avg"].fillna(df["_player_mean"])
        away_filled = df["_player_away_avg"].fillna(df["_player_mean"])
        df["_player_home_advantage"] = home_filled - away_filled
        
        # Streak (simplificado - vetorizado)
        df["_above_mean"] = (df["pontuacao"] > df["_player_mean"]).astype(int)
        df["_player_positive_streak"] = g["_above_mean"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=1).sum()
        ).fillna(0)
        
        # Consistency
        df["_player_consistency"] = g["_above_mean"].transform(
            lambda s: s.shift(1).expanding().mean()
        ).fillna(0.5)
        
        # Guardar
        self._player_features_df = df[[
            "atleta_id", "temporal_id",
            "_player_mean", "_player_std", "_player_count", "_player_max", "_player_min",
            "_player_median", "_player_last3_mean", "_player_last3_std",
            "_player_last5_mean", "_player_last5_std", "_player_last10_mean", "_player_last10_std",
            "_player_last_score", "_player_ewm", "_player_trend", "_player_form",
            "_player_home_avg", "_player_away_avg", "_player_cv", "_player_home_advantage",
            "_player_positive_streak", "_player_consistency"
        ]].copy()
        
        self._player_features_df = self._player_features_df.sort_values(["atleta_id", "temporal_id"])
        self._player_max_tid = self._player_features_df.groupby("atleta_id")["temporal_id"].max().to_dict()
        
        LOGGER.debug(f"   ✓ Tabelas de jogador prontas ({len(self._player_features_df)} registros)")

    # =========================================================================
    # TABELAS DE SCOUT
    # =========================================================================
    
    def _build_scout_tables(self):
        """Pré-calcula features de scout."""
        LOGGER.debug("Construindo tabelas de scout...")
        
        if "scout_dict" not in self.df.columns:
            self._scout_features_df = pd.DataFrame()
            self._scout_max_tid = {}
            return
        
        def extract_scouts(scout_dict):
            if scout_dict is None or not isinstance(scout_dict, dict):
                return {f"scout_{sc}": 0.0 for sc in ALL_SCOUTS}
            result = {}
            for sc in ALL_SCOUTS:
                try:
                    result[f"scout_{sc}"] = float(scout_dict.get(sc, 0))
                except (TypeError, ValueError):
                    result[f"scout_{sc}"] = 0.0
            return result
        
        scout_expanded = self.df["scout_dict"].apply(extract_scouts).apply(pd.Series)
        scout_expanded["atleta_id"] = self.df["atleta_id"].values
        scout_expanded["temporal_id"] = self.df["temporal_id"].values
        
        scout_df = scout_expanded.sort_values(["atleta_id", "temporal_id"]).reset_index(drop=True)
        g = scout_df.groupby("atleta_id")
        
        for sc in ALL_SCOUTS:
            col = f"scout_{sc}"
            if col in scout_df.columns:
                scout_df[f"_{col}_avg"] = g[col].transform(
                    lambda s: s.expanding().mean().shift(1)
                ).fillna(0.0)
        
        # Derivadas
        pos_cols = [f"_scout_{s}_avg" for s in POSITIVE_SCOUTS if f"_scout_{s}_avg" in scout_df.columns]
        neg_cols = [f"_scout_{s}_avg" for s in NEGATIVE_SCOUTS if f"_scout_{s}_avg" in scout_df.columns]
        
        scout_df["_pos_total"] = scout_df[pos_cols].sum(axis=1) if pos_cols else 0
        scout_df["_neg_total"] = scout_df[neg_cols].sum(axis=1) if neg_cols else 0
        scout_df["_scout_positive_ratio"] = scout_df["_pos_total"] / (scout_df["_pos_total"] + scout_df["_neg_total"] + 0.01)
        
        # Profiles
        def safe_get(df, col, default=0):
            return df[col] if col in df.columns else default
        
        scout_df["_profile_finisher"] = (
            safe_get(scout_df, "_scout_G_avg", 0) * 2 + 
            safe_get(scout_df, "_scout_FD_avg", 0) + 
            safe_get(scout_df, "_scout_FT_avg", 0)
        )
        scout_df["_profile_playmaker"] = (
            safe_get(scout_df, "_scout_A_avg", 0) * 2 + 
            safe_get(scout_df, "_scout_DS_avg", 0)
        )
        scout_df["_profile_defender"] = (
            safe_get(scout_df, "_scout_DS_avg", 0) * 2 + 
            safe_get(scout_df, "_scout_SG_avg", 0)
        )
        
        cols_to_keep = ["atleta_id", "temporal_id"] + [c for c in scout_df.columns if c.startswith("_")]
        self._scout_features_df = scout_df[cols_to_keep].copy()
        self._scout_max_tid = self._scout_features_df.groupby("atleta_id")["temporal_id"].max().to_dict()
        
        LOGGER.debug(f"   ✓ Tabelas de scout prontas ({len(self._scout_features_df)} registros)")

    # =========================================================================
    # GETTERS COM LOOKUP AS-OF
    # =========================================================================

    def _get_pos_vs_opp_mean_asof(self, posicao_id: int, opponent_id: int, temporal_id: int) -> float:
        s = self.pos_opp_series.get((posicao_id, opponent_id))
        if s is not None:
            tid = min(temporal_id, self.max_temporal_id + 1)
            tid = max(tid, 1)
            if tid in s.index and pd.notna(s.loc[tid]):
                return float(s.loc[tid])
        return self._get_pos_mean_asof(posicao_id, temporal_id)

    def _get_pos_mean_asof(self, posicao_id: int, temporal_id: int) -> float:
        s = self.pos_series.get(posicao_id)
        if s is not None:
            tid = min(temporal_id, self.max_temporal_id + 1)
            tid = max(tid, 1)
            if tid in s.index and pd.notna(s.loc[tid]):
                return float(s.loc[tid])
        return 3.0

    def _get_team_mean_asof(self, clube_id: int, temporal_id: int) -> float:
        s = self.team_series.get(clube_id)
        if s is not None:
            tid = min(temporal_id, self.max_temporal_id + 1)
            tid = max(tid, 1)
            if tid in s.index and pd.notna(s.loc[tid]):
                return float(s.loc[tid])
        return 0.0

    def _get_opp_concedes_scout_asof(self, opponent_id: int, scout: str, temporal_id: int) -> float:
        s = self.opp_scout_series.get((opponent_id, scout))
        if s is not None:
            tid = min(temporal_id, self.max_temporal_id + 1)
            tid = max(tid, 1)
            if tid in s.index and pd.notna(s.loc[tid]):
                return float(s.loc[tid])
        return 0.0

    def get_player_features(self, atleta_id: int, temporal_id: int) -> Optional[Dict[str, float]]:
        """
        Retorna features do jogador via lookup AS-OF.
        Para temporal_id futuro, retorna features do último temporal_id disponível.
        """
        max_tid_available = self._player_max_tid.get(atleta_id)
        if max_tid_available is None:
            return None
        
        # AS-OF: usar o menor entre solicitado e disponível
        lookup_tid = min(temporal_id, max_tid_available)
        
        player_data = self._player_features_df[
            (self._player_features_df["atleta_id"] == atleta_id) & 
            (self._player_features_df["temporal_id"] == lookup_tid)
        ]
        
        if player_data.empty:
            # Fallback: último registro
            player_data = self._player_features_df[
                self._player_features_df["atleta_id"] == atleta_id
            ].tail(1)
            if player_data.empty:
                return None
        
        row = player_data.iloc[0]
        
        count = row.get("_player_count", 0)
        if pd.isna(count) or count < MIN_GAMES:
            return None
        
        f: Dict[str, float] = {}
        
        f["player_games"] = float(count)
        f["player_mean"] = float(row.get("_player_mean", 0) or 0)
        f["player_std"] = float(row.get("_player_std", 0) or 0)
        f["player_median"] = float(row.get("_player_median", 0) or 0)
        f["player_max"] = float(row.get("_player_max", 0) or 0)
        f["player_min"] = float(row.get("_player_min", 0) or 0)
        f["player_cv"] = float(row.get("_player_cv", 0) or 0)
        
        f["player_last3_mean"] = float(row.get("_player_last3_mean", 0) or 0)
        f["player_last3_std"] = float(row.get("_player_last3_std", 0) or 0)
        f["player_last5_mean"] = float(row.get("_player_last5_mean", 0) or 0)
        f["player_last5_std"] = float(row.get("_player_last5_std", 0) or 0)
        f["player_last10_mean"] = float(row.get("_player_last10_mean", 0) or 0)
        f["player_last10_std"] = float(row.get("_player_last10_std", 0) or 0)
        
        f["player_last_score"] = float(row.get("_player_last_score", 0) or 0)
        f["player_trend"] = float(row.get("_player_trend", 0) or 0)
        f["player_form"] = float(row.get("_player_form", 0) or 0)
        
        f["player_home_avg"] = float(row.get("_player_home_avg") or f["player_mean"])
        f["player_away_avg"] = float(row.get("_player_away_avg") or f["player_mean"])
        f["player_home_advantage"] = float(row.get("_player_home_advantage", 0) or 0)
        
        f["player_pos_streak"] = float(row.get("_player_positive_streak", 0) or 0)
        f["player_positive_streak"] = float(row.get("_player_positive_streak", 0) or 0)
        f["player_consistency"] = float(row.get("_player_consistency", 0.5) or 0.5)
        
        return f

    def get_scout_features(self, atleta_id: int, temporal_id: int) -> Dict[str, float]:
        """Retorna features de scout via lookup AS-OF."""
        f: Dict[str, float] = {}
        
        for s in ALL_SCOUTS:
            f[f"scout_{s}_avg"] = 0.0
        f["scout_positive_ratio"] = 0.5
        f["profile_finisher"] = 0.0
        f["profile_playmaker"] = 0.0
        f["profile_defender"] = 0.0
        
        if self._scout_features_df.empty:
            return f
        
        max_tid_available = self._scout_max_tid.get(atleta_id)
        if max_tid_available is None:
            return f
        
        lookup_tid = min(temporal_id, max_tid_available)
        
        scout_data = self._scout_features_df[
            (self._scout_features_df["atleta_id"] == atleta_id) & 
            (self._scout_features_df["temporal_id"] == lookup_tid)
        ]
        
        if scout_data.empty:
            scout_data = self._scout_features_df[
                self._scout_features_df["atleta_id"] == atleta_id
            ].tail(1)
            if scout_data.empty:
                return f
        
        row = scout_data.iloc[0]
        
        for s in ALL_SCOUTS:
            col = f"_scout_{s}_avg"
            val = row.get(col, 0)
            f[f"scout_{s}_avg"] = float(val) if pd.notna(val) else 0.0
        
        f["scout_positive_ratio"] = float(row.get("_scout_positive_ratio", 0.5) or 0.5)
        f["profile_finisher"] = float(row.get("_profile_finisher", 0) or 0)
        f["profile_playmaker"] = float(row.get("_profile_playmaker", 0) or 0)
        f["profile_defender"] = float(row.get("_profile_defender", 0) or 0)
        
        return f

    def get_opponent_features(self, opponent_id: int, posicao_id: int, temporal_id: int) -> Dict[str, float]:
        f: Dict[str, float] = {}
        for s in ALL_SCOUTS:
            f[f"opp_concedes_{s}"] = self._get_opp_concedes_scout_asof(opponent_id, s, temporal_id)
        f["pos_vs_opp_mean"] = self._get_pos_vs_opp_mean_asof(posicao_id, opponent_id, temporal_id)
        return f

    def get_team_features(self, clube_id: int, temporal_id: int) -> Dict[str, float]:
        return {"team_avg": self._get_team_mean_asof(clube_id, temporal_id)}

    def calculate_all_features(
        self,
        atleta_id: int,
        posicao_id: int,
        clube_id: int,
        opponent_id: int,
        match_odds: MatchOdds,
        is_home: bool,
        temporal_id: int,
    ) -> Optional[Dict[str, float]]:
        """Calcula todas as features. Suporta temporal_id futuro."""
        player_f = self.get_player_features(atleta_id, temporal_id)
        if player_f is None:
            return None

        f = dict(player_f)
        f.update(match_odds.get_xg_features(is_home))
        f.update(self.get_scout_features(atleta_id, temporal_id))
        f.update(self.get_opponent_features(opponent_id, posicao_id, temporal_id))
        f.update(self.get_team_features(clube_id, temporal_id))

        f["is_home"] = 1.0 if is_home else 0.0
        f["posicao_id"] = float(posicao_id)

        f["player_x_xG"] = float(f["player_mean"] * f["team_xG"])
        f["finisher_x_opp_goals"] = float(f["profile_finisher"] * f.get("opp_concedes_G", 0.0))
        f["defender_x_clean_sheet"] = float(f["profile_defender"] * f["p_clean_sheet"])
        f["home_x_player_home"] = float(f["is_home"] * f["player_home_avg"])

        if posicao_id == 1:
            f["pos_xg_relevance"] = float(f["p_clean_sheet"])
        elif posicao_id in [2, 3]:
            f["pos_xg_relevance"] = float(f["p_clean_sheet"] * 0.7 + f["p_team_scores_1plus"] * 0.3)
        elif posicao_id == 4:
            f["pos_xg_relevance"] = float(f["team_xG"])
        elif posicao_id == 5:
            f["pos_xg_relevance"] = float(f["p_team_scores_2plus"])
        else:
            f["pos_xg_relevance"] = float(f["team_xG"])

        return f
