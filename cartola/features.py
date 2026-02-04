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
        
        # 1. DataFrame indexado por (atleta_id, temporal_id)
        self._player_games_indexed = self.df.set_index(["atleta_id", "temporal_id"])
        
        # 2. Pré-calcular features de jogador usando operações vetorizadas
        LOGGER.info("   Pré-calculando features de jogador...")
        self._player_features_df = self._build_player_features_vectorized()
        
        # 3. Pré-calcular features de scout (se disponível)
        if self._has_scout_columns():
            LOGGER.info("   Pré-calculando features de scout...")
            self._scout_features_df = self._build_scout_features_vectorized()
        else:
            self._scout_features_df = None
        
        # 4. Pré-calcular agregados por time/posição
        LOGGER.info("   Pré-calculando agregados por time/posição...")
        self._team_aggs = self._build_team_aggregates()
        self._pos_aggs = self._build_position_aggregates()
        self._pos_vs_opp_aggs = self._build_pos_vs_opp_aggregates()
        
        # 5. Caches para lookup rápido
        self._player_cache: Dict[Tuple[int, int], Dict] = {}
        
        LOGGER.info(f"   ✓ TemporalFeatureEngineer inicializado: {len(self.df)} registros")
    
    def _has_scout_columns(self) -> bool:
        """Verifica se há colunas de scout no DataFrame."""
        return any(col in self.df.columns for col in ALL_SCOUTS)
    
    def _build_player_features_vectorized(self) -> pd.DataFrame:
        """
        Constrói DataFrame de features de jogador usando operações VETORIZADAS.
        
        Evita iterrows() completamente.
        """
        # Agrupar por jogador
        player_groups = self.df.groupby("atleta_id")
        
        # Calcular estatísticas por jogador com shift(1) para evitar leakage
        features_list = []
        
        for atleta_id, group in player_groups:
            group = group.sort_values("temporal_id")
            
            # Usar shift(1) em todas as features para evitar data leakage
            # (a feature do temporal_id T usa apenas dados de T-1 e anteriores)
            
            # Pontuação acumulada com shift
            pts = group["pontuacao"].astype(float)
            
            # Média móvel (últimos 5 jogos, sem o atual)
            last5_mean = pts.shift(1).rolling(window=5, min_periods=1).mean()
            
            # Média geral (todos os jogos anteriores)
            cumsum = pts.shift(1).cumsum()
            cumcount = pd.Series(range(1, len(group) + 1), index=group.index)
            cumcount_shifted = cumcount - 1
            cumcount_shifted = cumcount_shifted.replace(0, np.nan)
            player_mean = cumsum / cumcount_shifted
            
            # Std dos últimos 5
            last5_std = pts.shift(1).rolling(window=5, min_periods=2).std().fillna(0)
            
            # Contagem de jogos (até o momento anterior)
            games_count = cumcount_shifted.fillna(0).astype(int)
            
            # Tendência (diferença entre últimas 3 e anteriores 3)
            last3 = pts.shift(1).rolling(window=3, min_periods=1).mean()
            prev3 = pts.shift(4).rolling(window=3, min_periods=1).mean()
            trend = (last3 - prev3).fillna(0)
            
            # Máximo e mínimo históricos
            max_pts = pts.shift(1).expanding().max()
            min_pts = pts.shift(1).expanding().min()
            
            # Streak de jogos positivos (pts > 0)
            # Calculado de forma vetorizada
            pts_positive = (pts.shift(1) > 0).astype(int)
            streak = pts_positive.groupby((pts_positive != pts_positive.shift()).cumsum()).cumcount() + 1
            streak = streak * pts_positive  # Zero se não positivo
            
            # DataFrame com features por temporal_id
            player_df = pd.DataFrame({
                "atleta_id": atleta_id,
                "temporal_id": group["temporal_id"].values,
                "player_mean": player_mean.values,
                "player_last5_mean": last5_mean.values,
                "player_last5_std": last5_std.values,
                "player_games": games_count.values,
                "player_trend": trend.values,
                "player_max": max_pts.values,
                "player_min": min_pts.values,
                "player_streak": streak.values,
            })
            
            features_list.append(player_df)
        
        if not features_list:
            return pd.DataFrame()
        
        result = pd.concat(features_list, ignore_index=True)
        # Indexar por (atleta_id, temporal_id) para lookup O(1)
        result = result.set_index(["atleta_id", "temporal_id"])
        
        return result
    
    def _build_scout_features_vectorized(self) -> pd.DataFrame:
        """
        Constrói DataFrame de features de scout usando operações VETORIZADAS.
        """
        scout_cols = [c for c in ALL_SCOUTS if c in self.df.columns]
        if not scout_cols:
            return pd.DataFrame()
        
        player_groups = self.df.groupby("atleta_id")
        features_list = []
        
        for atleta_id, group in player_groups:
            group = group.sort_values("temporal_id")
            
            scout_features = {"atleta_id": atleta_id, "temporal_id": group["temporal_id"].values}
            
            for scout in scout_cols:
                if scout not in group.columns:
                    continue
                    
                vals = pd.to_numeric(group[scout], errors="coerce").fillna(0).astype(float)
                
                # Média dos últimos 5 (com shift para evitar leakage)
                scout_features[f"scout_{scout}_last5_mean"] = (
                    vals.shift(1).rolling(window=5, min_periods=1).mean().values
                )
                
                # Total acumulado
                scout_features[f"scout_{scout}_total"] = vals.shift(1).cumsum().values
            
            # Scouts positivos e negativos agregados
            pos_scouts = [s for s in POSITIVE_SCOUTS if s in group.columns]
            neg_scouts = [s for s in NEGATIVE_SCOUTS if s in group.columns]
            
            if pos_scouts:
                pos_sum = group[pos_scouts].sum(axis=1).astype(float)
                scout_features["scouts_positive_last5"] = (
                    pos_sum.shift(1).rolling(window=5, min_periods=1).sum().values
                )
            
            if neg_scouts:
                neg_sum = group[neg_scouts].sum(axis=1).astype(float)
                scout_features["scouts_negative_last5"] = (
                    neg_sum.shift(1).rolling(window=5, min_periods=1).sum().values
                )
            
            player_df = pd.DataFrame(scout_features)
            features_list.append(player_df)
        
        if not features_list:
            return pd.DataFrame()
        
        result = pd.concat(features_list, ignore_index=True)
        result = result.set_index(["atleta_id", "temporal_id"])
        
        return result
    
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
        Obtém features de jogador via lookup indexado O(1).
        
        CORREÇÃO: Usa .loc com índice em vez de iteração.
        """
        # Verificar cache primeiro
        cache_key = (atleta_id, temporal_id)
        if cache_key in self._player_cache:
            return self._player_cache[cache_key]
        
        result = {}
        
        # Lookup no DataFrame indexado
        try:
            row = self._player_features_df.loc[(atleta_id, temporal_id)]
            for col in row.index:
                val = row[col]
                # CORREÇÃO: Checagem apropriada de NaN
                if pd.isna(val):
                    result[col] = 0.0
                else:
                    result[col] = float(val)
        except KeyError:
            # Jogador/temporal_id não encontrado - retornar zeros
            result = {
                "player_mean": 0.0,
                "player_last5_mean": 0.0,
                "player_last5_std": 0.0,
                "player_games": 0,
                "player_trend": 0.0,
                "player_max": 0.0,
                "player_min": 0.0,
                "player_streak": 0,
            }
        
        # Adicionar features de scout se disponível
        if self._scout_features_df is not None:
            try:
                scout_row = self._scout_features_df.loc[(atleta_id, temporal_id)]
                for col in scout_row.index:
                    val = scout_row[col]
                    if pd.isna(val):
                        result[col] = 0.0
                    else:
                        result[col] = float(val)
            except KeyError:
                pass
        
        # Cache para reutilização
        self._player_cache[cache_key] = result
        
        return result
    
    def _get_team_features(self, clube_id: int, temporal_id: int) -> Dict[str, float]:
        """Obtém features de time via lookup O(1)."""
        return self._team_aggs.get((clube_id, temporal_id), {
            "team_avg": 0.0,
            "team_std": 0.0,
            "team_last5_avg": 0.0,
        })
    
    def _get_position_features(self, posicao_id: int, temporal_id: int) -> Dict[str, float]:
        """Obtém features de posição via lookup O(1)."""
        return self._pos_aggs.get((posicao_id, temporal_id), {"pos_avg": 0.0})
    
    def _get_pos_vs_opp_features(
        self, posicao_id: int, opponent_id: int, temporal_id: int
    ) -> Dict[str, float]:
        """Obtém features de posição vs oponente via lookup O(1)."""
        return self._pos_vs_opp_aggs.get((posicao_id, opponent_id, temporal_id), {
            "pos_vs_opp_mean": 0.0,
        })
    
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
        """
        Calcula todas as features para um jogador em uma rodada.
        
        Usa lookups indexados O(1) para todas as consultas.
        """
        features = {}
        
        # 1. Features do jogador
        player_feats = self._get_player_features(atleta_id, temporal_id)
        features.update(player_feats)
        
        # 2. Features do time
        team_feats = self._get_team_features(clube_id, temporal_id)
        features.update(team_feats)
        
        # 3. Features da posição
        pos_feats = self._get_position_features(posicao_id, temporal_id)
        features.update(pos_feats)
        
        # 4. Features de posição vs oponente
        pos_vs_opp_feats = self._get_pos_vs_opp_features(posicao_id, opponent_id, temporal_id)
        features.update(pos_vs_opp_feats)
        
        # 5. Features do confronto (odds)
        match_feats = self._get_match_features(match_odds, is_home)
        features.update(match_feats)
        
        # 6. Features de identificação
        features["posicao_id"] = float(posicao_id)
        
        # Garantir que não há NaN nos features finais
        for k, v in features.items():
            # CORREÇÃO: Checagem apropriada
            if v is None or pd.isna(v):
                features[k] = 0.0
            elif np.isinf(v):
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
