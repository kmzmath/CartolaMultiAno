"""
CARTOLA EVALUATION - Backtest e Métricas CORRIGIDO v3
=====================================================
Correções:
1. itertuples() em vez de iterrows() no pré-cálculo
2. Estruturas indexadas por (atleta_id, temporal_id) para lookup rápido
3. Checagem apropriada de None/NaN
4. Métricas calculadas por grupo/rodada
"""

from __future__ import annotations
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd

from training_logger import LOGGER, TrainingLogger, POSICOES

from .config import TOP_K_VALUES, decompose_temporal_id, BACKTEST_ENSEMBLE_SEEDS, DEFAULT_K
from .io import safe_int, parse_bool, ensure_probability_simplex, OddsCache
from .models import RankingMetrics, PositionModels
from .features import TemporalFeatureEngineer

TRAIN_LOGGER: Optional[TrainingLogger] = None


class RankingBacktester:
    """
    Backtest de ranking usando temporal_id para ordenação temporal.
    
    OTIMIZAÇÕES v3:
    1. Features pré-calculadas com itertuples() (não iterrows())
    2. Estrutura indexada por (atleta_id, temporal_id) para lookup O(1)
    3. Checagem apropriada de None/NaN
    """
    
    def __init__(self, df: pd.DataFrame, fe: TemporalFeatureEngineer):
        self.df = df
        self.fe = fe
        
        # PRÉ-CALCULAR FEATURES para todas as linhas válidas
        LOGGER.info("Pré-calculando features para backtest...")
        self._precomputed: Dict[Tuple[int, int], Dict] = self._precompute_all_features()
        LOGGER.info(f"   ✓ {len(self._precomputed)} registros com features pré-calculadas")
        
        # ÍNDICE AUXILIAR: temporal_id -> lista de (atleta_id, posicao_id)
        self._index_by_tid: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (atleta_id, tid), data in self._precomputed.items():
            self._index_by_tid[tid].append((atleta_id, data["posicao_id"]))

    def _extract_probs(self, row) -> Optional[Tuple[float, float, float]]:
        """
        Extrai probabilidades de uma row (pode ser namedtuple de itertuples).
        
        CORREÇÃO: Checagem apropriada de None/NaN.
        """
        # Suporta tanto Series quanto namedtuple
        if hasattr(row, '_asdict'):
            # É namedtuple de itertuples
            p_win_val = getattr(row, 'p_team_win', None)
        else:
            # É Series de iterrows
            p_win_val = row.get("p_team_win", None)
        
        # CORREÇÃO: Checagem apropriada de None/NaN
        if p_win_val is None or pd.isna(p_win_val):
            return None
        
        p_win = float(p_win_val)
        
        if hasattr(row, '_asdict'):
            p_draw_val = getattr(row, 'p_draw', None)
            p_lose_val = getattr(row, 'p_team_lose', None)
            is_home = parse_bool(getattr(row, 'is_home', False))
        else:
            p_draw_val = row.get("p_draw", None)
            p_lose_val = row.get("p_team_lose", None)
            is_home = parse_bool(row.get("is_home", False))
        
        # CORREÇÃO: Checagem apropriada
        if p_draw_val is None or pd.isna(p_draw_val):
            p_draw = 1.0 / 3.0
        else:
            p_draw = float(p_draw_val)
        
        if p_lose_val is None or pd.isna(p_lose_val):
            p_lose = max(0.0, 1.0 - p_win - p_draw)
        else:
            p_lose = float(p_lose_val)

        if is_home:
            p_home, p_away = p_win, p_lose
        else:
            p_home, p_away = p_lose, p_win

        return ensure_probability_simplex(p_home, p_draw, p_away)

    def _precompute_all_features(self) -> Dict[Tuple[int, int], Dict]:
        """
        Pré-calcula features para TODOS os registros válidos.
        
        CORREÇÃO: Usa itertuples() em vez de iterrows() (muito mais rápido).
        
        Retorna dict[(atleta_id, temporal_id)] -> {features, pontuacao, posicao_id, ...}
        """
        precomputed: Dict[Tuple[int, int], Dict] = {}
        
        valid = self.df[(self.df["entrou_em_campo"] == True)].copy()
        total = len(valid)
        
        # CORREÇÃO: itertuples() em vez de iterrows() - ~10x mais rápido
        for i, row in enumerate(valid.itertuples(index=False)):
            if i % 10000 == 0 and i > 0:
                LOGGER.debug(f"   Pré-calculando features: {i}/{total}")
            
            probs = self._extract_probs(row)
            if probs is None:
                continue

            p_home, p_draw, p_away = probs
            match_odds = OddsCache.get_or_create(p_home, p_draw, p_away)

            # Acesso via getattr para namedtuple
            atleta_id = safe_int(getattr(row, 'atleta_id', None))
            pos_id = safe_int(getattr(row, 'posicao_id', None))
            clube_id = safe_int(getattr(row, 'clube_id', None))
            opp_id = safe_int(getattr(row, 'opponent_id', None))
            tid = safe_int(getattr(row, 'temporal_id', None))
            
            # CORREÇÃO: Checagem apropriada de None
            if atleta_id is None or pos_id is None or clube_id is None or opp_id is None or tid is None:
                continue

            feats = self.fe.calculate_all_features(
                atleta_id, pos_id, clube_id, opp_id, match_odds, 
                parse_bool(getattr(row, 'is_home', False)), tid
            )
            
            if not feats:
                continue

            key = (atleta_id, tid)
            if key in precomputed:
                LOGGER.warning(f"Duplicata ignorada: atleta_id={atleta_id}, tid={tid}")
                continue
                
            precomputed[key] = {
                "features": feats,
                "pontuacao": float(getattr(row, 'pontuacao', 0.0)),
                "posicao_id": int(pos_id),
                "clube_id": int(clube_id),
                "opponent_id": int(opp_id),
                "is_home": parse_bool(getattr(row, 'is_home', False)),
            }
        
        return precomputed

    def _get_features_for_round(self, tid: int) -> List[Tuple[int, Dict]]:
        """
        Retorna features para uma rodada específica.
        
        CORREÇÃO: Usa índice pré-calculado para lookup O(1).
        """
        result = []
        for atleta_id, pos_id in self._index_by_tid.get(tid, []):
            key = (atleta_id, tid)
            data = self._precomputed.get(key)
            if data:
                result.append((atleta_id, data))
        return result

    def run(
        self,
        start_temporal_id: int,
        end_temporal_id: int,
        global_params: Optional[Dict[str, Any]] = None,
        k_pos: int = DEFAULT_K,
        rank_by: str = "score",
    ) -> Dict[str, Any]:
        """
        Executa backtest usando temporal_id.
        Usa features pré-calculadas e train_all_fast().
        """
        global TRAIN_LOGGER

        start_temp, start_round = decompose_temporal_id(start_temporal_id)
        end_temp, end_round = decompose_temporal_id(end_temporal_id)
        
        LOGGER.info("=" * 80)
        LOGGER.info(f"BACKTEST (POSICIONAL): temporal_id {start_temporal_id}-{end_temporal_id}")
        LOGGER.info(f"   ({start_temp}/R{start_round} → {end_temp}/R{end_round}) | k_pos={k_pos} | rank_by={rank_by}")
        LOGGER.info(f"   Usando {len(BACKTEST_ENSEMBLE_SEEDS)} seeds: {BACKTEST_ENSEMBLE_SEEDS}")
        LOGGER.info("=" * 80)

        results = []

        for tid in range(start_temporal_id, end_temporal_id + 1):
            round_start = time.time()
            r = self._backtest_round(
                temporal_id=tid,
                global_params=global_params,
                k_pos=k_pos,
                rank_by=rank_by
            )
            if r:
                round_duration = time.time() - round_start
                results.append(r)

                m = r["metrics_weighted"]
                k = int(k_pos)

                nd = m.get(f"ndcg@{k}", np.nan)
                hr = m.get(f"hit_rate@{k}", np.nan)
                lift = m.get(f"lift@{k}", np.nan)
                regret = m.get(f"regret@{k}", np.nan)
                pct = m.get(f"pick_percentile_mean@{k}", np.nan)
                cov = m.get("interval_coverage", np.nan)

                temp, rd = decompose_temporal_id(tid)
                
                # CORREÇÃO: Checagem apropriada de NaN
                cov_str = f"{cov:.1%}" if not pd.isna(cov) else "N/A"
                print(
                    f"   {temp}/R{rd} (tid={tid}): n={r['n']} | "
                    f"NDCG@{k}={nd:.3f} | Hit@{k}={hr:.3f} | "
                    f"Lift@{k}={lift:.2f} | Regret@{k}={regret:.2f} | "
                    f"PctMean@{k}={pct:.1f} | Cobertura={cov_str}"
                )

                if TRAIN_LOGGER:
                    TRAIN_LOGGER.log_backtest_round(
                        rodada=tid,
                        n_predictions=r["n"],
                        metrics=m,
                        duration_s=round_duration,
                        metrics_by_pos=r.get("metrics_by_pos"),
                        n_by_pos=r.get("n_by_pos"),
                        metrics_macro=r.get("metrics_macro"),
                    )

        agg = self._aggregate(results)

        if TRAIN_LOGGER:
            TRAIN_LOGGER.log_backtest_summary({
                "overall_weighted": agg.get("overall_weighted", {}),
                "by_pos": agg.get("by_pos", {}),
                "n_rodadas": agg.get("n_rodadas", 0),
                "total": agg.get("total", 0),
            })

        return agg

    def _backtest_round(
        self,
        temporal_id: int,
        global_params: Optional[Dict[str, Any]] = None,
        k_pos: int = DEFAULT_K,
        rank_by: str = "score",
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest de uma rodada usando features pré-calculadas.
        """
        # Dados para treino: todos os temporal_id < atual
        train_data = [
            (aid, data) 
            for (aid, tid), data in self._precomputed.items() 
            if tid < temporal_id
        ]
        
        if len(train_data) < 200:
            return None

        # Preparar features de treino
        features_by_pos = defaultdict(lambda: ([], [], []))
        all_features, all_targets, all_groups = [], [], []

        for atleta_id, data in train_data:
            feats = data["features"]
            pts = data["pontuacao"]
            pos_id = data["posicao_id"]
            
            # Recuperar temporal_id do dado original
            # Como train_data vem do _precomputed, precisamos do tid
            # Vamos iterar diferente
            pass

        # CORREÇÃO: Iterar corretamente mantendo o tid
        for (aid, tid), data in self._precomputed.items():
            if tid >= temporal_id:
                continue
                
            feats = data["features"]
            pts = data["pontuacao"]
            pos_id = data["posicao_id"]
            
            all_features.append(feats)
            all_targets.append(pts)
            all_groups.append(tid)
            
            pos_feats, pos_targs, pos_grps = features_by_pos[pos_id]
            pos_feats.append(feats)
            pos_targs.append(pts)
            pos_grps.append(tid)

        if len(all_features) < 200:
            return None

        # Treinar modelos (rápido)
        models = PositionModels()
        models.train_all_fast(
            features_by_pos=features_by_pos,
            all_features=all_features,
            all_targets=all_targets,
            all_groups=all_groups,
            global_params=global_params,
            skip_quantiles=True,
        )

        # Dados de teste: apenas a rodada atual
        test_data = self._get_features_for_round(temporal_id)
        
        if len(test_data) < 40:
            return None

        # Preparar predições
        rows = []
        for atleta_id, data in test_data:
            feats = data["features"]
            pos_id = data["posicao_id"]
            pts = data["pontuacao"]

            m = models.get_model(pos_id)
            Xp = m.prepare_features([feats])
            rank_score, score_pred, p10, p90 = models.predict(pos_id, Xp)

            rows.append({
                "atleta_id": atleta_id,
                "posicao_id": pos_id,
                "rank_score": float(rank_score[0]),
                "score_pred": float(score_pred[0]),
                "p10": float(p10[0]) if not pd.isna(p10[0]) else np.nan,
                "p90": float(p90[0]) if not pd.isna(p90[0]) else np.nan,
                "actual": float(pts),
            })

        if len(rows) < 40:
            return None

        dfp = pd.DataFrame(rows)
        pred_col = "score_pred" if rank_by == "score" else "rank_score"

        def _topk_idx_desc(x: np.ndarray, k: int) -> np.ndarray:
            k = min(int(k), len(x))
            if k <= 0:
                return np.array([], dtype=int)
            return np.argsort(x)[::-1][:k]

        def _percentile_0_100(values: np.ndarray, v: float) -> float:
            n = len(values)
            if n <= 1:
                return 100.0
            c = float(np.sum(values <= v))
            return 100.0 * (c - 1.0) / (n - 1.0)

        metrics_by_pos: Dict[int, Dict[str, float]] = {}
        n_by_pos: Dict[int, int] = {}

        for pos_id, g in dfp.groupby("posicao_id"):
            pred = g[pred_col].to_numpy(dtype=float)
            act = g["actual"].to_numpy(dtype=float)
            sp  = g["score_pred"].to_numpy(dtype=float)
            p10 = g["p10"].to_numpy(dtype=float)
            p90 = g["p90"].to_numpy(dtype=float)

            n = int(len(g))
            n_by_pos[int(pos_id)] = n
            if n < 2:
                continue

            k = min(int(k_pos), n)
            idx_pred = _topk_idx_desc(pred, k)
            idx_oracle = _topk_idx_desc(act, k)

            mpos = RankingMetrics.calculate_all(
                rank_pred=pred,
                actual=act,
                score_pred=sp,
                p10=p10 if not np.isnan(p10).all() else None,
                p90=p90 if not np.isnan(p90).all() else None,
                k_values=[k],
                overlap_k=k,
                compute_spearman=False,
            )

            mean_pick = float(np.mean(act[idx_pred])) if len(idx_pred) else 0.0
            mean_all = float(np.mean(act)) if len(act) else 0.0
            mean_oracle = float(np.mean(act[idx_oracle])) if len(idx_oracle) else 0.0

            lift = mean_pick - mean_all
            regret = mean_oracle - mean_pick

            pick_percentiles = [_percentile_0_100(act, float(act[i])) for i in idx_pred]
            pick_pct_mean = float(np.mean(pick_percentiles)) if pick_percentiles else 0.0
            pick_pct_min = float(np.min(pick_percentiles)) if pick_percentiles else 0.0
            pick_pct_median = float(np.median(pick_percentiles)) if pick_percentiles else 0.0

            pct_above_mean = float(np.mean(act[idx_pred] >= mean_all)) if len(idx_pred) else 0.0

            mpos.update({
                f"mean_pick@{k}": mean_pick,
                "mean_all": mean_all,
                f"mean_oracle@{k}": mean_oracle,
                f"lift@{k}": float(lift),
                f"regret@{k}": float(regret),
                f"pick_percentile_mean@{k}": pick_pct_mean,
                f"pick_percentile_median@{k}": pick_pct_median,
                f"pick_percentile_min@{k}": pick_pct_min,
                f"pct_picks_above_mean@{k}": pct_above_mean,
            })

            metrics_by_pos[int(pos_id)] = mpos

        total_n = sum(n_by_pos.values())
        if total_n <= 0 or not metrics_by_pos:
            return None

        # Agregado ponderado
        weighted: Dict[str, float] = {}
        keys = set()
        for mp in metrics_by_pos.values():
            keys.update(mp.keys())

        for k in keys:
            num = 0.0
            den = 0.0
            for pos_id, mp in metrics_by_pos.items():
                if k not in mp:
                    continue
                val = mp[k]
                # CORREÇÃO: Checagem apropriada de NaN
                if pd.isna(val):
                    continue
                w = float(n_by_pos.get(pos_id, 0))
                if w <= 0:
                    continue
                num += w * float(val)
                den += w
            weighted[k] = (num / den) if den > 0 else np.nan

        # Agregado macro
        macro: Dict[str, float] = {}
        for k in keys:
            vals = []
            for mp in metrics_by_pos.values():
                if k in mp and not pd.isna(mp[k]):
                    vals.append(float(mp[k]))
            macro[k] = float(np.mean(vals)) if vals else np.nan

        return {
            "temporal_id": int(temporal_id),
            "n": int(len(dfp)),
            "n_by_pos": n_by_pos,
            "metrics_weighted": weighted,
            "metrics_macro": macro,
            "metrics_by_pos": metrics_by_pos,
        }

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agrega resultados de múltiplas rodadas."""
        if not results:
            return {
                "overall_weighted": {},
                "overall_macro": {},
                "by_pos": {},
                "n_rodadas": 0,
                "total": 0,
            }

        # Agregado por posição
        by_pos: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        n_by_pos_total: Dict[int, int] = defaultdict(int)

        for r in results:
            for pos_id, mp in r.get("metrics_by_pos", {}).items():
                for metric, val in mp.items():
                    if not pd.isna(val):
                        by_pos[pos_id][metric].append(val)
                n_by_pos_total[pos_id] += r.get("n_by_pos", {}).get(pos_id, 0)

        # Média por posição
        by_pos_mean: Dict[int, Dict[str, float]] = {}
        for pos_id, metrics in by_pos.items():
            by_pos_mean[pos_id] = {
                metric: float(np.mean(vals)) if vals else np.nan
                for metric, vals in metrics.items()
            }
            by_pos_mean[pos_id]["n_total"] = n_by_pos_total[pos_id]

        # Agregado global (ponderado por n)
        all_keys = set()
        for mp in by_pos_mean.values():
            all_keys.update(k for k in mp.keys() if k != "n_total")

        overall_weighted: Dict[str, float] = {}
        for k in all_keys:
            num = 0.0
            den = 0.0
            for pos_id, mp in by_pos_mean.items():
                if k not in mp:
                    continue
                val = mp[k]
                if pd.isna(val):
                    continue
                w = float(n_by_pos_total.get(pos_id, 0))
                if w <= 0:
                    continue
                num += w * float(val)
                den += w
            overall_weighted[k] = (num / den) if den > 0 else np.nan

        # Macro (média simples)
        overall_macro: Dict[str, float] = {}
        for k in all_keys:
            vals = []
            for mp in by_pos_mean.values():
                if k in mp and not pd.isna(mp[k]):
                    vals.append(float(mp[k]))
            overall_macro[k] = float(np.mean(vals)) if vals else np.nan

        return {
            "overall_weighted": overall_weighted,
            "overall_macro": overall_macro,
            "by_pos": by_pos_mean,
            "n_rodadas": len(results),
            "total": sum(r.get("n", 0) for r in results),
        }
