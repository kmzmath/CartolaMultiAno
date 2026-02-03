"""
CARTOLA EVALUATION - Backtest e Métricas OTIMIZADO v2
=====================================================
Versão otimizada com:
- Features pré-calculadas uma vez no __init__
- Usa train_all_fast() para backtest (3 seeds, sem q10/q90)
- interval_coverage = NaN no backtest (sem conformal)
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

from .config import TOP_K_VALUES, decompose_temporal_id, BACKTEST_ENSEMBLE_SEEDS
from .io import safe_int, parse_bool, ensure_probability_simplex, OddsCache
from .models import RankingMetrics, PositionModels
from .features import TemporalFeatureEngineer

TRAIN_LOGGER: Optional[TrainingLogger] = None


class RankingBacktester:
    """
    Backtest de ranking usando temporal_id para ordenação temporal.
    
    OTIMIZAÇÃO: Features são pré-calculadas uma vez no __init__
    para evitar recálculo em cada rodada do backtest.
    """
    
    def __init__(self, df: pd.DataFrame, fe: TemporalFeatureEngineer):
        self.df = df
        self.fe = fe
        
        # PRÉ-CALCULAR FEATURES para todas as linhas válidas
        LOGGER.info("Pré-calculando features para backtest...")
        self._precomputed = self._precompute_all_features()
        LOGGER.info(f"   ✓ {len(self._precomputed)} registros com features pré-calculadas")

    def _extract_probs(self, row: pd.Series) -> Optional[Tuple[float, float, float]]:
        if "p_team_win" not in row or pd.isna(row.get("p_team_win", np.nan)):
            return None
        p_win = float(row["p_team_win"])
        p_draw = float(row.get("p_draw", 1 / 3) if not pd.isna(row.get("p_draw", np.nan)) else 1 / 3)
        p_lose = row.get("p_team_lose", None)

        if p_lose is None or (isinstance(p_lose, float) and np.isnan(p_lose)):
            p_lose = max(0.0, 1.0 - p_win - p_draw)

        p_lose = float(p_lose)

        is_home = parse_bool(row.get("is_home", False))
        if is_home:
            p_home, p_away = p_win, p_lose
        else:
            p_home, p_away = p_lose, p_win

        return ensure_probability_simplex(p_home, p_draw, p_away)

    def _precompute_all_features(self) -> Dict[Tuple[int, int], Dict]:
        """
        Pré-calcula features para TODOS os registros válidos.
        Retorna dict[( atleta_id, temporal_id)] -> {features, pontuacao, posicao_id, ...}
        """
        precomputed = {}
        
        valid = self.df[(self.df["entrou_em_campo"] == True)].copy()
        total = len(valid)
        
        for i, (_, row) in enumerate(valid.iterrows()):
            if i % 10000 == 0 and i > 0:
                LOGGER.debug(f"   Pré-calculando features: {i}/{total}")
            
            probs = self._extract_probs(row)
            if probs is None:
                continue

            p_home, p_draw, p_away = probs
            match_odds = OddsCache.get_or_create(p_home, p_draw, p_away)

            atleta_id = safe_int(row["atleta_id"])
            pos_id = safe_int(row["posicao_id"])
            clube_id = safe_int(row["clube_id"])
            opp_id = safe_int(row["opponent_id"])
            tid = safe_int(row["temporal_id"])
            
            if None in [atleta_id, pos_id, clube_id, opp_id, tid]:
                continue

            feats = self.fe.calculate_all_features(
                atleta_id, pos_id, clube_id, opp_id, match_odds, 
                parse_bool(row.get("is_home", False)), tid
            )
            
            if not feats:
                continue

            key = (atleta_id, tid)
            precomputed[key] = {
                "features": feats,
                "pontuacao": float(row["pontuacao"]),
                "posicao_id": int(pos_id),
                "clube_id": int(clube_id),
                "opponent_id": int(opp_id),
                "is_home": parse_bool(row.get("is_home", False)),
            }
        
        return precomputed

    def run(
        self,
        start_temporal_id: int,
        end_temporal_id: int,
        global_params: Optional[Dict[str, Any]] = None,
        k_pos: int = 5,
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
                
                cov_str = f"{cov:.1%}" if not np.isnan(cov) else "N/A"
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
        k_pos: int = 5,
        rank_by: str = "score",
    ) -> Optional[Dict[str, Any]]:
        """
        Backtest de uma rodada usando features pré-calculadas.
        """
        # Separar treino/teste usando features pré-calculadas
        train_data = []
        test_data = []
        
        for (atleta_id, tid), data in self._precomputed.items():
            if tid < temporal_id:
                train_data.append(data)
            elif tid == temporal_id:
                test_data.append(data)
        
        if len(train_data) < 500 or len(test_data) < 40:
            return None

        OddsCache.clear()

        # Montar dados de treino
        features_by_pos = defaultdict(lambda: ([], [], []))
        all_f, all_t, all_g = [], [], []
        
        for data in train_data:
            feats = data["features"]
            pos_id = data["posicao_id"]
            pts = data["pontuacao"]
            
            # Extrair temporal_id das features (está implícito na chave original)
            # Precisamos de uma forma de obter o tid - vamos iterar de novo
            pass
        
        # Reconstruir com tid
        for (atleta_id, tid), data in self._precomputed.items():
            if tid >= temporal_id:
                continue
                
            feats = data["features"]
            pos_id = data["posicao_id"]
            pts = data["pontuacao"]
            
            features_by_pos[pos_id][0].append(feats)
            features_by_pos[pos_id][1].append(pts)
            features_by_pos[pos_id][2].append(tid)
            
            all_f.append(feats)
            all_t.append(pts)
            all_g.append(tid)
        
        if len(all_f) < 500:
            return None

        # Treinar modelos (FAST MODE)
        models = PositionModels()
        models.train_all_fast(
            dict(features_by_pos), 
            all_f, all_t, all_g,
            global_params=global_params,
            ensemble_seeds=BACKTEST_ENSEMBLE_SEEDS,
            skip_quantiles=True,  # Sem q10/q90 no backtest
        )

        # Previsões para teste
        rows = []
        for (atleta_id, tid), data in self._precomputed.items():
            if tid != temporal_id:
                continue
                
            feats = data["features"]
            pos_id = data["posicao_id"]
            pts = data["pontuacao"]
            
            rank_sc, score_pr, p10, p90 = models.predict(pos_id, feats)

            rows.append({
                "temporal_id": int(temporal_id),
                "posicao_id": int(pos_id),
                "atleta_id": int(atleta_id),
                "rank_score": float(rank_sc[0]),
                "score_pred": float(score_pr[0]),
                "p10": float(p10[0]) if not np.isnan(p10[0]) else np.nan,
                "p90": float(p90[0]) if not np.isnan(p90[0]) else np.nan,
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

            # Métricas clássicas
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

            # Métricas práticas
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
                if np.isnan(val):
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
                if k in mp and not np.isnan(mp[k]):
                    vals.append(float(mp[k]))
            macro[k] = float(np.mean(vals)) if vals else np.nan

        return {
            "temporal_id": int(temporal_id),
            "n": int(len(dfp)),
            "n_by_pos": n_by_pos,
            "metrics_by_pos": metrics_by_pos,
            "metrics_weighted": weighted,
            "metrics_macro": macro,
        }

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}

        agg: Dict[str, Any] = {
            "n_rodadas": int(len(results)),
            "total": int(sum(r["n"] for r in results)),
            "overall_weighted": {},
            "by_pos": {},
        }

        # Agregado geral
        overall_keys = set()
        for r in results:
            overall_keys.update(r.get("metrics_weighted", {}).keys())

        for k in sorted(overall_keys):
            vals = [r["metrics_weighted"].get(k, np.nan) for r in results]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            agg["overall_weighted"][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        # Agregado por posição
        pos_ids = set()
        for r in results:
            pos_ids.update((r.get("metrics_by_pos") or {}).keys())

        for pos_id in sorted(pos_ids):
            per_round = []
            for r in results:
                mp = (r.get("metrics_by_pos") or {}).get(pos_id)
                if mp is not None:
                    per_round.append(mp)

            if not per_round:
                continue

            keys = set()
            for mp in per_round:
                keys.update(mp.keys())

            agg_pos = {}
            for k in sorted(keys):
                vals = [mp.get(k, np.nan) for mp in per_round]
                vals = [v for v in vals if not np.isnan(v)]
                if not vals:
                    continue
                agg_pos[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

            agg["by_pos"][int(pos_id)] = agg_pos

        return agg
