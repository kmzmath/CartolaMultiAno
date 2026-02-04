"""
CARTOLA MODELS - Modelos de Machine Learning CORRIGIDO v3
=========================================================
Correções implementadas:
1. NDCG/Spearman calculados POR RODADA e depois agregados (média ponderada)
2. temporal_folds_by_round com unicidade garantida nos pontos de corte
3. Checagem de None/NaN usando `val is None` ou `pd.isna(val)`
4. Regularização mais forte nos defaults para combater overfitting
5. K consistente no tuning e backtest (k=5)
6. Ordenação estável (kind="mergesort")
"""

import math
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd

import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

from training_logger import LOGGER, TrainingLogger, POSICOES

from .config import (
    RANDOM_SEED, TOP_K_VALUES, CALIB_ROUNDS, CONFORMAL_ALPHA,
    HALF_LIFE_ROUNDS, OPTUNA_TRIALS, EARLY_STOPPING_ROUNDS,
    MAX_ESTIMATORS_TUNE, ENSEMBLE_SEEDS,
    # Novos limites de regularização
    DEFAULT_MAX_DEPTH, DEFAULT_NUM_LEAVES, DEFAULT_MIN_CHILD_SAMPLES,
    OPTUNA_MAX_DEPTH_RANGE, OPTUNA_NUM_LEAVES_RANGE, OPTUNA_MIN_CHILD_SAMPLES_RANGE,
)

try:
    from .config import BACKTEST_ENSEMBLE_SEEDS
except ImportError:
    BACKTEST_ENSEMBLE_SEEDS = [42, 72, 102]

TRAIN_LOGGER: Optional[TrainingLogger] = None


# =============================================================================
# UTIL - COM ORDENAÇÃO ESTÁVEL E UNICIDADE
# =============================================================================

def split_train_cal_by_round(groups: np.ndarray, calib_rounds: int = CALIB_ROUNDS):
    """Split treino/calibração por rodadas finais."""
    groups = np.asarray(groups, dtype=int)
    rounds = sorted(np.unique(groups).tolist())
    if len(rounds) <= calib_rounds + 5:
        train_mask = np.ones(len(groups), dtype=bool)
        cal_mask = np.zeros(len(groups), dtype=bool)
        return train_mask, cal_mask, []
    cal_rounds = rounds[-calib_rounds:]
    cal_mask = np.isin(groups, cal_rounds)
    train_mask = ~cal_mask
    return train_mask, cal_mask, cal_rounds


def _quantile_higher(x: np.ndarray, q: float) -> float:
    """Quantil usando método 'higher'."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def temporal_folds_by_round(
    rounds_sorted: List[int],
    n_splits: int = 5,
    min_train_rounds: int = 8,
    val_rounds: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """
    Gera folds temporais GARANTINDO UNICIDADE dos pontos de corte.
    
    CORREÇÃO: Usa np.unique para garantir que não haja pontos de corte duplicados.
    """
    rounds_sorted = sorted(rounds_sorted)
    n_rounds = len(rounds_sorted)
    
    if n_rounds < (min_train_rounds + val_rounds):
        return []
    
    max_start = n_rounds - val_rounds
    
    # Gerar pontos de corte candidatos
    cut_candidates = np.linspace(min_train_rounds, max_start, num=n_splits, dtype=int)
    
    # CORREÇÃO: Garantir unicidade
    cut_points = []
    seen = set()
    for cut in cut_candidates:
        cut = int(cut)
        if cut not in seen and cut >= min_train_rounds and cut <= max_start:
            seen.add(cut)
            cut_points.append(cut)
    
    # Se temos menos folds que desejado, tentar preencher com outros pontos
    if len(cut_points) < n_splits:
        for candidate in range(min_train_rounds, max_start + 1):
            if candidate not in seen and len(cut_points) < n_splits:
                cut_points.append(candidate)
                seen.add(candidate)
        cut_points = sorted(cut_points)
    
    folds = []
    for cut in cut_points:
        train_r = rounds_sorted[:cut]
        val_r = rounds_sorted[cut:cut + val_rounds]
        if len(val_r) >= val_rounds:
            folds.append((train_r, val_r))
    
    return folds


def sort_by_group(X, y: np.ndarray, groups: np.ndarray):
    """
    Ordena X, y, groups por grupo de forma ESTÁVEL.
    IMPORTANTE: Usa mergesort para garantir ordem determinística em empates.
    """
    order = np.argsort(groups, kind="mergesort")  # ESTÁVEL!

    if hasattr(X, "iloc"):
        Xs = X.iloc[order].reset_index(drop=True)
    else:
        Xs = X[order]

    ys = y[order]
    gs = groups[order]
    _, counts = np.unique(gs, return_counts=True)
    group_sizes = counts.tolist()
    return Xs, ys, gs, group_sizes


def make_relevance_labels_by_group(y: np.ndarray, groups: np.ndarray, max_rel: int = 30) -> np.ndarray:
    """Cria labels de relevância para LambdaRank."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups, dtype=int)
    rel = np.zeros(len(y), dtype=int)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            continue
        vals = y[idx]

        order = np.argsort(vals, kind="mergesort")  # ESTÁVEL!
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))

        if len(idx) == 1:
            rel[idx] = 0
        else:
            rel[idx] = np.floor(ranks * max_rel / (len(idx) - 1)).astype(int)

    return rel


def make_time_decay_weights(groups: np.ndarray, half_life_rounds: float = 6.0) -> np.ndarray:
    """Cria pesos de decay temporal."""
    g = np.asarray(groups, dtype=float)
    gmax = np.max(g)
    lam = np.log(2) / half_life_rounds
    w = np.exp(-lam * (gmax - g))
    return w.astype(float)


def mean_lift_at_k_temporal(
    pred: np.ndarray,
    actual: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
    min_group_size: Optional[int] = None,
) -> float:
    """Lift médio calculado por grupo e depois agregado."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    groups = np.asarray(groups, dtype=int)

    lifts: List[float] = []
    for g in np.unique(groups):
        m = (groups == g)
        n = int(np.sum(m))
        if n <= 1:
            continue
        if min_group_size is not None and n < int(min_group_size):
            continue

        k_eff = min(int(k), n)
        if k_eff <= 0:
            continue

        p = pred[m]
        a = actual[m]

        top_idx = np.argsort(p)[::-1][:k_eff]
        mean_pick = float(np.mean(a[top_idx])) if k_eff > 0 else 0.0
        mean_all = float(np.mean(a)) if n > 0 else 0.0
        lifts.append(mean_pick - mean_all)

    return float(np.mean(lifts)) if lifts else 0.0


# =============================================================================
# MÉTRICAS - CORRIGIDAS PARA AGREGAÇÃO POR GRUPO
# =============================================================================

class RankingMetrics:
    """
    Métricas de ranking com suporte a agregação por grupo (query/rodada).
    
    CORREÇÃO: Novos métodos que calculam métricas POR RODADA e agregam.
    """
    
    @staticmethod
    def spearman(pred: np.ndarray, actual: np.ndarray) -> float:
        """Spearman correlation."""
        pred = np.asarray(pred, dtype=float)
        actual = np.asarray(actual, dtype=float)
        if pred.size < 3:
            return 0.0
        if np.nanmax(pred) == np.nanmin(pred) or np.nanmax(actual) == np.nanmin(actual):
            return 0.0
        corr, _ = spearmanr(pred, actual)
        # CORREÇÃO: Checagem apropriada de NaN
        if corr is None or pd.isna(corr):
            return 0.0
        return float(corr)

    @staticmethod
    def hit_rate_at_k(pred: np.ndarray, actual: np.ndarray, k: int) -> float:
        """Hit rate (proporção de acertos no top-k)."""
        if len(pred) == 0:
            return 0.0
        k = min(int(k), len(pred))
        if k <= 0:
            return 0.0
        top_pred = set(np.argsort(pred)[-k:])
        top_act = set(np.argsort(actual)[-k:])
        return len(top_pred & top_act) / k

    @staticmethod
    def ndcg_at_k(pred: np.ndarray, actual: np.ndarray, k: int) -> float:
        """NDCG@k para um único grupo."""
        if len(pred) < 2:
            return 0.0
        k = min(int(k), len(pred))
        if k <= 0:
            return 0.0

        pred_order = np.argsort(pred)[::-1][:k]
        a = actual.astype(float)
        a_min, a_max = a.min(), a.max()
        if a_max - a_min < 1e-10:
            return 1.0  # Todos iguais = ranking perfeito
        a_norm = (a - a_min) / (a_max - a_min)

        dcg = 0.0
        for i, idx in enumerate(pred_order):
            rel = a_norm[idx]
            dcg += (2 ** rel - 1) / np.log2(i + 2)

        ideal_order = np.argsort(a)[::-1][:k]
        idcg = 0.0
        for i, idx in enumerate(ideal_order):
            rel = a_norm[idx]
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @classmethod
    def ndcg_at_k_by_group(
        cls,
        pred: np.ndarray,
        actual: np.ndarray,
        groups: np.ndarray,
        k: int,
        min_group_size: int = 2,
        weighted: bool = True,
    ) -> float:
        """
        NOVO: NDCG@k calculado POR GRUPO e depois agregado.
        
        Args:
            pred: Predições
            actual: Valores reais
            groups: IDs dos grupos (rodadas)
            k: Top-k para NDCG
            min_group_size: Tamanho mínimo do grupo para incluir
            weighted: Se True, pondera por tamanho do grupo
        
        Returns:
            NDCG médio (ponderado ou não) agregado por grupo
        """
        pred = np.asarray(pred, dtype=float)
        actual = np.asarray(actual, dtype=float)
        groups = np.asarray(groups, dtype=int)
        
        ndcgs = []
        weights = []
        
        for g in np.unique(groups):
            mask = (groups == g)
            n = int(np.sum(mask))
            if n < min_group_size:
                continue
            
            k_eff = min(k, n)
            ndcg_g = cls.ndcg_at_k(pred[mask], actual[mask], k_eff)
            ndcgs.append(ndcg_g)
            weights.append(n)
        
        if not ndcgs:
            return 0.0
        
        if weighted:
            total_w = sum(weights)
            if total_w == 0:
                return 0.0
            return sum(n * w for n, w in zip(ndcgs, weights)) / total_w
        else:
            return float(np.mean(ndcgs))

    @classmethod
    def spearman_by_group(
        cls,
        pred: np.ndarray,
        actual: np.ndarray,
        groups: np.ndarray,
        min_group_size: int = 3,
        weighted: bool = True,
    ) -> float:
        """
        NOVO: Spearman calculado POR GRUPO e depois agregado.
        """
        pred = np.asarray(pred, dtype=float)
        actual = np.asarray(actual, dtype=float)
        groups = np.asarray(groups, dtype=int)
        
        corrs = []
        weights = []
        
        for g in np.unique(groups):
            mask = (groups == g)
            n = int(np.sum(mask))
            if n < min_group_size:
                continue
            
            corr = cls.spearman(pred[mask], actual[mask])
            # CORREÇÃO: Checagem apropriada
            if corr is not None and not pd.isna(corr):
                corrs.append(corr)
                weights.append(n)
        
        if not corrs:
            return 0.0
        
        if weighted:
            total_w = sum(weights)
            if total_w == 0:
                return 0.0
            return sum(c * w for c, w in zip(corrs, weights)) / total_w
        else:
            return float(np.mean(corrs))

    @staticmethod
    def topk_overlap(pred: np.ndarray, actual: np.ndarray, k: int) -> Tuple[int, float]:
        """Overlap do top-k predito vs real."""
        k = min(int(k), len(pred))
        if k <= 0:
            return (0, 0.0)
        top_pred = set(np.argsort(pred)[-k:])
        top_act = set(np.argsort(actual)[-k:])
        overlap = len(top_pred & top_act)
        return overlap, overlap / k

    @staticmethod
    def interval_coverage(actual: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
        """Cobertura do intervalo de confiança."""
        if len(actual) == 0:
            return 0.0
        inside = np.sum((actual >= p10) & (actual <= p90))
        return inside / len(actual)

    @classmethod
    def calculate_all(
        cls,
        rank_pred: np.ndarray,
        actual: np.ndarray,
        score_pred: Optional[np.ndarray] = None,
        p10: Optional[np.ndarray] = None,
        p90: Optional[np.ndarray] = None,
        k_values: Optional[List[int]] = None,
        overlap_k: int = 11,
        compute_spearman: bool = False,
    ) -> Dict[str, float]:
        """Calcula todas as métricas para um único grupo."""
        if k_values is None:
            k_values = TOP_K_VALUES

        m: Dict[str, float] = {}

        if compute_spearman:
            m["spearman"] = cls.spearman(rank_pred, actual)

        for k in k_values:
            k = int(k)
            if k <= len(rank_pred) and k > 0:
                m[f"hit_rate@{k}"] = cls.hit_rate_at_k(rank_pred, actual, k)
                m[f"ndcg@{k}"] = cls.ndcg_at_k(rank_pred, actual, k)

        ov, pct = cls.topk_overlap(rank_pred, actual, overlap_k)
        m[f"top{overlap_k}_overlap"] = float(ov)
        m[f"top{overlap_k}_pct"] = float(pct)

        # CORREÇÃO: Checagem apropriada de None
        if score_pred is not None:
            m["mae"] = float(mean_absolute_error(actual, score_pred))
            m["rmse"] = float(math.sqrt(mean_squared_error(actual, score_pred)))

        if p10 is not None and p90 is not None:
            m["interval_coverage"] = float(cls.interval_coverage(actual, p10, p90))

        return m

    @classmethod
    def calculate_all_by_group(
        cls,
        rank_pred: np.ndarray,
        actual: np.ndarray,
        groups: np.ndarray,
        score_pred: Optional[np.ndarray] = None,
        k_values: Optional[List[int]] = None,
        weighted: bool = True,
    ) -> Dict[str, float]:
        """
        NOVO: Calcula métricas por grupo e agrega.
        
        Uso correto em LTR: cada grupo é uma query/rodada.
        """
        if k_values is None:
            k_values = TOP_K_VALUES
        
        m: Dict[str, float] = {}
        
        for k in k_values:
            m[f"ndcg@{k}"] = cls.ndcg_at_k_by_group(
                rank_pred, actual, groups, k, 
                min_group_size=max(2, k), weighted=weighted
            )
        
        m["spearman"] = cls.spearman_by_group(
            rank_pred, actual, groups, 
            min_group_size=3, weighted=weighted
        )
        
        # MAE/RMSE não precisam de agregação por grupo (são métricas pontuais)
        if score_pred is not None:
            m["mae"] = float(mean_absolute_error(actual, score_pred))
            m["rmse"] = float(math.sqrt(mean_squared_error(actual, score_pred)))
        
        return m


# =============================================================================
# MODELO PRINCIPAL
# =============================================================================

class RankingModel:
    def __init__(self, posicao_id: Optional[int] = None):
        self.posicao_id = posicao_id
        self.feature_columns: List[str] = []

        self.best_params_ranker: Optional[Dict[str, Any]] = None
        self.best_params_regressor: Optional[Dict[str, Any]] = None
        self.best_params: Optional[Dict[str, Any]] = None

        self.ranker: Optional[lgb.LGBMRanker] = None
        self.regressor: Optional[lgb.LGBMRegressor] = None
        self.q10: Optional[lgb.LGBMRegressor] = None
        self.q90: Optional[lgb.LGBMRegressor] = None

    def _infer_feature_columns(self, sample: Dict[str, float]) -> List[str]:
        cols = []
        for k, v in sample.items():
            # CORREÇÃO: Checagem apropriada de None/NaN
            if v is None:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                if not pd.isna(v) and not np.isinf(v):
                    cols.append(k)
        return sorted(cols)

    def prepare_features(self, features_list: List[Dict[str, float]]) -> pd.DataFrame:
        if not self.feature_columns:
            self.feature_columns = self._infer_feature_columns(features_list[0])

        df = pd.DataFrame(features_list)
        for c in self.feature_columns:
            if c not in df.columns:
                df[c] = 0.0
        df = df[self.feature_columns].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return df

    def _default_params(self) -> Dict[str, Any]:
        """
        Parâmetros default com REGULARIZAÇÃO MAIS FORTE.
        
        CORREÇÃO: 
        - max_depth reduzido (4 vs 8)
        - num_leaves reduzido (31 vs 63)
        - min_child_samples aumentado (50 vs 30)
        - reg_alpha e reg_lambda aumentados
        """
        return {
            "n_estimators": 900,
            "learning_rate": 0.03,
            "max_depth": DEFAULT_MAX_DEPTH,  # 4 (era 8)
            "num_leaves": DEFAULT_NUM_LEAVES,  # 31 (era 63)
            "min_child_samples": DEFAULT_MIN_CHILD_SAMPLES,  # 50 (era 30)
            "subsample": 0.7,  # 0.7 (era 0.8)
            "colsample_bytree": 0.7,  # 0.7 (era 0.8)
            "reg_alpha": 1.0,  # 1.0 (era 0.1)
            "reg_lambda": 5.0,  # 5.0 (era 1.0)
            "min_sum_hessian_in_leaf": 1.0,  # NOVO
            "max_bin": 255,
            "path_smooth": 1.0,  # 1.0 (era 0.0)
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
            "importance_type": "gain",
            "force_col_wise": True,
        }

    def optimize(
        self, 
        X, 
        y: np.ndarray, 
        groups: np.ndarray, 
        n_trials: int = OPTUNA_TRIALS,
        k: int = 5,  # CORREÇÃO: K consistente com backtest
    ) -> float:
        """
        Otimização Optuna com NDCG calculado POR RODADA.
        
        CORREÇÕES:
        1. NDCG calculado por rodada, não misturando dados de rodadas diferentes
        2. Ranges de hiperparâmetros mais conservadores
        3. K consistente com backtest (default k=5)
        """
        global TRAIN_LOGGER
        
        optuna_start = time.time()
        
        if hasattr(X, "to_numpy"):
            Xn = X.to_numpy()
        else:
            Xn = np.asarray(X)

        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups, dtype=int)

        rounds_sorted = sorted(np.unique(groups).tolist())
        folds = temporal_folds_by_round(rounds_sorted, n_splits=5, min_train_rounds=8, val_rounds=5)
        if not folds:
            return 0.0

        y_rank_all = make_relevance_labels_by_group(y, groups)
        w_all = make_time_decay_weights(groups, half_life_rounds=HALF_LIFE_ROUNDS)

        pos_name = POSICOES.get(self.posicao_id, "Global") if self.posicao_id else "Global"
        LOGGER.info(
            f"OPTUNA START | {pos_name} | samples={len(y)} | rounds={len(rounds_sorted)} | "
            f"folds={len(folds)} | trials={n_trials} | k={k}"
        )

        cb = None
        if hasattr(lgb, "early_stopping"):
            cb = lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)

        def objective(trial):
            trial_start = time.time()
            
            # CORREÇÃO: Ranges mais conservadores para evitar overfitting
            params = {
                "n_estimators": MAX_ESTIMATORS_TUNE,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),  # Max 0.05 (era 0.08)
                "max_depth": trial.suggest_int("max_depth", *OPTUNA_MAX_DEPTH_RANGE),  # 3-6 (era 3-12)
                "num_leaves": trial.suggest_int("num_leaves", *OPTUNA_NUM_LEAVES_RANGE),  # 16-63 (era 16-255)
                "min_child_samples": trial.suggest_int("min_child_samples", *OPTUNA_MIN_CHILD_SAMPLES_RANGE),  # 30-200 (era 10-200)
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.8),  # Max 0.8 (era 1.0)
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),  # Max 0.8 (era 1.0)
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),  # Min 0.1 (era 1e-8)
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),  # Min 1.0 (era 1e-8)
                "max_bin": trial.suggest_int("max_bin", 127, 255),  # Max 255 (era 511)
                "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),  # Max 10 (era 50)
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "verbose": -1,
                "importance_type": "gain",
                "force_col_wise": True,
            }

            fold_scores = []
            for fi, (train_rounds, val_rounds) in enumerate(folds):
                train_mask = np.isin(groups, train_rounds)
                val_mask = np.isin(groups, val_rounds)

                if train_mask.sum() < 200 or val_mask.sum() < 30:
                    continue

                Xtr = Xn[train_mask]
                gtr = groups[train_mask]
                ytr_rank = y_rank_all[train_mask]
                wtr = w_all[train_mask]

                Xva = Xn[val_mask]
                gva = groups[val_mask]
                yva = y[val_mask]
                yva_rank = y_rank_all[val_mask]

                # Ordenação estável para treino
                order_tr = np.argsort(gtr, kind="mergesort")
                Xtr_s = Xtr[order_tr]
                ytr_s = ytr_rank[order_tr]
                gtr_s = gtr[order_tr]
                wtr_s = wtr[order_tr]

                _, counts_tr = np.unique(gtr_s, return_counts=True)
                group_sizes_tr = counts_tr.tolist()

                # Ordenação estável para validação
                order_va = np.argsort(gva, kind="mergesort")
                Xva_s = Xva[order_va]
                yva_s_rank = yva_rank[order_va]
                gva_s = gva[order_va]

                _, counts_va = np.unique(gva_s, return_counts=True)
                group_sizes_va = counts_va.tolist()

                ranker = lgb.LGBMRanker(**params, objective="lambdarank")
                try:
                    fit_kwargs = dict(
                        X=Xtr_s,
                        y=ytr_s,
                        group=group_sizes_tr,
                        sample_weight=wtr_s,
                        eval_set=[(Xva_s, yva_s_rank)],
                        eval_group=[group_sizes_va],
                        eval_at=[k],  # CORREÇÃO: Usar k consistente
                        eval_metric="ndcg",
                    )
                    if cb is not None:
                        fit_kwargs["callbacks"] = [cb]
                    ranker.fit(**fit_kwargs)
                except Exception:
                    ranker.fit(Xtr_s, ytr_s, group=group_sizes_tr, sample_weight=wtr_s)

                pred_va = ranker.predict(Xva)
                
                # CORREÇÃO: Calcular NDCG POR RODADA e agregar
                ndcg_fold = RankingMetrics.ndcg_at_k_by_group(
                    pred_va, yva, gva, k=k, min_group_size=max(2, k), weighted=True
                )
                fold_scores.append(ndcg_fold)

            if not fold_scores:
                return 0.0
            
            mean_score = float(np.mean(fold_scores))
            trial_duration = time.time() - trial_start
            
            if TRAIN_LOGGER:
                log_params = {kk: vv for kk, vv in params.items() 
                             if kk not in ['n_jobs', 'verbose', 'force_col_wise', 'importance_type', 'n_estimators', 'random_state']}
                TRAIN_LOGGER.log_optuna_trial(
                    trial_number=trial.number,
                    params=log_params,
                    score=mean_score,
                    fold_scores=fold_scores,
                    posicao_id=self.posicao_id or -1,
                    duration_s=trial_duration
                )
            
            return -mean_score

        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=30)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = dict(study.best_params)
        self.best_params.update({
            "n_estimators": MAX_ESTIMATORS_TUNE,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
            "importance_type": "gain",
            "force_col_wise": True,
        })
        
        total_time = time.time() - optuna_start
        
        if TRAIN_LOGGER:
            TRAIN_LOGGER.log_optuna_summary(
                best_trial=study.best_trial.number,
                best_score=-study.best_value,
                best_params=self.best_params,
                total_trials=n_trials,
                total_time_s=total_time,
                posicao_id=self.posicao_id or -1
            )
        
        return -float(study.best_value)

    def optimize_regressor_lift_at_k(
        self,
        X,
        y: np.ndarray,
        groups: np.ndarray,
        k: int = 5,
        n_trials: int = OPTUNA_TRIALS,
    ) -> float:
        """Optuna para REGRESSOR com objetivo = lift@k."""
        global TRAIN_LOGGER

        optuna_start = time.time()

        if hasattr(X, "to_numpy"):
            Xn = X.to_numpy()
        else:
            Xn = np.asarray(X)

        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups, dtype=int)

        rounds_sorted = sorted(np.unique(groups).tolist())
        folds = temporal_folds_by_round(rounds_sorted, n_splits=5, min_train_rounds=8, val_rounds=5)
        if not folds:
            return 0.0

        w_all = make_time_decay_weights(groups, half_life_rounds=HALF_LIFE_ROUNDS)

        pos_name = POSICOES.get(self.posicao_id, "Global") if self.posicao_id else "Global"
        LOGGER.info(
            f"OPTUNA REGRESSOR(LIFT@{k}) START | {pos_name} | samples={len(y)} | "
            f"rounds={len(rounds_sorted)} | folds={len(folds)} | trials={n_trials}"
        )

        cb = None
        if hasattr(lgb, "early_stopping"):
            cb = lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)

        def objective(trial):
            trial_start = time.time()

            # CORREÇÃO: Ranges mais conservadores
            params = {
                "n_estimators": MAX_ESTIMATORS_TUNE,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "max_depth": trial.suggest_int("max_depth", *OPTUNA_MAX_DEPTH_RANGE),
                "num_leaves": trial.suggest_int("num_leaves", *OPTUNA_NUM_LEAVES_RANGE),
                "min_child_samples": trial.suggest_int("min_child_samples", *OPTUNA_MIN_CHILD_SAMPLES_RANGE),
                "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
                "max_bin": trial.suggest_int("max_bin", 127, 255),
                "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "verbose": -1,
                "importance_type": "gain",
                "force_col_wise": True,
                "objective": "regression",
            }

            fold_scores: List[float] = []

            for fi, (train_rounds, val_rounds) in enumerate(folds):
                train_mask = np.isin(groups, train_rounds)
                val_mask = np.isin(groups, val_rounds)

                if train_mask.sum() < 200 or val_mask.sum() < 30:
                    continue

                Xtr = Xn[train_mask]
                ytr = y[train_mask]
                wtr = w_all[train_mask]

                Xva = Xn[val_mask]
                yva = y[val_mask]
                gva = groups[val_mask]

                model = lgb.LGBMRegressor(**params)

                try:
                    fit_kwargs = dict(
                        X=Xtr,
                        y=ytr,
                        sample_weight=wtr,
                        eval_set=[(Xva, yva)],
                        eval_metric="l2",
                    )
                    if cb is not None:
                        fit_kwargs["callbacks"] = [cb]
                    model.fit(**fit_kwargs)
                except Exception:
                    model.fit(Xtr, ytr, sample_weight=wtr)

                pred_va = model.predict(Xva)

                lift = mean_lift_at_k_temporal(
                    pred=pred_va,
                    actual=yva,
                    groups=gva,
                    k=k,
                    min_group_size=k,
                )
                fold_scores.append(float(lift))

                interim = -float(np.mean(fold_scores))
                trial.report(interim, step=fi)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not fold_scores:
                return 0.0

            mean_lift = float(np.mean(fold_scores))
            trial_duration = time.time() - trial_start

            if TRAIN_LOGGER:
                log_params = {kk: vv for kk, vv in params.items()
                            if kk not in ["n_jobs","verbose","force_col_wise","importance_type","n_estimators","random_state","objective"]}
                TRAIN_LOGGER.log_optuna_trial(
                    trial_number=trial.number,
                    params=log_params,
                    score=mean_lift,
                    fold_scores=fold_scores,
                    posicao_id=self.posicao_id or -1,
                    duration_s=trial_duration
                )

            return -mean_lift

        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=30)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_lift = -float(study.best_value)

        self.best_params_regressor = dict(study.best_params)
        self.best_params_regressor.update({
            "n_estimators": MAX_ESTIMATORS_TUNE,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
            "importance_type": "gain",
            "force_col_wise": True,
            "objective": "regression",
        })

        total_time = time.time() - optuna_start

        if TRAIN_LOGGER:
            TRAIN_LOGGER.log_optuna_summary(
                best_trial=study.best_trial.number,
                best_score=best_lift,
                best_params=self.best_params_regressor,
                total_trials=n_trials,
                total_time_s=total_time,
                posicao_id=self.posicao_id or -1
            )

        LOGGER.info(f"OPTUNA REGRESSOR(LIFT@{k}) END | {pos_name} | best_lift={best_lift:.4f}")
        return best_lift

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        optimize: bool = True,
        optimize_regressor: bool = False,
        lift_k: int = 5,
        seed: int = RANDOM_SEED,
        skip_quantiles: bool = False,
    ):
        """Treina ranker + regressors."""
        global TRAIN_LOGGER

        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups, dtype=int)

        if optimize_regressor:
            best_lift = self.optimize_regressor_lift_at_k(X, y, groups, k=lift_k, n_trials=OPTUNA_TRIALS)
            print(f"      ✓ Melhor Lift@{lift_k} (CV temporal): {best_lift:.4f}")

        if optimize:
            best = self.optimize(X, y, groups, n_trials=OPTUNA_TRIALS, k=lift_k)  # CORREÇÃO: k consistente
            print(f"      ✓ Melhor NDCG@{lift_k} (CV temporal): {best:.4f}")

        if not self.best_params_ranker:
            if self.best_params:
                self.best_params_ranker = dict(self.best_params)
            else:
                self.best_params_ranker = self._default_params()

        if not self.best_params_regressor:
            if self.best_params:
                self.best_params_regressor = dict(self.best_params)
            else:
                self.best_params_regressor = self._default_params()

        self.best_params_ranker = dict(self.best_params_ranker)
        self.best_params_regressor = dict(self.best_params_regressor)

        self.best_params_ranker.pop("objective", None)
        self.best_params_ranker.pop("alpha", None)

        w = make_time_decay_weights(groups, half_life_rounds=HALF_LIFE_ROUNDS)

        uniq = sorted(np.unique(groups).tolist())
        val_rounds = uniq[-5:] if len(uniq) >= 10 else []
        val_mask = np.isin(groups, val_rounds) if val_rounds else np.zeros(len(groups), dtype=bool)
        tr_mask = ~val_mask

        cb = None
        if hasattr(lgb, "early_stopping"):
            cb = lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)

        y_rank = make_relevance_labels_by_group(y, groups)

        # ==== TREINO DO RANKER ====
        ranker_start = time.time()

        def fit_ranker(params_fit):
            Xtr = X[tr_mask] if hasattr(X, "iloc") else X[tr_mask]
            gtr = groups[tr_mask]
            ytr = y_rank[tr_mask]
            wtr = w[tr_mask]

            order_tr = np.argsort(gtr, kind="mergesort")
            Xtr_s = Xtr.iloc[order_tr].reset_index(drop=True) if hasattr(Xtr, "iloc") else Xtr[order_tr]
            ytr_s = ytr[order_tr]
            gtr_s = gtr[order_tr]
            wtr_s = wtr[order_tr]

            _, counts_tr = np.unique(gtr_s, return_counts=True)
            group_sizes_tr = counts_tr.tolist()

            ranker = lgb.LGBMRanker(**params_fit, objective="lambdarank")

            if val_mask.sum() >= 80:
                Xva = X[val_mask] if hasattr(X, "iloc") else X[val_mask]
                gva = groups[val_mask]
                yva_rank = y_rank[val_mask]

                order_va = np.argsort(gva, kind="mergesort")
                Xva_s = Xva.iloc[order_va].reset_index(drop=True) if hasattr(Xva, "iloc") else Xva[order_va]
                yva_s = yva_rank[order_va]
                gva_s = gva[order_va]

                _, counts_va = np.unique(gva_s, return_counts=True)
                group_sizes_va = counts_va.tolist()

                try:
                    fit_kwargs = dict(
                        X=Xtr_s,
                        y=ytr_s,
                        group=group_sizes_tr,
                        sample_weight=wtr_s,
                        eval_set=[(Xva_s, yva_s)],
                        eval_group=[group_sizes_va],
                        eval_at=[lift_k],  # CORREÇÃO: k consistente
                        eval_metric="ndcg",
                    )
                    if cb is not None:
                        fit_kwargs["callbacks"] = [cb]
                    ranker.fit(**fit_kwargs)
                except Exception:
                    ranker.fit(Xtr_s, ytr_s, group=group_sizes_tr, sample_weight=wtr_s)
            else:
                ranker.fit(Xtr_s, ytr_s, group=group_sizes_tr, sample_weight=wtr_s)

            best_it = getattr(ranker, "best_iteration_", None)
            return ranker, best_it

        params_rank = dict(self.best_params_ranker)
        params_rank["random_state"] = seed

        ranker_tmp, best_it = fit_ranker(params_rank)

        params_rank2 = dict(params_rank)
        # CORREÇÃO: Checagem apropriada de None
        if best_it is not None and isinstance(best_it, (int, np.integer)) and int(best_it) > 10:
            params_rank2["n_estimators"] = int(best_it)
        else:
            if params_rank2.get("n_estimators", 0) > 1500 and val_mask.sum() == 0:
                params_rank2["n_estimators"] = 900

        # Treino final no dataset completo
        Xs, y_rank_s, gs, group_sizes = sort_by_group(X, y_rank, groups)
        order_full = np.argsort(groups, kind="mergesort")
        w_s = w[order_full]

        self.ranker = lgb.LGBMRanker(**params_rank2, objective="lambdarank")
        self.ranker.fit(Xs, y_rank_s, group=group_sizes, sample_weight=w_s)

        ranker_duration = time.time() - ranker_start

        ranker_val_metrics = {}
        if val_mask.sum() >= 40 and self.ranker is not None:
            Xva_raw = X[val_mask] if hasattr(X, "iloc") else X[val_mask]
            pred_val = self.ranker.predict(Xva_raw)
            actual_val = y[val_mask]
            groups_val = groups[val_mask]
            
            # CORREÇÃO: Métricas calculadas por grupo
            ranker_val_metrics = {
                f"ndcg@{lift_k}": RankingMetrics.ndcg_at_k_by_group(
                    pred_val, actual_val, groups_val, k=lift_k, weighted=True
                ),
                "spearman": RankingMetrics.spearman_by_group(
                    pred_val, actual_val, groups_val, weighted=True
                ),
            }

        if TRAIN_LOGGER:
            TRAIN_LOGGER.log_model_training(
                model_type="Ranker",
                posicao_id=self.posicao_id or -1,
                seed=seed,
                train_samples=int(tr_mask.sum()),
                val_samples=int(val_mask.sum()),
                best_iteration=best_it,
                train_metrics={},
                val_metrics=ranker_val_metrics,
                duration_s=ranker_duration
            )

        # ==== TREINO DOS REGRESSORES ====
        def fit_regressor_with_log(obj: Optional[str] = None, alpha: Optional[float] = None, model_name: str = "Regressor"):
            reg_start = time.time()

            params = dict(self.best_params_regressor)
            params["random_state"] = seed
            params.setdefault("objective", "regression")

            if obj is not None:
                params["objective"] = obj
            else:
                params["objective"] = params.get("objective", "regression")

            # CORREÇÃO: Checagem apropriada de None
            if alpha is not None:
                params["alpha"] = float(alpha)
            else:
                params.pop("alpha", None)

            model = lgb.LGBMRegressor(**params)

            best_iter = None
            if val_mask.sum() >= 80:
                Xtr = X[tr_mask] if hasattr(X, "iloc") else X[tr_mask]
                ytr = y[tr_mask]
                wtr = w[tr_mask]

                Xva = X[val_mask] if hasattr(X, "iloc") else X[val_mask]
                yva = y[val_mask]

                try:
                    fit_kwargs = dict(
                        X=Xtr,
                        y=ytr,
                        sample_weight=wtr,
                        eval_set=[(Xva, yva)],
                    )
                    if cb is not None:
                        fit_kwargs["callbacks"] = [cb]
                    model.fit(**fit_kwargs)
                except Exception:
                    model.fit(Xtr, ytr, sample_weight=wtr)

                best_iter = getattr(model, "best_iteration_", None)
                # CORREÇÃO: Checagem apropriada
                if best_iter is not None and isinstance(best_iter, (int, np.integer)) and int(best_iter) > 10:
                    params2 = dict(params)
                    params2["n_estimators"] = int(best_iter)
                    model2 = lgb.LGBMRegressor(**params2)
                    model2.fit(X, y, sample_weight=w)
                    model = model2
            else:
                model.fit(X, y, sample_weight=w)

            reg_duration = time.time() - reg_start

            val_metrics = {}
            if val_mask.sum() >= 40:
                Xva_raw = X[val_mask] if hasattr(X, "iloc") else X[val_mask]
                pred = model.predict(Xva_raw)
                val_metrics = {
                    "mae": float(mean_absolute_error(y[val_mask], pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y[val_mask], pred))),
                }

            if TRAIN_LOGGER:
                TRAIN_LOGGER.log_model_training(
                    model_type=model_name,
                    posicao_id=self.posicao_id or -1,
                    seed=seed,
                    train_samples=int(tr_mask.sum()),
                    val_samples=int(val_mask.sum()),
                    best_iteration=best_iter,
                    train_metrics={},
                    val_metrics=val_metrics,
                    duration_s=reg_duration
                )

            return model

        self.regressor = fit_regressor_with_log(obj="regression", alpha=None, model_name="Regressor")
        
        if skip_quantiles:
            self.q10 = None
            self.q90 = None
        else:
            self.q10 = fit_regressor_with_log(obj="quantile", alpha=0.10, model_name="Quantile_10")
            self.q90 = fit_regressor_with_log(obj="quantile", alpha=0.90, model_name="Quantile_90")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rank_score = self.ranker.predict(X) if self.ranker else np.zeros(len(X))
        score_pred = self.regressor.predict(X) if self.regressor else np.zeros(len(X))
        p10 = self.q10.predict(X) if self.q10 else np.full(len(X), np.nan)
        p90 = self.q90.predict(X) if self.q90 else np.full(len(X), np.nan)
        return rank_score, score_pred, p10, p90

    def feature_importance(self, top_n: int = 25) -> List[Tuple[str, float]]:
        if not self.ranker or not self.feature_columns:
            return []
        imp = self.ranker.feature_importances_
        imp = imp / (imp.sum() + 1e-10)
        items = list(zip(self.feature_columns, imp))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_n]


# =============================================================================
# POSITION MODELS
# =============================================================================

class PositionModels:
    def __init__(self):
        self.global_models: List[RankingModel] = []
        self.models: Dict[int, List[RankingModel]] = {}
        self.global_model: Optional[RankingModel] = None
        self.global_params: Optional[Dict[str, Any]] = None

        self.conformal_delta_global: float = 0.0
        self.conformal_delta_by_pos: Dict[int, float] = {}

    def _fit_conformal_ensemble(
        self, 
        models_list: List[RankingModel], 
        X_cal: pd.DataFrame, 
        y_cal: np.ndarray,
        groups_cal: Optional[np.ndarray],  # NOVO: grupos para métricas por rodada
        alpha: float, 
        posicao_id: int = -1
    ) -> float:
        """
        Calibração conformal.
        
        CORREÇÃO: Adicionado groups_cal para poder calcular métricas por rodada se necessário.
        """
        global TRAIN_LOGGER
        
        # CORREÇÃO: Checagem apropriada de None
        if X_cal is None or len(X_cal) < 80 or not models_list:
            return 0.0

        y_cal = np.asarray(y_cal, dtype=float)

        q10s = []
        q90s = []
        for m in models_list:
            # CORREÇÃO: Checagem apropriada de None
            if m.q10 is None or m.q90 is None:
                continue
            q10s.append(m.q10.predict(X_cal))
            q90s.append(m.q90.predict(X_cal))

        if not q10s or not q90s:
            return 0.0

        q10p = np.mean(np.vstack(q10s), axis=0)
        q90p = np.mean(np.vstack(q90s), axis=0)

        inside_before = np.sum((y_cal >= q10p) & (y_cal <= q90p))
        coverage_before = inside_before / len(y_cal)

        s = np.maximum(np.maximum(q10p - y_cal, y_cal - q90p), 0.0)
        delta = _quantile_higher(s, 1.0 - alpha)
        
        inside_after = np.sum((y_cal >= (q10p - delta)) & (y_cal <= (q90p + delta)))
        coverage_after = inside_after / len(y_cal)
        
        if TRAIN_LOGGER:
            TRAIN_LOGGER.log_conformal_calibration(
                posicao_id=posicao_id,
                delta=delta,
                n_calibration=len(y_cal),
                alpha=alpha,
                empirical_coverage=coverage_after,
                coverage_before=coverage_before
            )
        
        return float(max(delta, 0.0))

    def train_all(
        self,
        features_by_pos,
        all_features,
        all_targets,
        all_groups,
        optimize_global: bool = True,
        global_params: Optional[Dict[str, Any]] = None,
    ):
        """Treina ensembles completos (global + por posição) + conformal."""
        global TRAIN_LOGGER

        LOGGER.info("=" * 80)
        LOGGER.info("TRAINING ALL MODELS (ensemble + conformal)")
        LOGGER.info("=" * 80)

        if not all_features:
            raise ValueError("all_features vazio.")

        tmp_cols = RankingModel()._infer_feature_columns(all_features[0])
        LOGGER.info(f"Feature columns: {len(tmp_cols)}")

        Xg_full = pd.DataFrame(all_features)
        yg_full = np.array(all_targets, dtype=float)
        gg_full = np.array(all_groups, dtype=int)

        for c in tmp_cols:
            if c not in Xg_full.columns:
                Xg_full[c] = 0.0
        Xg_full = Xg_full[tmp_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        tr_mask, cal_mask, _ = split_train_cal_by_round(gg_full, calib_rounds=CALIB_ROUNDS)
        Xg_tr, yg_tr, gg_tr = Xg_full[tr_mask], yg_full[tr_mask], gg_full[tr_mask]
        Xg_cal, yg_cal, gg_cal = Xg_full[cal_mask], yg_full[cal_mask], gg_full[cal_mask]

        LOGGER.info(f"Data split: train={tr_mask.sum()}, calibration={cal_mask.sum()}")

        # PARAMS GLOBAIS
        if global_params:
            base_params = dict(global_params)
            LOGGER.info("Using provided global params")
        else:
            if optimize_global:
                LOGGER.info("-" * 60)
                LOGGER.info("OPTUNA GLOBAL OPTIMIZATION")
                LOGGER.info("-" * 60)
                base = RankingModel()
                base.feature_columns = list(tmp_cols)
                best_ndcg = base.optimize(Xg_tr, yg_tr, gg_tr, n_trials=OPTUNA_TRIALS, k=5)  # k=5 consistente
                print(f"   ✓ Melhor NDCG@5: {best_ndcg:.4f}")
                base_params = dict(base.best_params) if base.best_params else None
            else:
                base_params = None

        if not base_params:
            base_params = RankingModel()._default_params()

        self.global_params = dict(base_params)

        # ENSEMBLE GLOBAL
        LOGGER.info("-" * 60)
        LOGGER.info("TRAINING GLOBAL ENSEMBLE")
        LOGGER.info("-" * 60)

        self.global_models = []
        ranker_metrics_global = []

        for seed in ENSEMBLE_SEEDS:
            m = RankingModel()
            m.feature_columns = list(tmp_cols)
            m.best_params_ranker = dict(base_params)
            m.best_params_regressor = dict(base_params)
            m.best_params_regressor["objective"] = "regression"
            m.best_params = dict(base_params)

            m.train(Xg_tr, yg_tr, gg_tr, optimize=False, optimize_regressor=False, seed=seed)
            self.global_models.append(m)

            if cal_mask.sum() >= 40 and m.ranker:
                pred = m.ranker.predict(Xg_cal)
                # CORREÇÃO: Métricas por grupo
                ranker_metrics_global.append({
                    "ndcg@5": RankingMetrics.ndcg_at_k_by_group(pred, yg_cal, gg_cal, k=5, weighted=True),
                    "spearman": RankingMetrics.spearman_by_group(pred, yg_cal, gg_cal, weighted=True),
                })

        if TRAIN_LOGGER and ranker_metrics_global:
            TRAIN_LOGGER.log_ensemble_summary(
                posicao_id=-1,
                n_models=len(self.global_models),
                seeds=ENSEMBLE_SEEDS,
                ranker_metrics=ranker_metrics_global,
                regressor_metrics=[]
            )

        self.global_model = self.global_models[0] if self.global_models else None

        # ENSEMBLES POR POSIÇÃO
        self.models = {}
        pos_trials = min(OPTUNA_TRIALS, 500)

        for pos_id, pos_name in POSICOES.items():
            if pos_id not in features_by_pos:
                continue

            feats, targs, grps = features_by_pos[pos_id]
            if len(feats) < 200:
                LOGGER.debug(f"Skipping {pos_name}: insufficient data ({len(feats)})")
                continue

            LOGGER.info("-" * 60)
            LOGGER.info(f"TRAINING {pos_name} (pos_id={pos_id})")
            LOGGER.info("-" * 60)

            Xp_full = pd.DataFrame(feats)
            yp_full = np.array(targs, dtype=float)
            gp_full = np.array(grps, dtype=int)

            for c in tmp_cols:
                if c not in Xp_full.columns:
                    Xp_full[c] = 0.0
            Xp_full = Xp_full[tmp_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

            trp, calp, _ = split_train_cal_by_round(gp_full, calib_rounds=CALIB_ROUNDS)
            Xp_tr, yp_tr, gp_tr = Xp_full[trp], yp_full[trp], gp_full[trp]
            Xp_cal, yp_cal, gp_cal = Xp_full[calp], yp_full[calp], gp_full[calp]

            # Usar parâmetros globais como base (sem otimização por posição para reduzir overfitting)
            reg_params_pos = dict(base_params)
            reg_params_pos["objective"] = "regression"

            ens_list = []
            ranker_metrics_pos = []

            for seed in ENSEMBLE_SEEDS:
                mp = RankingModel(posicao_id=pos_id)
                mp.feature_columns = list(tmp_cols)
                mp.best_params_ranker = dict(base_params)
                mp.best_params_regressor = dict(reg_params_pos)
                mp.best_params = dict(base_params)

                mp.train(Xp_tr, yp_tr, gp_tr, optimize=False, optimize_regressor=False, seed=seed)
                ens_list.append(mp)

                if calp.sum() >= 40 and mp.ranker:
                    pred = mp.ranker.predict(Xp_cal)
                    # CORREÇÃO: Métricas por grupo
                    ranker_metrics_pos.append({
                        "ndcg@5": RankingMetrics.ndcg_at_k_by_group(pred, yp_cal, gp_cal, k=5, weighted=True),
                        "spearman": RankingMetrics.spearman_by_group(pred, yp_cal, gp_cal, weighted=True),
                    })

            self.models[pos_id] = ens_list

            if TRAIN_LOGGER and ranker_metrics_pos:
                TRAIN_LOGGER.log_ensemble_summary(
                    posicao_id=pos_id,
                    n_models=len(ens_list),
                    seeds=ENSEMBLE_SEEDS,
                    ranker_metrics=ranker_metrics_pos,
                    regressor_metrics=[]
                )

        # CONFORMAL CALIBRATION
        LOGGER.info("-" * 60)
        LOGGER.info("CONFORMAL CALIBRATION")
        LOGGER.info("-" * 60)

        self.conformal_delta_global = self._fit_conformal_ensemble(
            models_list=self.global_models,
            X_cal=Xg_cal,
            y_cal=yg_cal,
            groups_cal=gg_cal,
            alpha=CONFORMAL_ALPHA,
            posicao_id=-1
        )

        self.conformal_delta_by_pos = {}
        for pos_id, ens_list in self.models.items():
            feats, targs, grps = features_by_pos[pos_id]

            Xp_full = pd.DataFrame(feats)
            yp_full = np.array(targs, dtype=float)
            gp_full = np.array(grps, dtype=int)

            for c in tmp_cols:
                if c not in Xp_full.columns:
                    Xp_full[c] = 0.0
            Xp_full = Xp_full[tmp_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

            _, calp, _ = split_train_cal_by_round(gp_full, calib_rounds=CALIB_ROUNDS)
            Xp_cal = Xp_full[calp]
            yp_cal = yp_full[calp]
            gp_cal = gp_full[calp]

            dpos = self._fit_conformal_ensemble(
                ens_list, Xp_cal, yp_cal, gp_cal,
                alpha=CONFORMAL_ALPHA,
                posicao_id=pos_id
            )
            self.conformal_delta_by_pos[pos_id] = dpos

        LOGGER.info("=" * 80)
        LOGGER.info("TRAINING COMPLETE")
        LOGGER.info("=" * 80)

    def train_all_fast(
        self,
        features_by_pos,
        all_features,
        all_targets,
        all_groups,
        global_params: Optional[Dict[str, Any]] = None,
        ensemble_seeds: Optional[List[int]] = None,
        skip_quantiles: bool = True,
    ):
        """
        Treino rápido para backtest.
        - Usa seeds reduzidos (default: 3)
        - Pula q10/q90 se skip_quantiles=True
        - Sem Optuna (usa global_params)
        - Sem conformal
        """
        if ensemble_seeds is None:
            ensemble_seeds = BACKTEST_ENSEMBLE_SEEDS

        if not all_features:
            raise ValueError("all_features vazio.")

        tmp_cols = RankingModel()._infer_feature_columns(all_features[0])

        Xg_full = pd.DataFrame(all_features)
        yg_full = np.array(all_targets, dtype=float)
        gg_full = np.array(all_groups, dtype=int)

        for c in tmp_cols:
            if c not in Xg_full.columns:
                Xg_full[c] = 0.0
        Xg_full = Xg_full[tmp_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        if not global_params:
            global_params = RankingModel()._default_params()

        self.global_params = dict(global_params)

        # ENSEMBLE GLOBAL (seeds reduzidos)
        self.global_models = []
        for seed in ensemble_seeds:
            m = RankingModel()
            m.feature_columns = list(tmp_cols)
            m.best_params_ranker = dict(global_params)
            m.best_params_regressor = dict(global_params)
            m.best_params_regressor["objective"] = "regression"
            m.best_params = dict(global_params)

            m.train(Xg_full, yg_full, gg_full, 
                   optimize=False, optimize_regressor=False, 
                   seed=seed, skip_quantiles=skip_quantiles)
            self.global_models.append(m)

        self.global_model = self.global_models[0] if self.global_models else None

        # ENSEMBLES POR POSIÇÃO (seeds reduzidos)
        self.models = {}
        for pos_id, pos_name in POSICOES.items():
            if pos_id not in features_by_pos:
                continue

            feats, targs, grps = features_by_pos[pos_id]
            if len(feats) < 200:
                continue

            Xp_full = pd.DataFrame(feats)
            yp_full = np.array(targs, dtype=float)
            gp_full = np.array(grps, dtype=int)

            for c in tmp_cols:
                if c not in Xp_full.columns:
                    Xp_full[c] = 0.0
            Xp_full = Xp_full[tmp_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)

            ens_list = []
            for seed in ensemble_seeds:
                mp = RankingModel(posicao_id=pos_id)
                mp.feature_columns = list(tmp_cols)
                mp.best_params_ranker = dict(global_params)
                mp.best_params_regressor = dict(global_params)
                mp.best_params_regressor["objective"] = "regression"
                mp.best_params = dict(global_params)

                mp.train(Xp_full, yp_full, gp_full, 
                        optimize=False, optimize_regressor=False, 
                        seed=seed, skip_quantiles=skip_quantiles)
                ens_list.append(mp)

            self.models[pos_id] = ens_list

    def predict(self, posicao_id: int, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predição com ensemble e ajuste conformal."""
        ens = self.models.get(posicao_id)
        if not ens:
            ens = self.global_models if self.global_models else []

        if not ens:
            n = len(X)
            return np.zeros(n), np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)

        rank_list, score_list, p10_list, p90_list = [], [], [], []

        for m in ens:
            rs, sc, lo, hi = m.predict(X)
            rank_list.append(rs)
            score_list.append(sc)
            if not np.isnan(lo).all():
                p10_list.append(lo)
            if not np.isnan(hi).all():
                p90_list.append(hi)

        rank_score = np.mean(np.vstack(rank_list), axis=0)
        score_pred = np.mean(np.vstack(score_list), axis=0)
        
        if p10_list:
            p10 = np.mean(np.vstack(p10_list), axis=0)
        else:
            p10 = np.full(len(X), np.nan)
            
        if p90_list:
            p90 = np.mean(np.vstack(p90_list), axis=0)
        else:
            p90 = np.full(len(X), np.nan)

        d = self.conformal_delta_by_pos.get(posicao_id, self.conformal_delta_global)
        if d and d > 0 and not np.isnan(p10).all():
            p10 = p10 - float(d)
            p90 = p90 + float(d)

        return rank_score, score_pred, p10, p90

    def get_model(self, posicao_id: int) -> RankingModel:
        ens = self.models.get(posicao_id)
        if ens and len(ens) > 0:
            return ens[0]
        if self.global_model:
            return self.global_model
        return RankingModel()


# =============================================================================
# BUILD AND TRAIN / GENERATE PREDICTIONS
# =============================================================================

from .io import OddsCache, MatchOdds, CartolaAPI, safe_int, parse_bool, ensure_probability_simplex
from .features import TemporalFeatureEngineer
from .report import PlayerPrediction


def build_and_train(
    df: pd.DataFrame,
    fe: TemporalFeatureEngineer,
    optimize_global: bool,
    global_params: Optional[Dict[str, Any]] = None,
) -> PositionModels:
    """Constrói e treina os modelos usando temporal_id como grupo."""
    LOGGER.info("Preparando treino final...")
    OddsCache.clear()

    features_by_pos = defaultdict(lambda: ([], [], []))
    all_f, all_t, all_g = [], [], []

    valid = df[df["entrou_em_campo"] == True].copy()
    if "p_team_win" not in valid.columns:
        raise ValueError("CSV não tem p_team_win.")
    
    if "temporal_id" not in valid.columns:
        raise ValueError("DataFrame não tem temporal_id.")

    # CORREÇÃO: Usar itertuples() em vez de iterrows()
    for row in valid.itertuples(index=False):
        # CORREÇÃO: Checagem apropriada de NaN
        p_win_val = getattr(row, 'p_team_win', None)
        if p_win_val is None or pd.isna(p_win_val):
            continue
            
        is_home = parse_bool(getattr(row, 'is_home', False))
        p_win = float(p_win_val)
        
        p_draw_val = getattr(row, 'p_draw', None)
        p_draw = float(p_draw_val) if p_draw_val is not None and not pd.isna(p_draw_val) else 1/3
        
        p_lose_val = getattr(row, 'p_team_lose', None)
        if p_lose_val is None or pd.isna(p_lose_val):
            p_lose = max(0.0, 1.0 - p_win - p_draw)
        else:
            p_lose = float(p_lose_val)

        if is_home:
            p_home, p_away = p_win, p_lose
        else:
            p_home, p_away = p_lose, p_win
        p_home, p_draw, p_away = ensure_probability_simplex(p_home, p_draw, p_away)
        mo = OddsCache.get_or_create(p_home, p_draw, p_away)

        atleta_id = safe_int(getattr(row, 'atleta_id', None))
        pos_id = safe_int(getattr(row, 'posicao_id', None))
        clube_id = safe_int(getattr(row, 'clube_id', None))
        opp_id = safe_int(getattr(row, 'opponent_id', None))
        tid = safe_int(getattr(row, 'temporal_id', None))
        
        # CORREÇÃO: Checagem apropriada de None
        if atleta_id is None or pos_id is None or clube_id is None or opp_id is None or tid is None:
            continue

        feats = fe.calculate_all_features(atleta_id, pos_id, clube_id, opp_id, mo, is_home, tid)
        if not feats:
            continue

        pontuacao = float(getattr(row, 'pontuacao', 0.0))
        
        all_f.append(feats)
        all_t.append(pontuacao)
        all_g.append(int(tid))

        pos_feats, pos_targs, pos_grps = features_by_pos[pos_id]
        pos_feats.append(feats)
        pos_targs.append(pontuacao)
        pos_grps.append(int(tid))

    LOGGER.info(f"Features extraídas: {len(all_f)} registros")

    pm = PositionModels()
    pm.train_all(features_by_pos, all_f, all_t, all_g, optimize_global=optimize_global, global_params=global_params)
    return pm


def generate_predictions(
    api: CartolaAPI,
    models: PositionModels,
    fe: TemporalFeatureEngineer,
    odds_by_clube: Dict[int, MatchOdds],
    atletas: List[Dict],
    rodada_atual: int,
    temporada: int = 2025,
) -> List[PlayerPrediction]:
    """Gera previsões para a rodada atual."""
    from .config import calculate_temporal_id, POSICOES
    
    tid_atual = calculate_temporal_id(temporada, rodada_atual)
    
    preds = []
    for a in atletas:
        atleta_id = safe_int(a.get("atleta_id"))
        pos_id = safe_int(a.get("posicao_id"))
        clube_id = safe_int(a.get("clube_id"))
        
        # CORREÇÃO: Checagem apropriada de None
        if atleta_id is None or pos_id is None or clube_id is None:
            continue

        mo = odds_by_clube.get(clube_id)
        if mo is None:
            continue

        is_home = mo.home_id == clube_id
        opp_id = mo.away_id if is_home else mo.home_id

        feats = fe.calculate_all_features(atleta_id, pos_id, clube_id, opp_id, mo, is_home, tid_atual)
        if not feats:
            continue

        m = models.get_model(pos_id)
        Xp = m.prepare_features([feats])
        rank_score, score_pred, p10, p90 = models.predict(pos_id, Xp)

        p = PlayerPrediction(
            atleta_id=atleta_id,
            posicao_id=pos_id,
            posicao=POSICOES.get(pos_id, "?"),
            apelido=a.get("apelido", "?"),
            clube_id=clube_id,
            clube=a.get("clube_nome", "?"),
            opponent_id=opp_id,
            opponent=a.get("adversario_nome", "?"),
            is_home=is_home,
            status=a.get("status_id", "?"),
            preco=float(a.get("preco", 0)),
            predicted_score=float(score_pred[0]),
            pred_p10=float(p10[0]) if not np.isnan(p10[0]) else None,
            pred_p90=float(p90[0]) if not np.isnan(p90[0]) else None,
            rank_score=float(rank_score[0]),
            features=feats,
        )
        preds.append(p)

    return preds
