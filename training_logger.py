import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Constantes de posições
POSICOES = {
    1: "Goleiro",
    2: "Lateral",
    3: "Zagueiro",
    4: "Meia",
    5: "Atacante",
    6: "Técnico",
}

LOGGER = logging.getLogger("cartola")


def _now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        # lida com numpy scalars
        return float(x)
    except Exception:
        return None


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class MetricsRecorder:
    """
    Grava métricas em CSV para análise posterior (formato longo).
    Colunas:
      ts, stage, posicao_id, seed, rodada, trial, fold, metric, value, extra
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_file()

    def _init_file(self):
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts", "stage", "posicao_id", "seed", "rodada",
                    "trial", "fold", "metric", "value", "extra"
                ])

    def log(
        self,
        stage: str,
        metric: str,
        value: float,
        posicao_id: int = -1,
        seed: int = -1,
        rodada: int = -1,
        trial: int = -1,
        fold: int = -1,
        extra: str = ""
    ):
        v = _to_float_or_none(value)
        if v is None:
            return  # só registramos numéricos aqui

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                _now_ts(),
                stage,
                int(posicao_id),
                int(seed),
                int(rodada),
                int(trial),
                int(fold),
                str(metric),
                float(v),
                str(extra) if extra is not None else ""
            ])

    def write_row(self, row: Dict[str, Any]):
        """
        Compatibilidade com chamadas antigas que escreviam "linha wide".
        Converte para vários registros no formato longo.
        """
        if not isinstance(row, dict) or not row:
            return

        stage = str(row.get("event", row.get("stage", "event")))
        posicao_id = int(row.get("posicao_id", -1) or -1)
        seed = int(row.get("seed", -1) or -1)
        rodada = int(row.get("rodada", -1) or -1)
        trial = int(row.get("trial", -1) or -1)
        fold = int(row.get("fold", -1) or -1)

        # Campos "fixos" que queremos logar como métricas também (se existirem)
        fixed_numeric = ["n_predictions", "duration_s", "train_samples", "val_samples", "best_iteration", "total_time_s"]

        for k in fixed_numeric:
            if k in row:
                v = _to_float_or_none(row.get(k))
                if v is not None:
                    self.log(stage=stage, metric=k, value=v,
                             posicao_id=posicao_id, seed=seed, rodada=rodada,
                             trial=trial, fold=fold)

        # Métricas prefixadas (ex: m_ndcg@10)
        for k, v in row.items():
            if not isinstance(k, str):
                continue
            if k.startswith("m_"):
                fv = _to_float_or_none(v)
                if fv is not None:
                    self.log(stage=stage, metric=k[2:], value=fv,
                             posicao_id=posicao_id, seed=seed, rodada=rodada,
                             trial=trial, fold=fold)

        # Qualquer coisa não numérica vai em extra (compactado)
        extras = {}
        for k, v in row.items():
            if k in ("event", "stage", "posicao_id", "seed", "rodada", "trial", "fold"):
                continue
            if isinstance(k, str) and (k.startswith("m_") or k in fixed_numeric):
                continue
            if _to_float_or_none(v) is None:
                extras[k] = v

        if extras:
            # guarda um "ping" numérico para marcar que houve extras
            self.log(stage=stage, metric="has_extra", value=1.0,
                     posicao_id=posicao_id, seed=seed, rodada=rodada,
                     trial=trial, fold=fold, extra=_safe_json(extras))


class TrainingLogger:
    """
    Logger de treino/backtest que:
      - grava métricas numéricas no MetricsRecorder (CSV longo)
      - grava payloads grandes em JSONL (um arquivo por tipo)
    """

    def __init__(self, recorder: Optional[MetricsRecorder] = None, log_dir: str = "logs"):
        self.recorder = recorder
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # JSONL "grandes"
        self.optuna_trials_jsonl = self.log_dir / "optuna_trials.jsonl"
        self.optuna_summaries_jsonl = self.log_dir / "optuna_summaries.jsonl"
        self.model_training_jsonl = self.log_dir / "model_training.jsonl"
        self.conformal_jsonl = self.log_dir / "conformal.jsonl"
        self.ensemble_jsonl = self.log_dir / "ensemble.jsonl"
        self.backtest_rounds_jsonl = self.log_dir / "backtest_rounds.jsonl"
        self.backtest_summary_jsonl = self.log_dir / "backtest_summary.jsonl"
        self.final_summary_jsonl = self.log_dir / "final_summary.jsonl"

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]):
        payload = dict(payload)
        payload["ts"] = payload.get("ts", _now_ts())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    # ---------------------------
    # OPTUNA
    # ---------------------------
    def log_optuna_trial(
        self,
        trial_number: int,
        params: Dict[str, Any],
        score: float,
        fold_scores: List[float],
        posicao_id: int = -1,
        duration_s: float = 0.0,
    ):
        self._append_jsonl(self.optuna_trials_jsonl, {
            "event": "optuna_trial",
            "posicao_id": int(posicao_id),
            "trial": int(trial_number),
            "score": float(score),
            "fold_scores": fold_scores,
            "duration_s": float(duration_s),
            "params": params,
        })

        if self.recorder:
            self.recorder.log("optuna_trial", "score", float(score), posicao_id=posicao_id, trial=trial_number)
            self.recorder.log("optuna_trial", "duration_s", float(duration_s), posicao_id=posicao_id, trial=trial_number)
            # fold scores (se quiser analisar variância)
            for i, fs in enumerate(fold_scores or []):
                fv = _to_float_or_none(fs)
                if fv is not None:
                    self.recorder.log("optuna_trial", f"fold_score_{i}", float(fv), posicao_id=posicao_id, trial=trial_number)

    def log_optuna_summary(
        self,
        best_trial: int,
        best_score: float,
        best_params: Dict[str, Any],
        total_trials: int,
        total_time_s: float,
        posicao_id: int = -1,
    ):
        self._append_jsonl(self.optuna_summaries_jsonl, {
            "event": "optuna_summary",
            "posicao_id": int(posicao_id),
            "best_trial": int(best_trial),
            "best_score": float(best_score),
            "total_trials": int(total_trials),
            "total_time_s": float(total_time_s),
            "best_params": best_params,
        })

        if self.recorder:
            self.recorder.log("optuna_summary", "best_score", float(best_score), posicao_id=posicao_id)
            self.recorder.log("optuna_summary", "total_trials", float(total_trials), posicao_id=posicao_id)
            self.recorder.log("optuna_summary", "total_time_s", float(total_time_s), posicao_id=posicao_id)

    # ---------------------------
    # TREINO MODELOS
    # ---------------------------
    def log_model_training(
        self,
        model_type: str,
        posicao_id: int,
        seed: int,
        train_samples: int,
        val_samples: int,
        best_iteration: Optional[int],
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        duration_s: float,
    ):
        self._append_jsonl(self.model_training_jsonl, {
            "event": "model_training",
            "model_type": str(model_type),
            "posicao_id": int(posicao_id),
            "seed": int(seed),
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
            "train_metrics": train_metrics or {},
            "val_metrics": val_metrics or {},
            "duration_s": float(duration_s),
        })

        if self.recorder:
            stage = f"train_{model_type}"
            self.recorder.log(stage, "duration_s", float(duration_s), posicao_id=posicao_id, seed=seed)
            self.recorder.log(stage, "train_samples", float(train_samples), posicao_id=posicao_id, seed=seed)
            self.recorder.log(stage, "val_samples", float(val_samples), posicao_id=posicao_id, seed=seed)
            if best_iteration is not None:
                self.recorder.log(stage, "best_iteration", float(best_iteration), posicao_id=posicao_id, seed=seed)

            for k, v in (val_metrics or {}).items():
                fv = _to_float_or_none(v)
                if fv is not None:
                    self.recorder.log(stage, f"val_{k}", float(fv), posicao_id=posicao_id, seed=seed)

            for k, v in (train_metrics or {}).items():
                fv = _to_float_or_none(v)
                if fv is not None:
                    self.recorder.log(stage, f"train_{k}", float(fv), posicao_id=posicao_id, seed=seed)

    # ---------------------------
    # CONFORMAL
    # ---------------------------
    def log_conformal_calibration(
        self,
        posicao_id: int,
        delta: float,
        n_calibration: int,
        alpha: float,
        empirical_coverage: float,
        coverage_before: Optional[float] = None,
    ):
        self._append_jsonl(self.conformal_jsonl, {
            "event": "conformal_calibration",
            "posicao_id": int(posicao_id),
            "delta": float(delta),
            "n_calibration": int(n_calibration),
            "alpha": float(alpha),
            "empirical_coverage": float(empirical_coverage),
            "coverage_before": float(coverage_before) if coverage_before is not None else None,
        })

        if self.recorder:
            self.recorder.log("conformal", "delta", float(delta), posicao_id=posicao_id)
            self.recorder.log("conformal", "empirical_coverage", float(empirical_coverage), posicao_id=posicao_id)
            if coverage_before is not None:
                self.recorder.log("conformal", "coverage_before", float(coverage_before), posicao_id=posicao_id)

    # ---------------------------
    # ENSEMBLE
    # ---------------------------
    def log_ensemble_summary(
        self,
        posicao_id: int,
        n_models: int,
        seeds: List[int],
        ranker_metrics: List[Dict[str, Any]],
        regressor_metrics: List[Dict[str, Any]],
    ):
        self._append_jsonl(self.ensemble_jsonl, {
            "event": "ensemble_summary",
            "posicao_id": int(posicao_id),
            "n_models": int(n_models),
            "seeds": list(seeds or []),
            "ranker_metrics": ranker_metrics or [],
            "regressor_metrics": regressor_metrics or [],
        })

        # Loga médias simples no CSV (útil pra comparar execuções)
        if self.recorder and ranker_metrics:
            # calcula médias por chave
            keys = set()
            for d in ranker_metrics:
                keys.update(d.keys())
            for k in keys:
                vals = []
                for d in ranker_metrics:
                    fv = _to_float_or_none(d.get(k))
                    if fv is not None:
                        vals.append(float(fv))
                if vals:
                    self.recorder.log("ensemble", f"ranker_{k}_mean", sum(vals) / len(vals), posicao_id=posicao_id)

    # ---------------------------
    # BACKTEST
    # ---------------------------
    def log_backtest_round(
        self,
        rodada: int,
        n_predictions: int,
        metrics: Dict[str, Any],
        duration_s: float,
        metrics_by_pos: Optional[Dict[str, Any]] = None,
        n_by_pos: Optional[Dict[str, Any]] = None,
        metrics_macro: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # CSV (flat)
        if self.recorder is not None:
            row = {
                "event": "backtest_round",
                "rodada": int(rodada),
                "n_predictions": int(n_predictions),
                "duration_s": float(duration_s),
            }
            for k, v in (metrics or {}).items():
                row[f"m_{k}"] = v
            # mantém compat
            self.recorder.write_row(row)

        # JSONL (detalhes grandes)
        self._append_jsonl(self.backtest_rounds_jsonl, {
            "event": "backtest_round",
            "rodada": int(rodada),
            "n_predictions": int(n_predictions),
            "duration_s": float(duration_s),
            "metrics_weighted": metrics or {},
            "metrics_by_pos": metrics_by_pos,
            "n_by_pos": n_by_pos,
            "metrics_macro": metrics_macro,
            **kwargs,
        })

    def log_backtest_summary(self, summary: Dict[str, Any]):
        self._append_jsonl(self.backtest_summary_jsonl, {
            "event": "backtest_summary",
            "summary": summary or {},
        })

        # Também registra alguns “principais” se existirem
        if self.recorder:
            overall = (summary or {}).get("overall_weighted", {}) or {}
            # tenta puxar mean de métricas comuns
            for k in ["ndcg@5", "ndcg@10", "hit_rate@5", "lift@5", "interval_coverage", "mae", "rmse"]:
                v = overall.get(k, None)
                if isinstance(v, dict):
                    fv = _to_float_or_none(v.get("mean"))
                else:
                    fv = _to_float_or_none(v)
                if fv is not None:
                    self.recorder.log("backtest_summary", k, float(fv))

    # ---------------------------
    # FINAL
    # ---------------------------
    def log_final_summary(
        self,
        backtest_results: Dict[str, Any],
        n_predictions: int,
        elapsed_time_s: float,
    ):
        self._append_jsonl(self.final_summary_jsonl, {
            "event": "final_summary",
            "n_predictions": int(n_predictions),
            "elapsed_time_s": float(elapsed_time_s),
            "backtest_results": backtest_results or {},
        })

        if self.recorder:
            self.recorder.log("final", "n_predictions", float(n_predictions))
            self.recorder.log("final", "elapsed_time_s", float(elapsed_time_s))

    def save_reports(self, output_dir: str = "logs"):
        """
        Best-effort: gera CSVs consolidados a partir dos JSONL.
        Não falha se não houver arquivos.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        try:
            import pandas as pd
        except Exception:
            LOGGER.warning("pandas não disponível; save_reports() vai pular relatórios.")
            return

        def jsonl_to_df(path: Path) -> Optional["pd.DataFrame"]:
            if not path.exists():
                return None
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        continue
            if not rows:
                return None
            return pd.DataFrame(rows)

        mapping = {
            "optuna_trials": self.optuna_trials_jsonl,
            "optuna_summaries": self.optuna_summaries_jsonl,
            "model_training": self.model_training_jsonl,
            "conformal": self.conformal_jsonl,
            "ensemble": self.ensemble_jsonl,
            "backtest_rounds": self.backtest_rounds_jsonl,
            "backtest_summary": self.backtest_summary_jsonl,
            "final_summary": self.final_summary_jsonl,
        }

        for name, path in mapping.items():
            df = jsonl_to_df(path)
            if df is None or df.empty:
                continue
            # evita colunas com dict/list enormes atrapalharem: transforma em string
            for c in df.columns:
                if df[c].map(lambda x: isinstance(x, (dict, list))).any():
                    df[c] = df[c].map(lambda x: _safe_json(x) if isinstance(x, (dict, list)) else x)
            df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> str:
    """
    Configura logging para arquivo e console.
    Retorna o caminho do arquivo de log.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"cartola_{run_id}.log"

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    LOGGER.addHandler(sh)
    LOGGER.addHandler(fh)

    LOGGER.info(f"Log file: {log_path}")
    return str(log_path)