#!/usr/bin/env python3
"""
CARTOLA FC - Ponto de Entrada Principal (Multi-Temporada)
==========================================================
Versão atualizada que suporta múltiplas temporadas usando temporal_id.

temporal_id = (temporada - 2023) * 38 + rodada_id

Uso:
    python main.py
    
    # Ou com arquivos específicos:
    python main.py --csv "player_games_with_odds_*.csv"
"""

import os
import sys
import time
import logging
import warnings
import argparse
from glob import glob

# Suprimir warnings de feature names do sklearn/lightgbm
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd

from training_logger import (
    TrainingLogger, 
    MetricsRecorder, 
    setup_logging,
    LOGGER,
)

from cartola import (
    # Config
    BACKTEST_RODADAS, OPTUNA_TRIALS, POSICOES, get_default_paths,
    calculate_temporal_id, decompose_temporal_id,
    
    # IO
    CartolaAPI, load_multi_year_data, load_historical_data, load_odds,
    OddsCache, safe_int, parse_bool, ensure_probability_simplex,
    save_models,
    
    # Validation
    DataValidator,
    
    # Features
    TemporalFeatureEngineer,
    
    # Models
    RankingModel, PositionModels,
    build_and_train, generate_predictions,
    
    # Evaluation
    RankingBacktester,
    
    # Report
    print_backtest, print_importance, print_matches, save_excel,
)

# Variáveis globais de logging
TRAIN_LOGGER = None
REC = None

# Verificar se ML está disponível
try:
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Dependências faltando: {e}")
    print("   pip install lightgbm optuna scipy scikit-learn pandas numpy requests")
    ML_AVAILABLE = False


def find_csv_files(base_dir: str, pattern: str = "player_games_with_odds_*.csv"):
    """
    Encontra arquivos CSV que correspondem ao padrão.
    
    Args:
        base_dir: Diretório base para busca
        pattern: Padrão glob para arquivos
        
    Returns:
        Lista de caminhos de arquivos encontrados
    """
    full_pattern = os.path.join(base_dir, pattern)
    files = sorted(glob(full_pattern))
    return files


def main():
    global TRAIN_LOGGER, REC
    
    if not ML_AVAILABLE:
        sys.exit(1)

    # Parse argumentos
    parser = argparse.ArgumentParser(description="Cartola FC - Previsão Multi-Temporada")
    parser.add_argument("--csv", type=str, default=None, 
                        help="Padrão glob para arquivos CSV (ex: 'player_games_with_odds_*.csv')")
    parser.add_argument("--single", type=str, default=None,
                        help="Arquivo CSV único (modo legado)")
    args = parser.parse_args()

    start_time = time.time()

    # ========== INICIALIZAÇÃO DOS LOGS ==========
    log_path = setup_logging(log_dir="logs", level=logging.INFO)
    
    REC = MetricsRecorder("logs/train_metrics.csv")
    TRAIN_LOGGER = TrainingLogger(recorder=REC)
    
    # Injeta nos módulos que precisam
    import cartola.models as models_module
    import cartola.evaluation as eval_module
    models_module.TRAIN_LOGGER = TRAIN_LOGGER
    eval_module.TRAIN_LOGGER = TRAIN_LOGGER
    
    LOGGER.info("=" * 100)
    LOGGER.info("🏆 CARTOLA FC - Ranking Edition v3.1 - MULTI-TEMPORADA")
    LOGGER.info("=" * 100)
    # ============================================

    print("=" * 100)
    print(" 🏆 CARTOLA FC - Ranking Edition v3.1 - MULTI-TEMPORADA")
    print(" ✓ Suporte a múltiplas temporadas | temporal_id contínuo")
    print(" ✓ Validação dura | Sem vazamento | CV temporal | Ranker | p10/p90 + cobertura")
    print("=" * 100)

    script_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else os.getcwd()
    )

    paths = get_default_paths(script_dir)
    odds_path = paths["odds"]
    output_path = paths["output"]
    model_path = paths["models"]

    # API
    print("\n🌐 API...")
    api = CartolaAPI()
    status = api.get_status()
    if not status:
        LOGGER.error("API falhou")
        sys.exit(1)

    rodada_atual = int(status.get("rodada_atual", 1))
    temporada = int(status.get("temporada", 2025))
    LOGGER.info(f"API: Rodada {rodada_atual} | Temporada {temporada}")
    print(f"   ✓ Rodada {rodada_atual} | Temporada {temporada}")

    atletas, clubes, _ = api.get_mercado()
    partidas = api.get_partidas()

    if not atletas or not partidas:
        LOGGER.error("Dados incompletos (atletas/partidas)")
        sys.exit(1)

    provaveis = [a for a in atletas if a.get("status_id") in [7, 2]]
    LOGGER.info(f"Atletas: {len(atletas)} | Prováveis/Dúvida: {len(provaveis)}")
    print(f"   ✓ Atletas: {len(atletas)} | Prováveis/Dúvida: {len(provaveis)}")

    # Odds + validação
    if not os.path.exists(odds_path):
        LOGGER.error(f"OddsCasas.txt não encontrado: {odds_path}")
        sys.exit(1)

    odds_by_clube, odds_pairs, odds_pair_counts, unknown_abbrs = load_odds(odds_path, api)

    print("\n🔍 Validação (DURA)...")
    validator = DataValidator()
    ok = validator.validate_all(
        status=status,
        partidas=partidas,
        clubes=clubes,
        odds_pairs=odds_pairs,
        odds_pair_counts=odds_pair_counts,
        odds_unknown_abbrs=unknown_abbrs,
        atletas=atletas,
        rodada_partidas=api.partidas_rodada,
        strict=True,
    )
    if not ok:
        LOGGER.error("Validação falhou. Corrija odds/jogos/abreviações antes de rodar.")
        sys.exit(1)
    LOGGER.info("Validação OK")
    print("   ✓ Validação OK")

    # ========== CARREGAMENTO DE DADOS (MULTI-TEMPORADA) ==========
    print("\n📂 Carregando dados históricos...")
    
    if args.single:
        # Modo legado: arquivo único
        csv_path = args.single
        if not os.path.exists(csv_path):
            LOGGER.error(f"CSV não encontrado: {csv_path}")
            sys.exit(1)
        df = load_historical_data(csv_path)
    else:
        # Modo multi-temporada
        if args.csv:
            csv_pattern = args.csv
        else:
            # Buscar arquivos automaticamente
            csv_files = find_csv_files(script_dir, "player_games_with_odds_*.csv")
            if not csv_files:
                # Fallback para arquivo único
                csv_path = paths["csv"]
                if os.path.exists(csv_path):
                    csv_files = [csv_path]
                else:
                    LOGGER.error(f"Nenhum CSV encontrado em: {script_dir}")
                    sys.exit(1)
            csv_pattern = csv_files
        
        df = load_multi_year_data(csv_pattern)
    
    # Estatísticas do dataset
    temporadas_loaded = sorted(df["temporada"].dropna().unique().tolist())
    min_tid = df["temporal_id"].min()
    max_tid = df["temporal_id"].max()
    
    print(f"   ✓ {len(df)} registros carregados")
    print(f"   ✓ Temporadas: {temporadas_loaded}")
    print(f"   ✓ Temporal IDs: {min_tid} → {max_tid}")
    
    LOGGER.info(f"Dataset: {len(df)} registros | Temporadas: {temporadas_loaded} | TIDs: {min_tid}-{max_tid}")

    # Feature Engineering temporal
    fe = TemporalFeatureEngineer(df)

    # ========== BACKTEST (USANDO TEMPORAL_ID) ==========
    max_tid = int(df["temporal_id"].max())
    start_bt = max(5, max_tid - BACKTEST_RODADAS + 1)

    global_params = None
    pre_bt = df[(df["temporal_id"] < start_bt) & (df["entrou_em_campo"] == True)].copy()
    if len(pre_bt) >= 800 and "p_team_win" in pre_bt.columns:
        start_temp, start_rd = decompose_temporal_id(start_bt)
        print(f"\n⚙️ Otimizando parâmetros globais (somente até temporal_id {start_bt - 1})...")
        LOGGER.info(f"Otimizando parâmetros globais (até temporal_id {start_bt - 1})")
        
        OddsCache.clear()
        feats_list, targets, groups = [], [], []

        for _, row in pre_bt.iterrows():
            if pd.isna(row.get("p_team_win", np.nan)):
                continue
            is_home = parse_bool(row.get("is_home", False))
            p_win = float(row["p_team_win"])
            p_draw = float(row.get("p_draw", 1 / 3) if not pd.isna(row.get("p_draw", np.nan)) else 1 / 3)
            p_lose = row.get("p_team_lose", None)
            if p_lose is None or (isinstance(p_lose, float) and np.isnan(p_lose)):
                p_lose = max(0.0, 1.0 - p_win - p_draw)
            p_lose = float(p_lose)

            if is_home:
                p_home, p_away = p_win, p_lose
            else:
                p_home, p_away = p_lose, p_win
            p_home, p_draw, p_away = ensure_probability_simplex(p_home, p_draw, p_away)
            mo = OddsCache.get_or_create(p_home, p_draw, p_away)

            atleta_id = safe_int(row["atleta_id"])
            pos_id = safe_int(row["posicao_id"])
            clube_id = safe_int(row["clube_id"])
            opp_id = safe_int(row["opponent_id"])
            tid = safe_int(row["temporal_id"])
            if None in [atleta_id, pos_id, clube_id, opp_id, tid]:
                continue

            feats = fe.calculate_all_features(atleta_id, pos_id, clube_id, opp_id, mo, is_home, tid)
            if not feats:
                continue

            feats_list.append(feats)
            targets.append(float(row["pontuacao"]))
            groups.append(int(tid))

        if len(feats_list) >= 800:
            tmp = RankingModel()
            Xtmp = tmp.prepare_features(feats_list)
            ytmp = np.array(targets, dtype=float)
            gtmp = np.array(groups, dtype=int)
            _ = tmp.optimize(Xtmp, ytmp, gtmp, n_trials=max(30, OPTUNA_TRIALS // 2))
            global_params = dict(tmp.best_params) if tmp.best_params else None
            LOGGER.info("Params globais (pré-backtest) definidos")
            print("   ✓ Params globais (pré-backtest) definidos.")
        else:
            LOGGER.warning("Dados insuficientes para otimização pré-backtest")
            print("   ⚠️ Dados insuficientes para otimização pré-backtest.")

    # Calcular descrição do período de backtest
    start_temp, start_rd = decompose_temporal_id(start_bt)
    end_temp, end_rd = decompose_temporal_id(max_tid)
    print(f"\n🔄 Backtest (temporal_ids {start_bt}-{max_tid})...")
    print(f"   ({start_temp}/R{start_rd} → {end_temp}/R{end_rd})")
    
    bt = RankingBacktester(df, fe)
    k_pos = 5
    bt_results = bt.run(start_bt, max_tid, global_params=global_params, k_pos=k_pos, rank_by="score")
    print_backtest(bt_results, k_pos=k_pos)

    optimize_global = (global_params is None)
    models = build_and_train(df, fe, optimize_global=optimize_global, global_params=global_params)

    # Salvar modelos
    save_models(models, fe, model_path)

    # Importance
    print_importance(models)

    # Partidas
    print_matches(odds_by_clube)

    # Previsões
    print(f"\n🔮 Gerando previsões (rodada {rodada_atual}, temporada {temporada})...")
    preds = generate_predictions(api, models, fe, odds_by_clube, atletas, rodada_atual, temporada)
    LOGGER.info(f"Previsões geradas: {len(preds)} jogadores")
    print(f"   ✓ {len(preds)} jogadores previstos")

    if not preds:
        LOGGER.error("Nenhuma previsão gerada")
        sys.exit(1)

    save_excel(preds, odds_by_clube, models, output_path, top_n_resumo=25)

    # ========== RESUMO FINAL E SALVAMENTO ==========
    elapsed_time = time.time() - start_time
    
    if TRAIN_LOGGER:
        TRAIN_LOGGER.log_final_summary(
            backtest_results=bt_results,
            n_predictions=len(preds),
            elapsed_time_s=elapsed_time
        )
        
        # Salva relatórios em CSV para análise posterior
        TRAIN_LOGGER.save_reports(output_dir="logs")
    # ===============================================

    print("\n" + "=" * 100)
    print(" ✅ Concluído")
    print(f" 📦 Modelos salvos em: {model_path}")
    print(f" 📊 Excel: {output_path}")
    print(f" 📊 Logs salvos em: {log_path}")
    print(f" 📈 Métricas CSV: logs/train_metrics.csv")
    print(f" ⏱️ Tempo total: {elapsed_time/60:.1f} min ({elapsed_time:.0f}s)")
    print("=" * 100)


if __name__ == "__main__":
    main()
