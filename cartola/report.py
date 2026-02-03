from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment

from training_logger import POSICOES

from .config import STATUS_ORD, SHEET_BY_POS
from .io import PoissonModel, MatchOdds

if TYPE_CHECKING:
    from .models import PositionModels


@dataclass
class PlayerPrediction:
    atleta_id: int
    apelido: str
    posicao_id: int
    posicao: str
    clube_id: int
    clube: str
    opponent_id: int
    opponent: str
    is_home: bool
    preco: float
    rank_score: float
    predicted_score: float
    pred_p10: float
    pred_p90: float
    interval_width: float
    status: str
    recommendation_score: float
    features: Dict[str, float] = field(default_factory=dict)


def print_backtest(results: Dict[str, Any], k_pos: int = 5):
    print("\n" + "=" * 100)
    print(f" 📊 BACKTEST (POSICIONAL) - TOP {k_pos} POR POSIÇÃO (com lift/percentil)")
    print("=" * 100)

    if not results:
        print("Sem resultados (dados insuficientes).")
        return

    print(f"\n📈 {results.get('n_rodadas', 0)} rodadas | {results.get('total', 0)} previsões")

    overall = results.get("overall_weighted", {})
    by_pos = results.get("by_pos", {})

    header = f"{'Métrica':<30} {'Média':>12} {'Std':>10} {'Min':>10} {'Max':>10}"

    main_keys = [
        f"ndcg@{k_pos}",
        f"hit_rate@{k_pos}",
        f"top{k_pos}_overlap",
        f"top{k_pos}_pct",
        f"lift@{k_pos}",
        f"regret@{k_pos}",
        f"pick_percentile_mean@{k_pos}",
        f"pick_percentile_median@{k_pos}",
        f"pick_percentile_min@{k_pos}",
        f"pct_picks_above_mean@{k_pos}",
        "interval_coverage",
        "mae",
        "rmse",
    ]

    print("\n--- Geral (ponderado por posição) ---")
    print(header)
    print("-" * len(header))
    for k in main_keys:
        if k in overall:
            m = overall[k]
            print(f"{k:<30} {m['mean']:>12.3f} {m['std']:>10.3f} {m['min']:>10.3f} {m['max']:>10.3f}")

    if by_pos:
        print("\n--- Por posição ---")
        for pos_id in sorted(by_pos.keys()):
            pname = POSICOES.get(pos_id, str(pos_id))
            mp = by_pos[pos_id]
            print(f"\n[{pname} | id={pos_id}]")
            print(header)
            print("-" * len(header))
            for k in main_keys:
                if k in mp:
                    m = mp[k]
                    print(f"{k:<30} {m['mean']:>12.3f} {m['std']:>10.3f} {m['min']:>10.3f} {m['max']:>10.3f}")

    if "interval_coverage" in overall:
        cov = overall["interval_coverage"]["mean"]
        if cov < 0.75:
            print(f"\n⚠️ Cobertura p10-p90 baixa ({cov:.1%}). Intervalos estreitos.")
        elif cov > 0.90:
            print(f"\n⚠️ Cobertura p10-p90 alta ({cov:.1%}). Intervalos largos.")


def print_importance(models: PositionModels, top_n: int = 25):
    print("\n" + "=" * 100)
    print(" 🎯 FEATURE IMPORTANCE (gain normalizado) - GLOBAL RANKER")
    print("=" * 100)

    if not models.global_model:
        print("Sem modelo global.")
        return
    imp = models.global_model.feature_importance(top_n)
    if not imp:
        print("Sem importance disponível.")
        return

    print(f"\n{'Feature':<40} {'Importância':>15}")
    print("-" * 55)
    for feat, v in imp:
        bar = "█" * int(v * 120)
        print(f"{feat[:39]:<40} {v:>12.4f}  {bar}")


def print_matches(odds_by_clube: Dict[int, MatchOdds]):
    print("\n" + "=" * 100)
    print(" 📊 PARTIDAS (odds → 1X2 → Poisson xG)")
    print("=" * 100)
    shown = set()
    for m in odds_by_clube.values():
        key = (m.home_id, m.away_id)
        if key in shown:
            continue
        shown.add(key)
        pcs_home = PoissonModel.poisson_prob(m.lambda_away, 0) * 100
        pcs_away = PoissonModel.poisson_prob(m.lambda_home, 0) * 100
        print(f"\n{m.home_team} vs {m.away_team}")
        print(f"   Prob: {m.p_home * 100:.1f}% / {m.p_draw * 100:.1f}% / {m.p_away * 100:.1f}%")
        print(
            f"   xG: {m.lambda_home:.2f} - {m.lambda_away:.2f} | "
            f"P(CS): {m.home_team}={pcs_home:.0f}% / {m.away_team}={pcs_away:.0f}%"
        )


def _autosize_and_format_sheet(ws, df: pd.DataFrame):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for j, col in enumerate(df.columns, start=1):
        max_len = max(len(str(col)), *(len(str(x)) for x in df[col].astype(str).values[:200]))
        ws.column_dimensions[get_column_letter(j)].width = min(max(10, max_len + 2), 45)

    fmt_2 = "0.00"
    fmt_1 = "0.0"
    fmt_pct = "0.0%"

    colmap = {c: i+1 for i, c in enumerate(df.columns)}
    def set_fmt(colname, fmt):
        if colname not in colmap:
            return
        j = colmap[colname]
        for r in range(2, ws.max_row + 1):
            ws.cell(row=r, column=j).number_format = fmt

    for c in ["previsao_pontos", "p10", "p90", "intervalo", "rank_score",
              "team_xG", "opp_xG", "player_mean", "player_last5_mean", "team_avg", "pos_vs_opp_mean"]:
        set_fmt(c, fmt_2)

    for c in ["preco"]:
        set_fmt(c, fmt_1)

    for c in ["p_clean_sheet", "p_team_scores_2plus"]:
        set_fmt(c, fmt_pct)

def safe_int(v, default=0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except (ValueError, TypeError):
        return default

def preds_to_df(preds: List[PlayerPrediction]) -> pd.DataFrame:
    rows = []
    for p in preds:
        rows.append({
            "posicao_id": p.posicao_id,
            "posicao": p.posicao,
            "status": p.status,
            "atleta_id": p.atleta_id,
            "apelido": p.apelido,
            "clube": p.clube,
            "adversario": p.opponent,
            "mandante": p.is_home,
            "previsao_pontos": p.predicted_score,
            "p10": p.pred_p10,
            "p90": p.pred_p90,
            "intervalo": p.interval_width,
            "rank_score": p.rank_score,
            "preco": p.preco,
            "team_xG": p.features.get("team_xG", 0.0),
            "opp_xG": p.features.get("opp_xG", 0.0),
            "p_clean_sheet": p.features.get("p_clean_sheet", 0.0),
            "p_team_scores_2plus": p.features.get("p_team_scores_2plus", 0.0),
            "tendencia": p.features.get("player_trend", 0.0),
            "player_mean": p.features.get("player_mean", 0.0),
            "player_last5_mean": p.features.get("player_last5_mean", 0.0),
            "team_avg": p.features.get("team_avg", 0.0),
            "pos_vs_opp_mean": p.features.get("pos_vs_opp_mean", 0.0),
            "jogos": safe_int(p.features.get("player_games"), 0),
        })

    df = pd.DataFrame(rows)

    df["status_ord"] = df["status"].map(lambda s: STATUS_ORD.get(s, 9)).astype(int)
    df = df.sort_values(
        ["posicao_id", "status_ord", "previsao_pontos"],
        ascending=[True, True, False]
    ).drop(columns=["status_ord"])

    col_order = [
        "posicao_id","posicao","status","apelido","clube","adversario","mandante",
        "previsao_pontos","p10","p90","intervalo","rank_score","preco",
        "team_xG","opp_xG","p_clean_sheet","p_team_scores_2plus","tendencia",
        "player_mean","player_last5_mean","team_avg","pos_vs_opp_mean","jogos","atleta_id"
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    return df


def matches_to_df(odds_by_clube: Dict[int, MatchOdds]) -> pd.DataFrame:
    rows = []
    seen = set()
    for m in odds_by_clube.values():
        key = (m.home_id, m.away_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "casa": m.home_team,
            "fora": m.away_team,
            "odd_casa": m.odd_home,
            "odd_empate": m.odd_draw,
            "odd_fora": m.odd_away,
            "p_casa": m.p_home,
            "p_empate": m.p_draw,
            "p_fora": m.p_away,
            "xG_casa": m.lambda_home,
            "xG_fora": m.lambda_away,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["casa", "fora"])
    return df


def importance_to_df(models: PositionModels, top_n: int = 40) -> pd.DataFrame:
    if not models.global_model:
        return pd.DataFrame(columns=["feature", "importance"])
    imp = models.global_model.feature_importance(top_n=top_n)
    return pd.DataFrame([{"feature": f, "importance": v} for f, v in imp])


def save_excel(
    preds: List[PlayerPrediction],
    odds_by_clube: Dict[int, MatchOdds],
    models: PositionModels,
    path: str,
    top_n_resumo: int = 25,
):
    df_all = preds_to_df(preds)

    resumo_parts = []

    top_score = df_all.sort_values(["previsao_pontos"], ascending=False).head(top_n_resumo).copy()
    top_score.insert(0, "lista", f"TOP {top_n_resumo} - Pontos")
    resumo_parts.append(top_score)

    for pos_id, pos_name in POSICOES.items():
        d = df_all[df_all["posicao_id"] == pos_id].copy()
        if d.empty:
            continue
        t = d.sort_values(["previsao_pontos"], ascending=False).head(top_n_resumo)
        t.insert(0, "lista", f"{pos_name} - TOP {top_n_resumo} (Pontos)")
        resumo_parts.append(t)

    df_resumo = pd.concat(resumo_parts, ignore_index=True) if resumo_parts else pd.DataFrame()

    df_matches = matches_to_df(odds_by_clube)
    df_imp = importance_to_df(models, top_n=40)

    sheet_by_pos = {1:"Goleiros",2:"Laterais",3:"Zagueiros",4:"Meias",5:"Atacantes",6:"Tecnicos"}

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if not df_resumo.empty:
            df_resumo.to_excel(writer, sheet_name="Resumo", index=False)

        df_all.to_excel(writer, sheet_name="Jogadores", index=False)

        for pos_id, name in sheet_by_pos.items():
            d = df_all[df_all["posicao_id"] == pos_id].copy()
            if not d.empty:
                d.to_excel(writer, sheet_name=name, index=False)

        if not df_matches.empty:
            df_matches.to_excel(writer, sheet_name="Partidas", index=False)
        if not df_imp.empty:
            df_imp.to_excel(writer, sheet_name="Importance", index=False)

    from openpyxl import load_workbook
    wb = load_workbook(path)
    for sh in wb.sheetnames:
        ws = wb[sh]
        values = list(ws.values)
        if not values or len(values) < 2:
            continue
        cols = list(values[0])
        data = values[1:]
        df_tmp = pd.DataFrame(data, columns=cols)
        _autosize_and_format_sheet(ws, df_tmp)
    wb.save(path)

