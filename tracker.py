"""Historial de predicciones y récord de aciertos.

Reglas:
  * Una predicción se puede actualizar mientras el partido NO haya empezado
    (nuevas lesiones, clima, QB). Al iniciar el partido queda CONGELADA.
  * Si se calcula una semana ya jugada (backtest), esas filas se guardan con
    en_vivo=False para no mezclarlas con el récord real.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as C

HISTORY_DIR = C.ROOT / "history"
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY = HISTORY_DIR / "predicciones.csv"
ET = ZoneInfo("America/New_York")

COLS = ["season", "week", "game_id", "gameday", "gametime", "away", "home", "p_away", "p_home",
        "pick", "margen_local", "p_home_mercado", "spread_line", "qb_away", "qb_home", "factores",
        "modelo", "generado", "en_vivo", "away_score", "home_score", "ganador", "acierto",
        "acierto_mercado"]


TEXT_COLS = ["game_id", "gameday", "gametime", "away", "home", "pick", "qb_away", "qb_home",
             "factores", "modelo", "generado", "ganador"]


def load_history() -> pd.DataFrame:
    if not HISTORY.exists():
        return pd.DataFrame(columns=COLS)
    h = pd.read_csv(HISTORY)
    for c in COLS:
        if c not in h:
            h[c] = np.nan
    for c in TEXT_COLS:
        h[c] = h[c].astype(object)
    return h[COLS]


def _kickoff(row) -> datetime:
    t = str(row.get("gametime") or "13:00")
    t = t if ":" in t else "13:00"
    return datetime.fromisoformat(f"{pd.Timestamp(row['gameday']).date()}T{t}").replace(tzinfo=ET)


def log_predictions(res: pd.DataFrame, bundle: dict, now: datetime | None = None) -> pd.DataFrame:
    """Inserta/actualiza predicciones respetando el congelamiento al kickoff."""
    now = now or datetime.now(ET)
    h = load_history()
    model = f"{bundle.get('name')} | {bundle.get('feature_set', '')}"
    stamp = now.strftime("%Y-%m-%d %H:%M")
    new_rows = []
    for _, r in res.iterrows():
        started = pd.notna(r.get("result")) or _kickoff(r) <= now
        exists = (h["game_id"] == r["game_id"]).any()
        if started and exists:
            continue  # congelada
        row = {c: r.get(c) for c in COLS if c in r}
        row.update({"modelo": model, "generado": stamp, "en_vivo": not started})
        h = h[h["game_id"] != r["game_id"]]
        new_rows.append(row)
    if new_rows:
        h = pd.concat([h, pd.DataFrame(new_rows)], ignore_index=True)
    h = h.sort_values(["season", "week", "gameday", "game_id"])
    h[COLS].to_csv(HISTORY, index=False)
    return h


def update_results(schedules: pd.DataFrame) -> pd.DataFrame:
    """Califica el historial con los marcadores reales de nflverse."""
    h = load_history()
    if h.empty:
        return h
    s = schedules.set_index("game_id")[["away_score", "home_score", "result", "home_team", "away_team"]]
    s = s.reindex(h["game_id"])
    played = s["result"].notna().values
    h.loc[played, "away_score"] = s["away_score"].values[played]
    h.loc[played, "home_score"] = s["home_score"].values[played]
    res = s["result"].values
    winner = np.where(res > 0, s["home_team"].values, np.where(res < 0, s["away_team"].values, "EMPATE"))
    h.loc[played, "ganador"] = winner[played]
    tie = played & (res == 0)
    h["acierto"] = np.where(played & ~tie, (h["pick"] == h["ganador"]).astype(float), np.nan)
    mpick = np.where(h["p_home_mercado"] >= 50, h["home"], h["away"])
    has_mkt = played & ~tie & h["p_home_mercado"].notna().values
    h["acierto_mercado"] = np.where(has_mkt, (mpick == h["ganador"]).astype(float), np.nan)
    h[COLS].to_csv(HISTORY, index=False)
    return h


def summary(h: pd.DataFrame, only_live: bool = True) -> dict:
    g = h[h["acierto"].notna()]
    if only_live:
        g = g[g["en_vivo"].astype(str).str.lower().isin(["true", "1"])]
    if g.empty:
        return {"n": 0}
    y = (g["ganador"] == g["home"]).astype(float)
    p = (g["p_home"] / 100).clip(1e-4, 1 - 1e-4)
    out = {"n": len(g), "aciertos": int(g["acierto"].sum()), "fallos": int(len(g) - g["acierto"].sum()),
           "accuracy": g["acierto"].mean(), "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())}
    m = g[g["acierto_mercado"].notna()]
    if len(m):
        pm = (m["p_home_mercado"] / 100).clip(1e-4, 1 - 1e-4)
        ym = (m["ganador"] == m["home"]).astype(float)
        out |= {"accuracy_mercado": m["acierto_mercado"].mean(),
                "log_loss_mercado": float(-(ym * np.log(pm) + (1 - ym) * np.log(1 - pm)).mean())}
    return out


def by_week(h: pd.DataFrame, only_live: bool = True) -> pd.DataFrame:
    g = h[h["acierto"].notna()]
    if only_live:
        g = g[g["en_vivo"].astype(str).str.lower().isin(["true", "1"])]
    if g.empty:
        return pd.DataFrame()
    w = (g.groupby(["season", "week"]).agg(partidos=("acierto", "size"), aciertos=("acierto", "sum"),
                                            aciertos_mercado=("acierto_mercado", "sum"))
         .reset_index())
    w["accuracy"] = w["aciertos"] / w["partidos"]
    w["accuracy_mercado"] = w["aciertos_mercado"] / w["partidos"]
    w["acumulado"] = w["aciertos"].cumsum() / w["partidos"].cumsum()
    w["acumulado_mercado"] = w["aciertos_mercado"].cumsum() / w["partidos"].cumsum()
    return w
