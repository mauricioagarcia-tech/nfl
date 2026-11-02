"""Página web interactiva del predictor NFL.

    streamlit run app.py

Pestañas:
  * Semana  -> predicciones (con ajustes de último minuto: QB fuera, QB que regresa, lesiones)
  * Récord  -> aciertos y fallos acumulados vs. el mercado, semana por semana
  * Modelo  -> configuración y validación del modelo
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config as C
from data_loader import load_all, load_teams
from model import load_model, train
from predict_week import detect_week, manual_injuries, run_week
from tracker import by_week, load_history, log_predictions, summary, update_results

st.set_page_config(page_title="NFL Predictor", page_icon="🏈", layout="wide")

st.markdown("""
<style>
.game {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 16px 10px; margin-bottom:6px;}
.game .meta {font-size:12px; opacity:.65; margin-bottom:6px;}
.game .teams {display:flex; justify-content:space-between; font-weight:600; font-size:17px;}
.game .pct {font-variant-numeric: tabular-nums;}
.game .bar {display:flex; height:10px; border-radius:6px; overflow:hidden; margin:6px 0 8px;}
.game .info {display:flex; flex-wrap:wrap; gap:14px; font-size:13px; opacity:.85;}
.badge {padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600;}
.ok {background:rgba(34,160,90,.18); color:#1f9d55;} .ko {background:rgba(220,60,60,.16); color:#d64545;}
.pend {background:rgba(128,128,128,.15);} .bt {background:rgba(90,120,220,.15); color:#5a78dc;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos (cacheados 1 hora; el botón "Actualizar datos" fuerza la descarga)
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=3600, show_spinner="Descargando datos de nflverse...")
def get_data(season: int, token: int):
    return load_all(refresh=True, current_season=season)


@st.cache_data(ttl=86400)
def get_teams() -> dict:
    t = load_teams()
    return {r.team_abbr: (r.team_name, r.team_color if isinstance(r.team_color, str) else "#888")
            for r in t.itertuples()}


TEAMS = get_teams()
name = lambda t: TEAMS.get(t, (t, "#888"))[0]  # noqa: E731
color = lambda t: TEAMS.get(t, (t, "#888"))[1]  # noqa: E731

if "token" not in st.session_state:
    st.session_state.token = 0

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏈 NFL Predictor")
    season = st.number_input("Temporada", 2023, 2035, C.CURRENT_SEASON)
    data = get_data(season, st.session_state.token)
    hist = update_results(data.schedules)
    current = detect_week(data.schedules, season)
    weeks = sorted(data.schedules.loc[data.schedules["season"] == season, "week"].unique().tolist())
    week = st.selectbox("Semana", weeks, index=weeks.index(current) if current in weeks else 0,
                        format_func=lambda w: f"Semana {w}" + ("  (actual)" if w == current else ""))
    games = data.schedules[(data.schedules["season"] == season) & (data.schedules["week"] == week)]
    week_teams = sorted(set(games["home_team"]) | set(games["away_team"]))

    with st.expander("Ajustes de último minuto"):
        qb_out = st.multiselect("Equipos SIN su QB titular", week_teams)
        qb_start_txt = st.text_area("QB titular forzado (uno por línea)", placeholder="BAL=Lamar Jackson")
        use_wx = st.checkbox("Usar pronóstico del clima", value=True)
        inj_file = st.file_uploader("CSV de lesiones extra", type="csv",
                                    help="Columnas: team, full_name, position, report_status")

    run = st.button("Calcular predicciones", type="primary", width="stretch")
    c1, c2 = st.columns(2)
    if c1.button("Actualizar datos", width="stretch"):
        st.session_state.token += 1
        st.rerun()
    retrain = c2.button("Reentrenar", width="stretch", help="~3 minutos")

    bundle = load_model()
    if bundle:
        st.caption(f"Modelo: **{bundle['name']}** · variables: {bundle.get('feature_set', '')}  \n"
                   f"Datos hasta {bundle['trained_through']}")

# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
if retrain:
    with st.status("Reentrenando con validación anidada...", expanded=True) as s:
        bundle = train(data)
        s.update(label=f"Modelo reentrenado: {bundle['name']}", state="complete")

if run:
    qb_start = dict(l.split("=", 1) for l in qb_start_txt.splitlines() if "=" in l)
    extra = pd.DataFrame()
    if inj_file is not None:
        tmp = C.OUTPUT_DIR / "_lesiones_app.csv"
        tmp.write_bytes(inj_file.getvalue())
        extra = manual_injuries(str(tmp), data.rosters, season, week)
    with st.status("Calculando predicciones...", expanded=True) as s:
        res, bundle = run_week(data, season, week, use_weather_api=use_wx, qb_out=qb_out,
                               qb_start=qb_start, extra_injuries=extra, status=s.write)
        log_predictions(res, bundle)
        hist = update_results(data.schedules)
        s.update(label="Predicciones actualizadas y guardadas en el historial", state="complete")

# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
tab_w, tab_r, tab_m = st.tabs(["Semana", "Récord", "Modelo"])

with tab_w:
    st.subheader(f"Temporada {season} · Semana {week}")
    wk = hist[(hist["season"] == season) & (hist["week"] == week)].copy()
    if wk.empty:
        st.info("Todavía no hay predicciones para esta semana. Pulsa **Calcular predicciones**.")
    else:
        live = wk["en_vivo"].astype(str).str.lower().isin(["true", "1"])
        done = wk["acierto"].notna()
        m1, m2, m3 = st.columns(3)
        m1.metric("Partidos", len(wk))
        if done.any():
            m2.metric("Aciertos esta semana", f"{int(wk.loc[done, 'acierto'].sum())} de {int(done.sum())}")
            m3.metric("Mercado", f"{int(wk.loc[done, 'acierto_mercado'].sum())} de {int(done.sum())}")
        st.caption(f"Última actualización: {wk['generado'].max()} (las predicciones se congelan al iniciar cada partido)")

        cols = st.columns(2)
        for i, r in enumerate(wk.sort_values(["gameday", "gametime"]).itertuples()):
            if pd.notna(r.acierto):
                badge = (f"<span class='badge {'ok' if r.acierto else 'ko'}'>"
                         f"{'✓ Acierto' if r.acierto else '✗ Fallo'} · {int(r.away_score)}-{int(r.home_score)}</span>")
            else:
                badge = "<span class='badge pend'>Pendiente</span>"
            if not live.loc[r.Index]:
                badge += " <span class='badge bt'>backtest</span>"
            mk = ""
            if pd.notna(r.p_home_mercado):
                mfav = r.home if r.p_home_mercado >= 50 else r.away
                mk = f"Vegas: <b>{mfav}</b> {max(r.p_home_mercado, 100 - r.p_home_mercado):.0f}%"
            mg = "" if pd.isna(r.margen_local) else f"Margen: <b>{r.pick}</b> por {abs(r.margen_local):.1f}"
            qbs = (f"QBs: {r.qb_away} vs {r.qb_home}" if isinstance(r.qb_away, str) else "")
            html = f"""
            <div class="game">
              <div class="meta">{pd.Timestamp(r.gameday):%a %d %b} · {r.gametime if isinstance(r.gametime, str) else ''} ET &nbsp; {badge}</div>
              <div class="teams"><span>{name(r.away)}</span><span>{name(r.home)}</span></div>
              <div class="teams pct"><span>{r.p_away:.1f}%</span><span>{r.p_home:.1f}%</span></div>
              <div class="bar"><div style="width:{r.p_away}%;background:{color(r.away)}"></div>
                               <div style="width:{r.p_home}%;background:{color(r.home)}"></div></div>
              <div class="info"><span>Pick: <b>{r.pick}</b></span><span>{mg}</span><span>{mk}</span><span>{qbs}</span></div>
            </div>"""
            with cols[i % 2]:
                st.markdown(html, unsafe_allow_html=True)
                if isinstance(r.factores, str) and r.factores:
                    with st.expander("Factores clave"):
                        st.markdown("\n".join(f"- {f}" for f in r.factores.split(" | ")))
        st.download_button("Descargar CSV de la semana", wk.to_csv(index=False).encode(),
                           f"predicciones_{season}_w{week:02d}.csv", "text/csv")

with tab_r:
    incl_bt = st.toggle("Incluir backtests (semanas calculadas después de jugarse)", value=False)
    sm = summary(hist, only_live=not incl_bt)
    if sm.get("n", 0) == 0:
        st.info("Aún no hay partidos calificados en el historial.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Récord", f"{sm['aciertos']}-{sm['fallos']}")
        b.metric("Accuracy", f"{sm['accuracy']:.1%}",
                 f"{(sm['accuracy'] - sm.get('accuracy_mercado', np.nan)) * 100:+.1f} pts vs. mercado")
        c.metric("Accuracy del mercado", f"{sm.get('accuracy_mercado', np.nan):.1%}")
        d.metric("Log Loss (menor = mejor)", f"{sm['log_loss']:.3f}",
                 f"{sm['log_loss'] - sm.get('log_loss_mercado', np.nan):+.3f} vs. mercado", delta_color="inverse")

        w = by_week(hist, only_live=not incl_bt)
        w["semana"] = w["season"].astype(str) + "-S" + w["week"].astype(str).str.zfill(2)
        long = w.melt(id_vars=["semana"], value_vars=["acumulado", "acumulado_mercado"],
                      var_name="serie", value_name="valor")
        long["serie"] = long["serie"].map({"acumulado": "Modelo (acumulado)",
                                           "acumulado_mercado": "Mercado (acumulado)"})
        bars = alt.Chart(w).mark_bar(opacity=.35).encode(
            x=alt.X("semana:N", title=None), y=alt.Y("accuracy:Q", title="Accuracy",
                                                     axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.3, 1])),
            tooltip=["semana", "aciertos", "partidos", alt.Tooltip("accuracy:Q", format=".1%")])
        lines = alt.Chart(long).mark_line(point=True).encode(
            x="semana:N", y="valor:Q", color=alt.Color("serie:N", title=None, legend=alt.Legend(orient="top")),
            strokeDash=alt.condition(alt.datum.serie == "Mercado (acumulado)", alt.value([4, 3]), alt.value([1, 0])),
            tooltip=["semana", "serie", alt.Tooltip("valor:Q", format=".1%")])
        ref = alt.Chart(pd.DataFrame({"y": [0.65]})).mark_rule(strokeDash=[2, 4], opacity=.5).encode(y="y:Q")
        st.altair_chart((bars + lines + ref).properties(height=320), width="stretch")
        st.caption("Barras: accuracy de cada semana. Línea punteada horizontal: 65%, lo esperado según la validación.")

        g = hist[hist["acierto"].notna()].copy()
        if not incl_bt:
            g = g[g["en_vivo"].astype(str).str.lower().isin(["true", "1"])]
        g["partido"] = g["away"] + " @ " + g["home"]
        g["prob_pick"] = np.where(g["pick"] == g["home"], g["p_home"], g["p_away"])
        g["marcador"] = g["away_score"].astype(int).astype(str) + "-" + g["home_score"].astype(int).astype(str)
        g["modelo_ok"] = np.where(g["acierto"] == 1, "✅", "❌")
        g["mercado_ok"] = np.where(g["acierto_mercado"] == 1, "✅", np.where(g["acierto_mercado"] == 0, "❌", "—"))
        st.dataframe(
            g.sort_values(["season", "week", "gameday"], ascending=False)[
                ["season", "week", "partido", "pick", "prob_pick", "marcador", "ganador", "modelo_ok", "mercado_ok"]],
            hide_index=True, width="stretch",
            column_config={"season": "Temporada", "week": "Semana", "partido": "Partido", "pick": "Pick",
                           "prob_pick": st.column_config.ProgressColumn("Confianza", format="%.1f%%",
                                                                        min_value=0, max_value=100),
                           "marcador": "Marcador (V-L)", "ganador": "Ganador",
                           "modelo_ok": "Modelo", "mercado_ok": "Mercado"})

with tab_m:
    if bundle is None:
        st.info("No hay modelo entrenado. Pulsa **Reentrenar**.")
    else:
        st.markdown(f"**Algoritmo:** `{bundle['name']}`  ·  **variables:** {bundle.get('feature_set')}  ·  "
                    f"**parámetros:** {bundle.get('params')}  \n**Entrenado:** {bundle['trained_at']} con "
                    f"{bundle['n_train']} partidos (hasta {bundle['trained_through']})")
        if "nested" in bundle:
            p = bundle["nested_pooled"]
            st.metric("Validación anidada (estimación honesta)",
                      f"{p['accuracy']:.1%}", f"Log Loss {p['log_loss']:.3f} · AUC {p['roc_auc']:.3f}",
                      delta_color="off")
            st.dataframe(bundle["nested"], hide_index=True, width="stretch")
        st.markdown("**Variables del modelo:** " + ", ".join(f"`{f}`" for f in bundle["features"]))
