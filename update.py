"""Actualizador automático: descarga datos, califica el historial y predice la semana actual.

Pensado para ejecutarse programado (cron, Programador de tareas de Windows o GitHub Actions),
así la página siempre muestra predicciones y récord al día sin intervención.

    python update.py                 # semana actual
    python update.py --retrain       # además reentrena el modelo (recomendado 1 vez por semana)
"""
from __future__ import annotations

import argparse
import logging

import config as C
from data_loader import load_all
from predict_week import detect_week, run_week
from tracker import log_predictions, summary, update_results

log = logging.getLogger("update")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=C.CURRENT_SEASON)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-weather-api", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data = load_all(refresh=True, current_season=args.season)
    h = update_results(data.schedules)
    log.info("Historial calificado: %s", summary(h))

    week = detect_week(data.schedules, args.season)
    weeks = data.schedules.loc[data.schedules["season"] == args.season, "week"].unique()
    # semana actual + la siguiente (así el lunes ya aparece la próxima semana en la página)
    for i, w in enumerate([week, week + 1]):
        if w not in weeks:
            continue
        res, bundle = run_week(data, args.season, w, use_weather_api=not args.no_weather_api,
                               retrain=args.retrain and i == 0, status=log.info)
        log_predictions(res, bundle)
        log.info("Semana %s: %d partidos procesados (modelo %s, datos hasta %s)",
                 w, len(res), bundle["name"], bundle["trained_through"])


if __name__ == "__main__":
    main()
