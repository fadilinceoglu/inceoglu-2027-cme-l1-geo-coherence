"""Render diagnostic overviews from this run's prepared records and boundaries."""
import pandas as pd
from config import study
from config.paths import PROCESSED_DIR, DIAGNOSTIC_DIR
from figures.overview import build_registered_overview
from ingest.missions import TEMPERATURE_UNITS


def build():
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    for craft, info in study.SATELLITES.items():
        frame = pd.read_csv(PROCESSED_DIR / f"{craft}_overview_1min.csv", index_col="time", parse_dates=["time"])
        frame.index = pd.to_datetime(frame.index, utc=True)
        name = "solar1" if craft == "swfo" else craft
        print(build_registered_overview(craft, frame,
              out_path=DIAGNOSTIC_DIR / f"overview_{name}.pdf", frame_label=info["frame"],
              t_unit="n/a" if craft.startswith("goes") else TEMPERATURE_UNITS[craft]))


if __name__ == "__main__":
    build()
