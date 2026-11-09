"""Fit raw GOES magnitude from the four signed ARTEMIS-P1 |B| waveforms."""
from __future__ import annotations

import json
from typing import Any

from pipeline.build_p1_goes_reconstruction import build_source_analysis


def build() -> dict[str, Any]:
    return build_source_analysis(
        source="artemis_p1", output_prefix="p1_scalar_goes_reconstruction", input_kind="scalar",
    )


if __name__ == "__main__":
    result = build()
    print(json.dumps({key: value["targets"]["raw_bt"]["full_fit"]
                      for key, value in result["spacecraft"].items()}, indent=2))
