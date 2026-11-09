"""Independently fit raw GOES magnitude from four signed SOLAR-1 |B| waveforms.

Its own measured shock-to-response shift is used. No P1 model is transferred.
"""
from __future__ import annotations

import json
from typing import Any

from pipeline.build_p1_goes_reconstruction import build_source_analysis


def build() -> dict[str, Any]:
    return build_source_analysis(
        source="swfo", output_prefix="solar1_goes_reconstruction", input_kind="scalar",
    )


if __name__ == "__main__":
    result = build()
    print(json.dumps({key: value["targets"]["raw_bt"]["full_fit"]
                      for key, value in result["spacecraft"].items()}, indent=2))
