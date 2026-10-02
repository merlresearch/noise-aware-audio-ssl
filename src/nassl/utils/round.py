# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from decimal import ROUND_HALF_UP, Decimal


def myround(value: float, unit: str = "0.01") -> float:
    return float(
        Decimal(str(value)).quantize(
            Decimal(unit),
            rounding=ROUND_HALF_UP,
        )
    )
