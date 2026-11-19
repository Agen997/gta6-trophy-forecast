#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (*.in) in MAKEFUN_INPUT_DIR (default /input) and writes
one forecast per case to MAKEFUN_OUTPUT_DIR (default /output): for each trophy
five quantiles of its Exophase earn rate (percent) at the case's target.

Model, per trophy, on the logit scale of the earn rate (the scale the pinball
loss uses; rates clamped to 0.01..99.99 percent before the logit):

  * Exophase refreshes its counts in jumps, so consecutive reads with an
    identical rate are collapsed into a single point first.
  * A recency-weighted straight line of logit(rate) against time is fitted
    (1-day half-life), its slope clamped to a sane range.
  * The median anchors near the last observed level (blended with the fitted
    value there) and adds the trend through a saturating damper, so the
    effective slope shrinks as the horizon grows instead of running away.
  * The spread comes from a per-day volatility estimated from the fit's
    residual steps (shrunk toward a prior), scaled by sqrt(horizon), combined
    with a baseline random-walk term and a fixed refresh-jump term. Short or
    flat series fall back to the last level with a wide default spread.
  * Quantiles are the median shifted by z * sigma on the logit scale.

Deterministic, standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime
from statistics import NormalDist

# --- tuning constants -------------------------------------------------------
RATE_MIN = 0.01            # percent; the page prints 2 decimals
RATE_MAX = 99.99           # percent
HALF_LIFE_DAYS = 1.0       # recency weight half-life for the trend fit
SLOPE_MAX = 2.5            # logit per day, clamp on the fitted slope
GAIN_MAX = 3.0             # logit, cap on the total damped trend move
DAMP_TAU_DAYS = 2.0        # trend damper time constant
ANCHOR_LAST_WEIGHT = 0.65  # weight on the last read vs the fitted value there
BASE_RW_VOL = 0.30         # baseline random-walk vol, logit per sqrt-day
JUMP_VOL = 0.20            # fixed refresh-jump vol, logit
FALLBACK_RW_VOL = 0.45     # random-walk vol for short/flat series
VOL_PRIOR = 0.35           # prior per-day volatility for shrinkage
VOL_PRIOR_PAIRS = 3.0      # pseudo-count of the volatility prior
VOL_DAY_MAX = 1.5          # cap on the estimated per-day volatility
SIGMA_MIN = 0.15
SIGMA_MAX = 3.0
MIN_FIT_POINTS = 3
MIN_FIT_SPAN_DAYS = 0.1

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()


# --- small helpers ----------------------------------------------------------
def parse_ts(text):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(rate):
    return RATE_MIN if rate < RATE_MIN else RATE_MAX if rate > RATE_MAX else rate


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def dedupe(points):
    """Collapse consecutive points with an identical value, keeping the last."""
    out = []
    for t, y in points:
        if out and out[-1][1] == y:
            out[-1] = (t, y)
        else:
            out.append((t, y))
    return out


def wls_line(ts, ys, ws):
    """Weighted least squares y ~ a + b*t. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mt = sum(w * t for t, w in zip(ts, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (t - mt) ** 2 for t, w in zip(ts, ws))
    if sxx <= 0:
        return None
    sxy = sum(w * (t - mt) * (y - my) for t, y, w in zip(ts, ys, ws))
    b = sxy / sxx
    return (my - b * mt, b)


# --- per-trophy forecast ----------------------------------------------------
def forecast_trophy(obs, t_target_days, zs):
    """obs: [(t_days, rate_pct), ...] non-null, time-sorted. Returns quantiles."""
    if not obs:
        median_logit = logit_pct(RATE_MIN)
        sigma = math.sqrt(FALLBACK_RW_VOL ** 2 * 4.5 + JUMP_VOL ** 2)
        return [inv_logit_pct(median_logit + z * sigma) for z in zs]

    pts = dedupe([(t, logit_pct(r)) for t, r in obs])
    t_last, y_last = pts[-1]
    horizon = max(0.0, t_target_days - t_last)
    span = t_last - pts[0][0]

    fit = None
    if len(pts) >= MIN_FIT_POINTS and span >= MIN_FIT_SPAN_DAYS:
        ts = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ws = [0.5 ** ((t_last - t) / HALF_LIFE_DAYS) for t in ts]
        fit = wls_line(ts, ys, ws)

    if fit is None:
        # Too little movement to fit: hold the last level, spread wide.
        median_logit = y_last
        sigma = math.sqrt(FALLBACK_RW_VOL ** 2 * horizon + JUMP_VOL ** 2)
    else:
        a, b = fit
        b = max(-SLOPE_MAX, min(SLOPE_MAX, b))
        anchor = ANCHOR_LAST_WEIGHT * y_last + (1.0 - ANCHOR_LAST_WEIGHT) * (a + b * t_last)
        # Damped trend: the gain saturates at b * DAMP_TAU_DAYS, and the total
        # move is capped so one jumpy fit cannot fling the median across the
        # whole logit range.
        gain = b * DAMP_TAU_DAYS * (1.0 - math.exp(-horizon / DAMP_TAU_DAYS)) if horizon > 0 else 0.0
        gain = max(-GAIN_MAX, min(GAIN_MAX, gain))
        median_logit = anchor + gain

        # Per-day volatility from the fit's residual steps, shrunk to a prior.
        resid = [y - (a + b * t) for t, y in pts]
        num = VOL_PRIOR_PAIRS * VOL_PRIOR ** 2
        den = VOL_PRIOR_PAIRS
        for i in range(len(pts) - 1):
            dt = pts[i + 1][0] - pts[i][0]
            if dt <= 0:
                continue
            w = 0.5 ** ((t_last - pts[i + 1][0]) / HALF_LIFE_DAYS)
            num += w * (resid[i + 1] - resid[i]) ** 2 / dt
            den += w
        vol_day = min(VOL_DAY_MAX, math.sqrt(num / den)) if den > 0 else VOL_PRIOR
        sigma = math.sqrt(vol_day ** 2 * horizon + BASE_RW_VOL ** 2 * horizon + JUMP_VOL ** 2)

    sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
    return [inv_logit_pct(median_logit + z * sigma) for z in zs]


def clean_quantiles(qs):
    """Round to 4 decimals, clamp to [0, 100], enforce non-decreasing."""
    out = []
    prev = 0.0
    for q in qs:
        v = round(max(0.0, min(100.0, q)), 4)
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


# --- case handling ----------------------------------------------------------
def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [_NORM.inv_cdf(float(lv)) for lv in levels]

    if history:
        t0 = parse_ts(history[0]["at"])
    else:
        t0 = parse_ts(case["at"])
    t_target_days = (parse_ts(case["target"]) - t0) / 86400.0

    # Per-trophy observation lists, in the case's trophy order.
    obs = [[] for _ in trophies]
    for read in history:
        t_days = (parse_ts(read["at"]) - t0) / 86400.0
        rates = read.get("rates") or []
        for i in range(min(len(rates), len(obs))):
            r = rates[i]
            if r is not None:
                obs[i].append((t_days, float(r)))

    rows = []
    for i, trophy in enumerate(trophies):
        qs = forecast_trophy(obs[i], t_target_days, zs)
        rows.append({"id": trophy["id"], "q": clean_quantiles(qs)})
    return {"trophies": rows}


def fallback_case(case):
    """Last-resort persistence forecast if anything above goes wrong."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [_NORM.inv_cdf(float(lv)) for lv in levels]
    try:
        horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    except (TypeError, ValueError):
        horizon_days = 0.0
    sigma = max(SIGMA_MIN, 0.5 * math.sqrt(max(horizon_days, 0.0)))
    last_rates = (history[-1].get("rates") if history else None) or []
    rows = []
    for i, trophy in enumerate(trophies):
        r = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else RATE_MIN
        base = logit_pct(float(r))
        rows.append({"id": trophy["id"], "q": clean_quantiles(
            [inv_logit_pct(base + z * sigma) for z in zs])})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        stem = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception as exc:  # never crash a case if a fallback will do
            print(f"{stem}: forecast failed ({exc!r}), using fallback", file=sys.stderr)
            result = fallback_case(case)
        text = json.dumps(result)
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
