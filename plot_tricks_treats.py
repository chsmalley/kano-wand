#!/usr/bin/env python3
"""Plot today's trick and treat counts in 15-minute intervals from 6 PM."""

import argparse
import csv
import logging
import math
from datetime import datetime, time, timedelta, timezone
from pathlib import Path


SPELL_TRICKS = {
    "WINGARDIUM_LEVIOSA": "SPIDER",
    "FLIPENDO": "PINGPONG",
    "LOCOMOTOR": "PINGPONG",
    "EXPELLIARMUS": "BUBBLE",
    "AVIS": "BAT",
    "INCENDIO": "FLAME",
    "LUMOS": "LIGHTS",
    "REDUCTO": "STIR",
    "REDUCIO": "STIR",
    "ENGORGIO": "SPIDER",
    "AGUAMENTI": "POTION",
}
INTERVAL = timedelta(minutes=15)
TRICKS = sorted(set(SPELL_TRICKS.values()))


def parse_timestamp(value):
    timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone()


def read_events(path, start, end):
    """Count mapped tricks and treats in 15-minute bins starting at 6 PM."""
    bins = {
        name: [0] * (math.floor((end - start) / INTERVAL) + 1)
        for name in (*TRICKS, "Treats")
    }

    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                timestamp = parse_timestamp(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                logging.warning("Skipping event with invalid timestamp: %s", row)
                continue

            if not start <= timestamp <= end:
                continue

            event = (row.get("event") or "").strip().lower()
            if event == "treat":
                name = "Treats"
            elif event == "spell":
                name = SPELL_TRICKS.get((row.get("spell") or "").strip())
            elif event == "trick":
                name = (row.get("trick") or "").strip().upper()
                if name not in TRICKS:
                    name = None
            else:
                name = None

            if name is not None:
                index = int((timestamp - start) // INTERVAL)
                bins[name][index] += 1

    return bins


def main():
    parser = argparse.ArgumentParser(
        description="Plot today's tricks and treats in 15-minute bins from 6 PM."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("event_long.csv"),
        help="event CSV path (default: event_long.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("plot_tricks_treats.html"),
        help="HTML file to save the chart to (default: plot_tricks_treats.html)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="save the HTML chart without opening it in a browser",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    now = datetime.now().astimezone()
    start = datetime.combine(now.date(), time(18, 0), tzinfo=now.tzinfo)
    if now < start:
        parser.error("today's 6 PM plotting window has not started yet")

    counts = read_events(args.input, start, now)
    bin_count = len(next(iter(counts.values())))
    labels = [
        (start + index * INTERVAL).strftime("%-I:%M %p")
        for index in range(bin_count)
    ]

    import plotly.graph_objects as go

    fig = go.Figure()
    for name, values in counts.items():
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=values,
            hovertemplate="%{x}<br>%{y} %{fullData.name}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Tricks and treats since 6 PM — {now:%Y-%m-%d}",
        xaxis_title="Local time (15-minute bins)",
        yaxis_title="Events",
        barmode="stack",
        hovermode="x unified",
        xaxis={"tickangle": -45},
        legend_title="Event",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, auto_open=not args.no_open)
    print(f"Saved interactive chart to {args.output}")


if __name__ == "__main__":
    main()
