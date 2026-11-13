"""Athletica Intelligentia: baseline possession labels, augmented with dead-ball detection and gap filling.

    uv run python solution/run_inference.py      # from the repository root
    ./solution/clean.sh                          # remove our submissions, reports and scoreboard

1. Reads the tracking with the challenge's own baseline (`starter/baseline.build`) to get the
   nearest-player team per frame.
2. Augments those codes with our two steps (explained in solution/walkthrough.ipynb):
     - dead_ball_v2: players slow down together, anchored by a ball exit / set-piece start -> D
     - gap_fill:     a missing-ball gap with the same team on both sides is filled with that team
3. Smooths at every window in `windows`, forces dead-ball frames to D, and writes one submission
   per window to evaluation/results/.
4. Scores everything with evaluation/run_evaluation.py (scoreboard.csv + reports/*.json) and prints
   baseline vs ours side by side.

Importing this module runs nothing, so the walkthrough notebook uses these exact functions.
"""

import json
import os
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games  # noqa: E402
from starter.baseline import (AWAY, HOME, NOBODY, build, smooth,  # noqa: E402
                              team_of_each_track, write_submission)

# ==================================================================================================
# Configuration -- the shipped solution. Explore new values in solution/walkthrough.ipynb first.
# ==================================================================================================

SUBMISSION_PREFIX = 'athletica_intelligentia_smooth_'
RESULTS_DIR = os.path.join(ROOT, 'evaluation', 'results')
REPORTS_DIR = os.path.join(ROOT, 'evaluation', 'reports')
SCOREBOARD = os.path.join(ROOT, 'evaluation', 'scoreboard.csv')
RUN_EVALUATION = os.path.join(ROOT, 'evaluation', 'run_evaluation.py')

windows = [12, 25, 50, 75, 125, 200]      # smooth() half-widths written and compared

FPS = 25

PIPELINE = {                              # runs in this order; False = skipped
    'dead_ball_v2': True,
    'gap_fill': True,
}

# --- dead-ball features ---------------------------------------------------------------------------
OUT_MARGIN_M = 0.5               # ball seen this far beyond a line = outside the pitch
OUT_OF_PLAY_MARGIN_M = 2.0       # ball last seen this close to a line before vanishing = ball exit
EXIT_MERGE_S = 5                 # exits closer than this are one stoppage
PLAYER_SPEED_SMOOTH_FRAMES = 25

# --- dead_ball_v2 ---------------------------------------------------------------------------------
STOP_PLAYER_SPEED = 1.5          # mean player speed below this = players have stopped playing
STOP_MIN_S = 3.0                 # ... for at least this long
ANCHOR_PRE_S = 15                # a ball exit up to this long before the slowdown anchors it
ANCHOR_POST_S = 5                # a set-piece start up to this long after the slowdown anchors it
RESTART_PAD_S = 1.0              # dead until set-piece start + this
RESTART_AT = 'set_piece'         # 'set_piece' | 'acceleration'
USE_UNANCHORED = True            # also count unanchored slowdowns (fouls, injuries) that are ...
UNANCHORED_PLAYER_SPEED = 1.2    # ... slower than this on average
UNANCHORED_MIN_S = 5.0           # ... and at least this long

# --- gap_fill -------------------------------------------------------------------------------------
BASE_WINDOW_FRAMES = 200
GAP_FILL_MULTIPLIER = 2.5        # longest same-team gap filled = BASE_WINDOW_FRAMES * this (20 s)
CONTEXT_FRAMES = 25              # frames either side of a gap used to decide the team
MAX_SWITCH_GAP_FRAMES = 50       # different teams either side: filled only up to this
SWITCH_MODE = 'split'            # 'split' | 'left' | 'dead'

HACKATHON_METRICS = ['submission', 'macro_f1', 'spell_f1_mean', 'poss']

PERFORMANCE_METRICS = [
    'f1_H', 'f1_A', 'f1_D',
    'precision_H', 'recall_H',
    'precision_A', 'recall_A',
    'precision_D', 'recall_D',
    'alive_dead_f1'
]


def config():
    """Every tunable above as a dict: written next to the reports, and diffed by the notebook."""
    names = ['FPS', 'PIPELINE', 'OUT_MARGIN_M', 'OUT_OF_PLAY_MARGIN_M', 'EXIT_MERGE_S',
             'PLAYER_SPEED_SMOOTH_FRAMES', 'STOP_PLAYER_SPEED', 'STOP_MIN_S', 'ANCHOR_PRE_S',
             'ANCHOR_POST_S', 'RESTART_PAD_S', 'RESTART_AT', 'USE_UNANCHORED',
             'UNANCHORED_PLAYER_SPEED', 'UNANCHORED_MIN_S', 'BASE_WINDOW_FRAMES',
             'GAP_FILL_MULTIPLIER', 'CONTEXT_FRAMES', 'MAX_SWITCH_GAP_FRAMES', 'SWITCH_MODE']
    return {name: globals()[name] for name in names}


# ==================================================================================================
# Inputs beyond the baseline codes
# ==================================================================================================

def load_tracking(raw_path, n_slots, pitch):
    """One pass over the raw ATD -> ball (n, 2) in metres and player speeds (n, slots) in m/s."""
    length, width = pitch
    rows = []
    with open(raw_path) as handle:
        for line in handle:
            rows.append(np.array(line.replace(';', ',').replace(':', ',').split(','),
                                 dtype=np.float32))
    data = np.vstack(rows)
    frame = data[:, 0].astype(np.int64) - 1                 # frame_id counts from 1
    body = np.full((int(frame.max()) + 1, n_slots * 3 + 3), np.nan, dtype=np.float32)
    body[frame] = data[:, 1:]
    players = body[:, :n_slots * 3].reshape(len(body), n_slots, 3)
    ball_xy = body[:, -3:-1] * np.array([length, width], dtype=np.float32)
    return ball_xy, players[:, :, 2]


def load_set_piece_starts(path):
    """Sorted video frames where Smart Tagging starts a SET PIECES instance (= the restart)."""
    starts = []
    for inst in ET.parse(path).getroot().iter('instance'):
        if (inst.findtext('code') or '').strip().upper() == 'SET PIECES':
            starts.append(int(round(float(inst.findtext('start')) * FPS)))
    return np.array(sorted(starts), dtype=np.int64)


def graded_mask(n, blocks):
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    return graded


def runs_of(mask):
    """(starts, ends_exclusive) of every run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[::2], edges[1::2]


def smoothed_player_speed(player_speed):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mean = np.nanmean(player_speed, axis=1)
    return pd.Series(mean).rolling(PLAYER_SPEED_SMOOTH_FRAMES, center=True,
                                   min_periods=1).mean().values


def ball_exits(ball_xy, graded, pitch):
    """Frames where the ball left play: seen outside the pitch, or last seen near a line before a
    gap. Graded frames only, merged when closer than EXIT_MERGE_S."""
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    visible = ~np.isnan(x)
    line_dist = np.minimum.reduce([x, length - x, y, width - y])
    outside = visible & (np.nan_to_num(line_dist, nan=0) < -OUT_MARGIN_M)

    gap_starts = runs_of(~visible)[0]
    before = np.nan_to_num(line_dist[np.maximum(gap_starts - 1, 0)], nan=99)
    gap_exits = gap_starts[(gap_starts > 0) & (before < OUT_OF_PLAY_MARGIN_M)]

    exits = np.unique(np.r_[runs_of(outside)[0], gap_exits])
    exits = exits[graded[exits]]
    return exits[np.r_[True, np.diff(exits) > EXIT_MERGE_S * FPS]]


def find_gaps(codes, blocks, context):
    """Runs of NOBODY inside each graded block: (start, end_exclusive, length, left, right)."""
    def side_team(lo, hi):
        window = codes[lo:hi]
        home, away = int(np.sum(window == HOME)), int(np.sum(window == AWAY))
        return NOBODY if home + away == 0 else (HOME if home >= away else AWAY)

    gaps = []
    for _, first, last in blocks:
        starts, ends = runs_of(codes[first:last + 1] == NOBODY)
        for start, end in zip(starts + first, ends + first):
            gaps.append((start, end, end - start,
                         side_team(max(first, start - context), start),
                         side_team(end, min(last + 1, end + context))))
    return gaps


# ==================================================================================================
# The two steps (solution/walkthrough.ipynb)
# ==================================================================================================

def dead_ball_v2(codes, dead, player_speed, exits, set_piece_starts, graded):
    """Slowdowns of the whole team set, anchored by a ball exit or a set-piece start -> dead."""
    mask = np.zeros(len(codes), dtype=bool)
    speed = np.nan_to_num(player_speed, nan=99)
    for s, e in zip(*runs_of(speed < STOP_PLAYER_SPEED)):
        if e - s < STOP_MIN_S * FPS:
            continue
        ex = exits[(exits >= s - ANCHOR_PRE_S * FPS) & (exits < e)]
        sp = set_piece_starts[(set_piece_starts >= s) & (set_piece_starts <= e + ANCHOR_POST_S * FPS)]
        if len(ex) or len(sp):
            lo = min(s, ex[0]) if len(ex) else s
            if len(sp):
                restart = sp[-1] + int(RESTART_PAD_S * FPS)
                hi = restart if RESTART_AT == 'set_piece' else max(e, restart)
            else:
                hi = e
        elif (USE_UNANCHORED and e - s >= UNANCHORED_MIN_S * FPS
              and speed[s:e].mean() < UNANCHORED_PLAYER_SPEED):
            lo, hi = s, e
        else:
            continue
        mask[lo:hi] = True
    mask &= graded
    codes = codes.copy()
    codes[mask] = NOBODY
    return codes, dead | mask


def gap_fill(codes, dead, blocks):
    """Fill missing-ball gaps from the team on both sides. Dead frames are never filled."""
    max_same = int(round(BASE_WINDOW_FRAMES * GAP_FILL_MULTIPLIER))
    out = codes.copy()
    for start, end, length, left, right in find_gaps(codes, blocks, CONTEXT_FRAMES):
        if left == NOBODY or right == NOBODY or dead[start:end].any():
            continue
        if left == right:
            if length <= max_same:
                out[start:end] = left
        elif length <= MAX_SWITCH_GAP_FRAMES:
            if SWITCH_MODE == 'split':
                middle = start + length // 2
                out[start:middle], out[middle:end] = left, right
            elif SWITCH_MODE == 'left':
                out[start:end] = left
    return out, dead


def augment(codes, ball_xy, player_speed_raw, set_piece_starts, blocks, pitch, pipeline=None):
    """Baseline codes -> (augmented codes, dead mask). The single entry point for our logic."""
    pipeline = PIPELINE if pipeline is None else pipeline
    graded = graded_mask(len(codes), blocks)
    dead = np.zeros(len(codes), dtype=bool)
    if pipeline.get('dead_ball_v2'):
        exits = ball_exits(ball_xy, graded, pitch)
        codes, dead = dead_ball_v2(codes, dead, smoothed_player_speed(player_speed_raw), exits,
                                   set_piece_starts, graded)
    if pipeline.get('gap_fill'):
        codes, dead = gap_fill(codes, dead, blocks)
    return codes, dead


def finalize(codes, dead, window):
    """smooth() as in the baseline, then dead-ball frames forced to D."""
    labels = smooth(codes, window)
    labels[dead] = 'D'
    return labels


# ==================================================================================================
# Display
# ==================================================================================================

def dataframe_to_table(df, title, title_style):
    table = Table(
        title=title,
        title_style=title_style,
        show_header=True,
        header_style="bold",
        border_style=title_style
    )

    table.add_column("Index", style="dim")

    for col in df.columns:
        table.add_column(str(col))

    for idx, row in df.head(5).iterrows():
        table.add_row(
            str(idx),
            *[str(value) for value in row]
        )

    return table


def comparison_table(df_base, df_ours, metrics, title, border_style):
    """Baseline vs ours per window; ours green when better, red when worse at 4 decimals."""
    table = Table(
        title=title,
        title_style="bold white",
        show_header=True,
        header_style="bold",
        border_style=border_style,
        expand=True
    )

    table.add_column("Metric", style="bold white", min_width=18)

    for window in windows:
        table.add_column(f"Baseline {window}", justify="center", style="white")
        table.add_column(f"Ours {window}", justify="center")

    for metric in metrics:

        if metric == "submission":
            continue

        values = [metric]

        for window in windows:

            # Safely retrieve values
            baseline = (
                df_base[metric].get(window, None)
                if window in df_base.index else None
            )

            ours = (
                df_ours[metric].get(window, None)
                if window in df_ours.index else None
            )

            # Format values
            baseline_str = f"{baseline:.4f}" if pd.notna(baseline) else "-"
            ours_str = f"{ours:.4f}" if pd.notna(ours) else "-"

            # Colour our value relative to its baseline. Compare at the
            # displayed precision so "equal" matches what is on screen.
            if pd.notna(baseline) and pd.notna(ours):
                b, o = round(baseline, 4), round(ours, 4)

                if o > b:
                    ours_str = f"[bold green]{ours_str}[/bold green]"

                elif o < b:
                    ours_str = f"[bold red]{ours_str}[/bold red]"

                else:
                    ours_str = f"[grey50]{ours_str}[/grey50]"

            values.extend([baseline_str, ours_str])

        table.add_row(*values)

    return table


# ==================================================================================================
# Run
# ==================================================================================================

def main():
    paths = games.asset_paths(source='tactical')
    for name, path in sorted(paths.items()):
        print('%-14s %-46s %s' % (name, os.path.basename(path),
                                  'ok' if os.path.isfile(path) else 'MISSING'))
    print('\n\n')
    codes = build()  # build(): -> use meta_path

    seen = codes != NOBODY
    print('ball located on %d of %d frames (%.1f%%)' % (seen.sum(),
                                                        len(seen),
                                                        100 * seen.mean()))

    # --------------------------------------------------
    # Our augmentation
    # --------------------------------------------------

    pitch = games.GAMES[games.GAME]["pitch"]
    blocks = games.half_video_blocks(games.GAME)
    n_slots = len(team_of_each_track(paths['atd_meta']))
    print('reading tracking for the augmentation ...')
    ball_xy, player_speed_raw = load_tracking(paths['atd_raw'], n_slots, pitch)
    set_piece_starts = load_set_piece_starts(paths['smart_tagging'])

    aug_codes, dead = augment(codes, ball_xy, player_speed_raw, set_piece_starts, blocks, pitch)
    graded = graded_mask(len(codes), blocks)
    print('pipeline: %s' % ' -> '.join(n for n, on in PIPELINE.items() if on))
    print('  dead-ball frames (forced D): %.1f%% of graded' % (100 * dead[graded].mean()))
    print('  frames with no team before smoothing: %.1f%% of graded (baseline %.1f%%)\n'
          % (100 * np.mean(aug_codes[graded] == NOBODY), 100 * np.mean(codes[graded] == NOBODY)))

    for window in windows:
        labels = finalize(aug_codes, dead, window)
        path = write_submission(
            os.path.join(RESULTS_DIR, f'{SUBMISSION_PREFIX}{window}.csv'),
            labels)

    sys.stdout.flush()  # keep our output ahead of the scorer's when piped
    subprocess.run([sys.executable, RUN_EVALUATION, '--no-detail'], cwd=ROOT)

    # Record exactly what produced these files, next to run_evaluation's reports.
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, 'athletica_intelligentia_config.json'), 'w') as handle:
        json.dump({'windows': windows, 'config': config(),
                   'dead_share_graded': float(dead[graded].mean())}, handle, indent=2)

    # --------------------------------------------------
    # Load scoreboard
    # --------------------------------------------------

    df = pd.read_csv(SCOREBOARD)

    # Extract smoothing window from submission filename
    df["window"] = pd.to_numeric(
        df["submission"].str.extract(
            r"smooth_?(\d+)",
            expand=False
        ),
        errors="coerce"
    )

    # Treat baseline_raw as the ground truth for window 0
    df.loc[
        df["submission"] == "baseline_raw",
        "window"
    ] = 0

    # --------------------------------------------------
    # Separate baseline and our results
    # --------------------------------------------------

    df_gt = df[
        df["submission"].str.contains("baseline", na=False)
    ].copy()

    df_ours = df[
        df["submission"].str.contains(
            "athletica_intelligentia",
            na=False
        )
    ].copy()

    # Keep only the relevant metrics, indexed by smoothing window
    df_gt_hackathon = df_gt[["window"] + HACKATHON_METRICS].set_index("window")
    df_gt_performance = df_gt[["window"] + PERFORMANCE_METRICS].set_index("window")
    df_ours_hackathon = df_ours[["window"] + HACKATHON_METRICS].set_index("window")
    df_ours_performance = df_ours[["window"] + PERFORMANCE_METRICS].set_index("window")

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    console = Console()

    console.print()
    console.print(comparison_table(df_gt_hackathon, df_ours_hackathon, HACKATHON_METRICS,
                                   "HACKATHON COMPARISON", "bright_blue"))
    console.print()

    console.print()
    console.print(comparison_table(df_gt_performance, df_ours_performance, PERFORMANCE_METRICS,
                                   "PERFORMANCE COMPARISON", "green"))
    console.print()


if __name__ == '__main__':
    main()
