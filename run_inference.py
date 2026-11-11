import games
from starter.baseline import build, smooth, write_submission, NOBODY

import os
import pandas as pd
from rich.console import Console
from rich.table import Table
import sys
import subprocess
sys.path.insert(0, os.path.abspath('..'))

HACKATHON_METRICS = ['submission', 'macro_f1', 'spell_f1_mean', 'poss']

PERFORMANCE_METRICS = [
    'f1_H', 'f1_A', 'f1_D',
    'precision_H', 'recall_H',
    'precision_A', 'recall_A',
    'precision_D', 'recall_D',
    'alive_dead_f1'
]

windows = [12, 25, 50, 75, 125, 200]

for name, path in sorted(games.asset_paths(source='tactical').items()):
    print('%-14s %-46s %s' % (name, os.path.basename(path),
                              'ok' if os.path.isfile(path) else 'MISSING'))
print('\n\n')
codes = build()  # build(): -> use meta_path

seen = codes != NOBODY
print('ball located on %d of %d frames (%.1f%%)' % (seen.sum(),
                                                    len(seen),
                                                    100 * seen.mean()))

for window in windows:
    labels_raw = smooth(codes, window)   # no smoothing yet
    path = write_submission(
        f'evaluation/results/athletica_intelligentia_smooth_{str(window)}.csv',
        labels_raw)

subprocess.run(['uv', 'run', 'python',
                'evaluation/run_evaluation.py', '--no-detail'])


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


# --------------------------------------------------
# Load scoreboard
# --------------------------------------------------

df = pd.read_csv('evaluation/scoreboard.csv')

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

# Keep only the relevant metrics
df_gt_hackathon = df_gt[["window"] + HACKATHON_METRICS]
df_gt_performance = df_gt[["window"] + PERFORMANCE_METRICS]
df_ours_hackathon = df_ours[["window"] + HACKATHON_METRICS]
df_ours_performance = df_ours[["window"] + PERFORMANCE_METRICS]

# Index by smoothing window
df_gt_hackathon = df_gt_hackathon.set_index("window")
df_gt_performance = df_gt_performance.set_index("window")
df_ours_hackathon = df_ours_hackathon.set_index("window")
df_ours_performance = df_ours_performance.set_index("window")


# --------------------------------------------------
# Create Rich comparison table
# --------------------------------------------------

console = Console()

table_hackathon = Table(
    title="HACKATHON COMPARISON",
    title_style="bold white",
    show_header=True,
    header_style="bold",
    border_style="bright_blue",
    expand=True
)

table_hackathon.add_column("Metric", style="bold white", min_width=18)

for window in windows:
    table_hackathon.add_column(
        f"Baseline {window}",
        justify="center",
        style="white"
    )

    table_hackathon.add_column(
        f"Ours {window}",
        justify="center"
    )


# --------------------------------------------------
# Populate comparison table
# --------------------------------------------------

for metric in HACKATHON_METRICS:

    if metric == "submission":
        continue

    values = [metric]

    for window in windows:

        # Safely retrieve values
        baseline = (
            df_gt_hackathon[metric].get(window, None)
            if window in df_gt_hackathon.index else None
        )

        ours = (
            df_ours_hackathon[metric].get(window, None)
            if window in df_ours_hackathon.index else None
        )

        # Format values
        baseline_str = (
            f"{baseline:.4f}"
            if pd.notna(baseline) else "-"
        )

        ours_str = (
            f"{ours:.4f}"
            if pd.notna(ours) else "-"
        )

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

    table_hackathon.add_row(*values)


# --------------------------------------------------
# Display
# --------------------------------------------------

console.print()
console.print(table_hackathon)
console.print()


table_performance = Table(
    title="PERFORMANCE COMPARISON",
    title_style="bold white",
    show_header=True,
    header_style="bold",
    border_style="green",
    expand=True
)

table_performance.add_column("Metric", style="bold white", min_width=18)

for window in windows:
    table_performance.add_column(
        f"Baseline {window}",
        justify="center",
        style="white"
    )

    table_performance.add_column(
        f"Ours {window}",
        justify="center"
    )


# --------------------------------------------------
# Populate comparison table
# --------------------------------------------------

for metric in PERFORMANCE_METRICS:

    if metric == "submission":
        continue

    values = [metric]

    for window in windows:

        # Safely retrieve values
        baseline = (
            df_gt_performance[metric].get(window, None)
            if window in df_gt_performance.index else None
        )

        ours = (
            df_ours_performance[metric].get(window, None)
            if window in df_ours_performance.index else None
        )

        # Format values
        baseline_str = (
            f"{baseline:.4f}"
            if pd.notna(baseline) else "-"
        )

        ours_str = (
            f"{ours:.4f}"
            if pd.notna(ours) else "-"
        )

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

    table_performance.add_row(*values)


# --------------------------------------------------
# Display
# --------------------------------------------------

console.print()
console.print(table_performance)
console.print()
