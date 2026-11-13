#!/bin/sh
# Remove everything solution/run_inference.py produces, so the next run starts clean.
# Baseline submissions in evaluation/results/ are left alone. Works from any directory.
cd "$(dirname "$0")/.." || exit 1

REPORTS="evaluation/reports/"
RESULTS="evaluation/results"
SCORE="evaluation/scoreboard.csv"
TEAM="athletica_intelligentia"

echo "Cleaning $REPORTS..."
if [ -d "$REPORTS" ]; then
  rm -r $REPORTS
fi

echo "Cleaning $RESULTS..."
if [ -d "$RESULTS" ]; then
    for file in "$RESULTS"/*"$TEAM"*.csv; do
        [ -f "$file" ] || continue
        rm -- "$file"
    done
fi

echo "Cleaning $SCORE"
if [ -f "$SCORE" ]; then
  rm $SCORE
fi
