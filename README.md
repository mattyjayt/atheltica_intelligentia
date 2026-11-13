# Athletica Intelligentia

Our entry to the **More Than A Hack** possession challenge: label every frame of a football match
(FC Barcelona vs Athletic Club) with who has the ball. The labels are `H` (home), `A` (away) or
`D` (dead ball).

We start from the challenge baseline and add two steps, each motivated by what the data shows:

1. **Dead-ball detection.** Find stoppages from how the players move, not only from whether the
   camera can see the ball.
2. **Gap filling.** When the ball is hidden during play, give it to the team that clearly had it
   before and after.

No machine learning and no training. Every step is a readable rule over the tracking data and the
Smart Tagging events.

## The solution in three files

| file | what it is |
|---|---|
| [`solution/walkthrough.ipynb`](solution/walkthrough.ipynb) | **Start here.** A tutorial-style walkthrough: the problem, the data analysis that led to our approach, how each step works, and the results |
| [`solution/run_inference.py`](solution/run_inference.py) | The implementation. Writes our submissions, scores them with the official scorer, and prints baseline vs ours |
| [`solution/clean.sh`](solution/clean.sh) | Removes everything `run_inference.py` produces |

The challenge's own files (`starter/`, `evaluation/`, `games.py`) are used unchanged.

## Quick start

```bash
uv sync                                   # install dependencies
uv run python solution/run_inference.py   # write submissions, score them, print the comparison
./solution/clean.sh                       # start again from a clean slate
```

The match data is gated and is not part of this repository. See [`DATA.md`](DATA.md) for how to
get it, and unpack it into `data/`.

Outputs:

- `evaluation/results/athletica_intelligentia_smooth_<window>.csv`: one submission per smoothing window
- `evaluation/scoreboard.csv` and `evaluation/reports/*.json`: the official scores
- `evaluation/reports/athletica_intelligentia_config.json`: the exact settings used for the run

## The approach

**The baseline** gives each frame to the team of the player nearest the ball. It says `D` when the
ball is not visible, then smooths with a majority vote. Its weak spot is the dead ball. On the
public ground truth:

- **Two thirds of real dead-ball time has the ball visible.** The ball is out of play or being
  placed for a restart, but still tracked, so the baseline calls it possession.
- **Almost half of the frames where the ball is missing are live play.** The ball is hidden
  behind players or in the air, so the baseline calls it dead.

**Step 1: dead-ball detection.** The clearest sign that play has stopped is that **all the players
slow down together**. Average player speed roughly halves, from about 2.3 to 1.2 m/s. Our rule:

- A stretch where average player speed stays below 1.5 m/s for at least 3 s is a dead ball if it
  is *anchored* by evidence of a stoppage:
  - the ball left play shortly before it: seen outside the pitch, or lost near a line;
  - a Smart Tagging set piece starts inside it or just after.
- It runs from the moment the ball left play to the restart.
- Slower, longer stretches count without an anchor. These are fouls and injuries that Smart
  Tagging does not list.
- These frames are forced to `D` after smoothing.

**Step 2: gap filling.** A missing-ball gap with the same team on both sides is filled with that
team, up to 20 s. Short gaps with different teams on either side are split down the middle. Gaps
are never filled across a dead ball from Step 1.

Every threshold is a named setting at the top of `solution/run_inference.py`. The walkthrough
explains where each one comes from and shows that the results are not sensitive to the exact values.

## How we got here

1. **Measured the baseline's weak spot.** It finds only 21% of dead-ball frames (`recall_D` 0.21),
   and dead ball is a third of macroF1.
2. **Tested its assumption that "ball missing" means "dead ball".** It is wrong both ways (see
   above), so a better dead-ball signal was needed.
3. **Measured candidate signals against the public ground truth.** For each variable we asked how
   well it separates dead frames from live ones (AUC; 0.5 is useless). The two strongest were
   **average player speed** (0.79) and **ball distance to the nearest line** (0.77). A stationary
   ball was useless (0.51).
4. **Looked at what the set pieces mark.** Lining up all 59 set pieces showed that a set piece's
   *start* is the restart: players are slowest exactly then. The ball usually left play 10-30 s
   earlier. That gave the shape of a dead ball: **from the ball leaving play, to the restart**.
5. **Noticed that stoppages without a set piece still show the slowdown.** Two of the four in the
   public window are like this. So player speed became the backbone of Step 1, and exits and set
   pieces became the anchors.
6. **Filled what was left.** With stoppages handled, a remaining missing-ball gap is most likely a
   hidden ball, which led to Step 2.

## Variables

### What we read from the data

| input | source | meaning |
|---|---|---|
| frame | all sources | One video frame, 25 per second. 199,319 frames; only the two halves (158,857 frames) are graded |
| ball position | tracking (ATD) | Where the ball is on the pitch, converted to metres (105.3 × 68 m). Missing when the camera system cannot see it (~41% of graded frames) |
| player position and team | tracking (ATD) | Every tracked player's position and team. Used by the baseline to find the player nearest the ball |
| player speed | tracking (ATD) | How fast each tracked player is moving, in m/s |
| set-piece start | Smart Tagging | When a throw-in, corner, goal kick, free kick or kick-off begins (59 in the match). Converted from seconds to a frame |
| baseline team | `starter/baseline.py` | Per frame: the team of the nearest player when the ball is visible, otherwise *no team* |

### What we compute

**Step 1: dead-ball detection**

| variable | meaning | value | why this value |
|---|---|---|---|
| team speed | Average speed of all tracked players, smoothed over 1 s | | About 1.2 m/s when play is dead, 2.3 m/s when live |
| slowdown | A stretch where team speed stays below a threshold for a minimum time | < 1.5 m/s for ≥ 3 s | Between the dead and live levels; results barely change from 1.2 to 1.8 m/s |
| distance to line | Metres from the ball to the nearest touchline or goal line; negative when outside the pitch | | About 4 m when dead, 13 m when live |
| ball exit | The moment the ball left play: seen more than 0.5 m outside the pitch, or lost within 2 m of a line. Exits less than 5 s apart count as one | 0.5 m / 2 m / 5 s | Small margins absorb tracking noise |
| anchor | Evidence that a slowdown is a real stoppage: an exit up to 15 s before it, or a set-piece start inside it or up to 5 s after | 15 s / 5 s | Exits typically come 10-30 s before the restart |
| restart | The end of the dead ball: set-piece start + 1 s | 1 s | Waiting for players to speed up instead scored worse |
| unanchored slowdown | A slowdown with no anchor that is slower and longer: fouls and injuries | < 1.2 m/s for ≥ 5 s | Stricter, because there is no other evidence; finding these lifts dead-ball recall from 0.52 to 0.84 |
| dead ball | Every frame from the exit (or slowdown start) to the restart (or slowdown end). Forced to `D` after smoothing | | |

**Step 2: gap filling**

| variable | meaning | value | why this value |
|---|---|---|---|
| gap | A run of frames with no team inside one half: the ball is missing and the frame is not a Step 1 dead ball | | |
| team either side | The majority team in the second before and the second after the gap | 1 s | Long enough to ignore a one-frame flicker |
| longest same-team fill | Longest gap filled when the same team is on both sides | 20 s (8 s × 2.5) | Mid-point of the 8-30 s range we compared; most gaps last 5-30 s |
| longest switch fill | Longest gap filled when the teams differ; the gap is split in half between them | 2 s | The hand-over happened inside the gap, so only short gaps can be split safely |

**Final step: smoothing.** The baseline's majority vote over ±*window* frames, for each window from
±12 (0.5 s) to ±200 (8 s). The Step 1 dead-ball frames are then set to `D`.

### Scoring terms

| term | meaning |
|---|---|
| possession | An uninterrupted run of `H` or of `A`. A dead ball ends one possession |
| macroF1 | The F1 score of `H`, `A` and `D` averaged equally: right label on the right frame |
| spellF1 | How well predicted possessions match the true ones in time, averaged over overlap thresholds 0.3-0.7. **The ranking metric** |
| precision / recall | Of the frames we gave a label, the share that were right / of the frames that truly had that label, the share we found |
| AUC | How well one variable separates dead from live frames: 0.5 = no better than chance, near 0 or 1 = clean separation |
| public window | The first five minutes of play (17 possessions), the only ground truth available to participants |

## Results

Scored with the official scorer against the public ground truth (the first five minutes of the
match):

| smoothing window | spellF1 baseline | spellF1 ours | macroF1 baseline | macroF1 ours |
|---|---|---|---|---|
| ±12 | 0.101 | 0.106 | 0.607 | 0.768 |
| ±25 | 0.287 | 0.298 | 0.620 | 0.788 |
| ±50 (published baseline) | 0.326 | **0.446** | 0.621 | **0.811** |
| ±75 | 0.400 | **0.520** | 0.603 | **0.818** |
| ±125 | 0.424 | **0.486** | 0.581 | **0.812** |
| ±200 | 0.193 | 0.288 | 0.515 | 0.789 |

- **Ours is better at every window, on both metrics.** Dead-ball F1 at ±50 rises from 0.32 to 0.78.
- **The public window is only 17 possessions,** so small differences are noise. One missed
  possession moves spellF1 by about 0.06. Treat these numbers as evidence that the method works,
  not as a precise ranking of windows.

## What we tried

Scores are spellF1 / macroF1 on the public five minutes, at the published ±50 smoothing. The
baseline scores 0.326 / 0.621.

| approach | what it means | result | verdict |
|---|---|---|---|
| **Dead-ball detection** (Step 1) | Call play stopped when all players slow down together, backed by the ball leaving play or a set piece | 0.369 / 0.809 | ✅ Kept. Dead-ball recall goes from 0.21 to 0.84, the biggest single gain |
| **Gap filling** (Step 2) | When the ball is hidden, give it to the team that had it just before and just after | 0.379 / 0.614 | ✅ Kept. Fewer, longer possessions. Run after Step 1 so stoppages are not filled |
| **Both together** | Step 1, then Step 2, then the baseline's smoothing | **0.446 / 0.811** | ✅ Our solution |
| Interpolating the ball across short gaps | Draw a straight line between the last and next ball sighting, then use the nearest player as usual | no change | ❌ Recovered 0.1% of missing frames: almost every gap is longer than 4 s |
| "Hold owner" | If nobody is close to the ball (> 3 m), assume it is in flight and keep the last team that was | 0.297 / 0.598 | ❌ Worse. Only helped at ~1.5 m, and that value was tuned on five minutes |
| First version of Step 1 | Dead ball = the ball vanished near a line, or the ball sat still | 0.344 / 0.644 | ❌ Found only 28% of dead-ball time: two thirds of it has the ball visible |
| Stationary ball | Use how long the ball has not moved as a sign of a stoppage | no separation | ❌ During stoppages the ball is carried or rolled back, rarely still |
| Set-piece intervals | Label every frame inside a Smart Tagging set-piece interval as dead ball | not used as `D` | ❌ Intervals are a fixed 35 s and mostly cover live play after the kick; only their *start* (the restart) is useful |
| Corner zone | Treat the ball sitting near a corner flag as a corner kick being set up | no effect | ❌ The tracker sees the ball near a corner flag for ~4 s all match |
| Restart on acceleration | End a dead ball when the players speed up again, instead of at the set-piece start | 0.322 / 0.803 | ❌ Players are still slow after the kick, by which time a team has the ball |

**Ideas not yet tried:**

- **Team shape when the ball is missing.** The team in possession spreads out and the defending
  team compacts.
- **Smart Tagging's `Team` tag as a prior on who has the ball.** It is noisy, and missing on every
  throw-in.
- **A minimum possession length inside the smoothing step.**

## Food for thought

- **We lose a little home-team recall.** At every window, `recall_H` is about 0.02-0.03 below the
  baseline's (0.859 vs 0.880 at ±50).
  - Almost all of the loss is Barcelona possession being called *dead*. Our hypothesis: Barcelona's
    slow, patient build-up looks like a stoppage to a rule built on player speed.
  - Maybe some of those frames really are borderline. Or maybe we need another angle, such as
    whether the ball keeps moving between teammates, to tell slow play from stopped play.
- **How much of the match is dead?** We label about 46% of the full match as dead ball, against
  20% in the public five minutes. Typical professional matches have only 52-55 minutes of real
  playing time, so this is plausible, but it cannot be confirmed without the full ground truth.
