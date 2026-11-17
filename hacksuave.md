# hacksuave — technical document

More than a Hack 2027, Phase 1: frame-level ball possession.

- **Repository:** https://github.com/RetoBlues/more-than-a-hack-2027
- **Model:** `starter/advanced_model_v3.py`
- **Submitted file:** `hacksuave.csv`, SHA-256 `115b2b4c75091d311dfaf49794f801f0b363de60a8b9a7ac86c34ae491f5a4aa`

## 1. Approach

The model has no learned parameters. For every frame it turns the tracking and the Smart Tagging
into evidence for three states (home in possession, away in possession, dead ball) and then
finds the single most likely sequence of states for each half with a Viterbi decoder.

We chose this over a per-frame classifier followed by smoothing for two reasons.

- **The ranking metric counts possessions, not frames.** A decoder with an explicit cost for
  changing state decides where possessions begin and end directly. Our earlier version (v2, in
  `starter/04_advanced_model.ipynb`) smoothed with a majority window and then erased short spells
  in a second pass; both steps were fixes for the same symptom.
- **The challenge definition says possession stays with the last team while the ball is loose or
  contested.** A sticky state is that rule written as a model: the state only changes when the
  evidence against it lasts long enough.

### Evidence per frame

Each frame gets a penalty against the states it argues against. A frame with no information
penalises nothing, and the decoder then simply carries the current state forward.

| situation | argues against | weight per frame |
|---|---|---|
| ball visible, nearest tracked player is on team T | the other team | 0.276 |
| same, and the ball is not lying still | dead ball | 0.138 |
| ball lost for under 3 s, the 5 nearest players of team T are closer to its last position | the other team | 0.138 |
| ball missing for a stretch of 6 s or longer | both teams | 0.042, or 0.085 if the players are walking |
| within 8 s before to 2 s after the start of a set-piece tag | both teams | 0.064, or 0.127 if the ball is missing or the players are walking |
| ball visible and still, players walking | both teams | 0.102 |
| ball missing in a short gap, players walking | both teams | 0.051 |

Definitions used above:

- **Nearest player:** distances are computed in metres (105.3 m by 68.0 m) so the two pitch axes
  are comparable.
- **Still:** the ball moved less than 0.5 m/s over one second, measured from its positions.
- **Walking:** the median speed of all tracked players, averaged over one second, is below
  1.5 m/s.

### Decoder

Three states with per-frame transition probabilities: stay 0.997, switch team directly 0.001,
enter dead ball 0.002, leave dead ball to either team 0.0015. A stay probability of 0.997 means an
expected dwell of about 13 s, the order of a football possession.

The weights are not free parameters. Each is derived from how long that kind of evidence must
persist before it may change the state. A detour into another state and back costs two
transitions, so evidence worth `cost / (seconds × 25)` per frame pays for the detour after
`seconds`:

| evidence | must persist for | detour cost |
|---|---|---|
| nearest player on the other team | 2 s | 13.81 (two team switches) |
| long ball-less gap with players walking | 6 s | 12.71 (into dead ball and out) |
| tagged restart, with ball missing or players walking | 4 s | 12.71 |
| still ball with players walking | 5 s | 12.71 |

The two halves are decoded separately so no state leaks across the interval. Frames outside the
graded halves are written as `D`; they are not scored.

## 2. Assets: used and dropped

| asset | status | how |
|---|---|---|
| Tactical ATD, raw text | **used** | player positions, speeds and team, ball position; one streaming pass |
| Tactical ATD metadata XML | **used** | which track slots belong to which team |
| Smart Tagging | **used, `SET PIECES` only** | prior on the seconds before each tagged restart |
| Graded-half and kick-off frames from `games.py` and `load_smart_tagging.py` | **used** | decode each half separately; work out which end each team defends |
| Smart Tagging `Team` tag and all phase codes | dropped | phases are not possessions, and throw-ins carry no team |
| The ATD ball `speed` field | dropped | we recompute speed over one second from positions instead |
| Tactical homography, tactical video | not used | only needed for visual debugging |
| Broadcast video, ATD and homography | not used | not synchronised to the tactical timeline |

Three findings about the data shaped the model. All were measured on the tracking and the Smart
Tagging alone, without reading any ground truth.

- **A set-piece tag marks the restart, not the stoppage.** All 53 corner, throw-in, goal-kick and
  free-kick tags are exactly 35 s long, and the 6 kick-off tags are 5 s. Aligned on the start of
  those tags, the ball is visible on 54% of frames twenty seconds earlier, 24% two seconds
  earlier, and 80% eight seconds later. The stoppage is therefore before the tag. Our v2 labelled
  the whole tagged interval (23.6% of the match) as dead ball, which is mostly live play after
  the restart.
- **The ball goes missing in long stretches.** There are 150 ball-less gaps in the graded halves
  with a median length of about 10 s, and over 80% of ball-less frames sit in gaps longer than
  10 s. A "last known ball position" is stale for most of them, so the nearest-group fallback is
  only trusted for the first 3 s of gaps shorter than 6 s. That is 0.8% of graded frames.
- **Long gaps look like stoppages.** The players are walking on 86% of long-gap frames, against
  37% of frames where the ball is visible and moving. In the 8 s before a tagged restart the ball
  is in a long gap 70% of the time and the players are walking 77% of the time; in the 6 to 25 s
  after it those figures are 21% and 38%.

## 3. Training data

None. The model is rule-based and no labelled data from this or any other match was used.

The public 5-minute ground truth was used only as a correctness check on finished versions.
During development one sweep of the decoder's stay probability was scored against it; we
discarded that result and did not use it to set any parameter. The thresholds of the submitted
model were chosen by judgement from the statistics in section 2 and were not fitted. The 2 s
minimum spell is the same value v2 used.

## 4. Possession rules implemented

- **Who has the ball.** While the ball is visible, the team of the nearest tracked player. This
  is evidence, not a decision: a touch, deflection or duel won by the other team only becomes a
  change of possession if the other team stays nearest for about 2 s.
- **Loose or contested ball.** Possession stays with the last team, because the decoder holds its
  state until contrary evidence outlasts the 2 s threshold.
- **Ball not tracked, briefly.** For the first 3 s the 5 players of each team nearest the ball's
  last position vote at half weight. After that there is no evidence and the state is carried.
  Each team's likely goalkeeper (its player nearest its own goal line, if within 16.5 m of it) is
  left out of that vote.
- **Dead ball.** A stoppage is declared when the ball is gone for 6 s or more while the players
  walk, or for about 12 s regardless; when a tagged restart follows within 8 s; or when the ball
  lies still for 5 s while the players walk. A visible, moving ball counts against dead ball.
- **Restarts.** After a stoppage the first possessor is whichever team the evidence names. The
  `Team` tag is not used.
- **Direct turnovers.** A change from one team to the other without a stoppage is allowed, at a
  lower prior than a change through a stoppage.

On the graded frames the submitted file is 32.8% `H`, 21.9% `A` and 45.4% `D`, with 295
possessions of mean 11.8 s and 96 stoppages of median 21 s.

## 5. Results on the public window

Scored with `evaluation/run_evaluation.py` on the 5 public minutes (17 possessions):

| version | macroF1 | spellF1 | possessions |
|---|---|---|---|
| `baseline_smooth50` (published floor) | 0.6207 | 0.3256 | 26 |
| v2, notebook | 0.6678 | 0.3879 | 16 |
| **v3, submitted** | **0.6580** | **0.4375** | 15 |

These numbers do not show that v3 is better than v2. The scorer's bootstrap (1,000 draws) reports
that gaps below about 0.35 spellF1 are within noise on 17 possessions. We chose v3 because the
errors it removes are visible in the data itself, and because only 2 of the 59 set-piece tags
fall inside the public window, so that window barely measures v2's main error.

## 6. Limitations

- **Stoppages with a visible ball are missed.** On the public window v3 finds only 30% of
  dead-ball frames (precision 57%). A ball placed for a free kick or goal kick with players
  nearby looks like possession to this model.
- **Stoppage boundaries are approximate.** A stoppage starts when the ball disappears, not when
  the referee stops play, and spellF1 is sensitive to boundaries a second off.
- **A long tracking loss during live play becomes a false stoppage**, which splits one possession
  into two.
- **Nearest player is not controlled possession.** Passes are not detected; control is only
  approximated by the 2 s persistence rule.
- **The goalkeeper exclusion has no measurable effect.** Switching it off changes none of the
  199,319 labels, because the fallback it belongs to acts on under 1% of graded frames.
- **The thresholds are judgement calls on a single match.** They are not validated on other
  matches, and the full-match score is unknown to us.
- **Set-piece tagging is incomplete and treated uniformly.** Goals and the celebrations after them
  have no special handling.

With more time we would use where the ball leaves the pitch to detect and classify stoppages,
detect placed-ball restarts from player formation, fill ball gaps from the broadcast tracking, and
validate the thresholds on labelled data from other matches.

## 7. Reproduction

Python 3.12.9. The model needs numpy only; pandas is for the validator and scorer.

```bash
pip install -r requirements-submission.txt
# unpack the data bundle into data/ (or set MTAH_DATA_DIR)
python starter/advanced_model_v3.py --name hacksuave
python evaluation/validate_submission.py evaluation/results/hacksuave.csv
```

This writes `evaluation/results/hacksuave.csv` in about 25 s. The model is deterministic: two
runs in the pinned environment produced byte-identical files with the SHA-256 given at the top.
