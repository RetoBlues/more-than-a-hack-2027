"""v3: per-frame evidence for {H, A, D}, decoded in one pass with Viterbi.

What changed from v2 (04_advanced_model.ipynb: CoM fallback + SET PIECES override + smooth(50) +
short-spell eraser), and why. Every number below was measured on the tracking and the Smart
Tagging alone -- no ground truth was read to choose any of it.

  1. Dead ball. Every corner / throw-in / goal-kick / free-kick tag is a fixed 35 s window that
     STARTS at about the restart: ball visibility falls from 54% twenty seconds before a tag to
     24% two seconds before it, and is back to 80% seven seconds after. The stoppage is before
     the tag, not inside it, so v2's "whole tag = D" labelled live play as dead. Here the tag is
     a prior on the seconds leading up to it, and the stoppage itself is found from the tracking:
     the ball gone for a long stretch, or lying still, while the players are walking.

  2. Missing ball. 80-90% of ball-less frames sit in gaps longer than 10 s, where a "last known
     ball position" is stale. The top-5 centre-of-mass fallback is therefore only trusted for the
     first seconds of a gap; the likely goalkeeper of each team is left out of it.

  3. Smoothing. One Viterbi decode over {H, A, D} replaces the majority window and the
     short-spell eraser. Per-frame evidence is deliberately weak -- the flicker in the raw signal
     comes in bursts of about a second, not single frames -- so a state only changes when the
     evidence against it outlasts MIN_SPELL_S.

numpy + standard library only, one streaming pass over the raw ATD.

    python starter/advanced_model_v3.py
"""

import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
import baseline  # noqa: E402
import load_smart_tagging  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
OUT_NAME = "my_hybrid_submission_v3.csv"

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# Params
FPS = 25
TOP_N = 5                # players per team in the centre-of-mass fallback
BOX_DEPTH_M = 16.5       # a "goalkeeper" is only dropped if he is this close to his own goal line
FRESH_S = 3.0            # how long a last-known ball position is trusted after the ball vanishes
LONG_GAP_S = 6.0         # a ball-less stretch at least this long is evidence of a stoppage
STILL_SPEED = 0.5        # m/s over one second; below this the ball is "lying still"
WALK_SPEED = 1.5         # m/s; median player speed below this is "everyone is walking"
TAG_PRE_S = 8.0          # a tagged restart says the ball was dead for about this long before it
TAG_POST_S = 2.0         # ...and the tag starts this long before the ball is actually back

# Decoder. Expected dwell is 1 / (1 - stay): about 13 s, the order of a football possession.
STAY = 0.997
SWITCH_TEAM = 0.001      # H <-> A without a stoppage
TO_DEAD = 0.002
FROM_DEAD = 0.0015
MIN_SPELL_S = 2.0        # a contrary burst shorter than this never becomes a possession
MIN_STILL_S = 5.0        # a still ball has to last this long to be a stoppage on its own
MIN_TAG_S = 4.0          # a tagged restart needs this much of its prior window to count alone


def home_goal_near_zero_by_half(raw_path, team_of_track):
    """Does the home team defend the x=0 end? One bool per half, read off the two kickoffs.

    At the instant of a kickoff both teams stand in their own half, so the home team's mean x at
    that frame says which end is theirs. Teams swap at the interval -- measured here rather than
    assumed, and sampled at the real kickoff frames, not at the start of the graded block.
    """
    home_slots = np.flatnonzero(team_of_track == HOME)
    result = []
    for half in ("1st Half", "2nd Half"):
        frame = load_smart_tagging.KICKOFF_VIDEO_FRAMES[half]
        with open(raw_path) as handle:
            line = next(itertools.islice(handle, frame, frame + 1))
        groups = line.split(FIELD_SEP)[1].split(GROUP_SEP)
        xs = [float(groups[s][:groups[s].index(",")]) for s in home_slots
              if groups[s][0] != MISSING]
        result.append(bool(np.mean(xs) < 0.5) if xs else True)
    return result


def parse_atd(raw_path, team_of_track, n_frames, pitch, second_half_start,
              exclude_goalkeeper=True):
    """One streaming pass over the raw ATD. Returns a dict of per-frame arrays.

    nearest      team of the single nearest player; NOBODY where the ball is not visible
    fallback     team whose TOP_N nearest players are, on average, closer to the last-known ball
    ball_present / ball_x / ball_y (metres)
    player_speed median speed of the tracked players, m/s -- low when everyone is walking
    """
    length, width = pitch
    out = {
        "nearest": np.full(n_frames, NOBODY, dtype=np.int8),
        "fallback": np.full(n_frames, NOBODY, dtype=np.int8),
        "ball_present": np.zeros(n_frames, dtype=bool),
        "ball_x": np.full(n_frames, np.nan),
        "ball_y": np.full(n_frames, np.nan),
        "player_speed": np.full(n_frames, np.nan),
    }
    home_slots = np.flatnonzero(team_of_track == HOME)
    away_slots = np.flatnonzero(team_of_track == AWAY)
    home_near_zero = (home_goal_near_zero_by_half(raw_path, team_of_track)
                      if exclude_goalkeeper else None)
    n_slots = len(team_of_track)
    last_bx, last_by = 0.5 * length, 0.5 * width  # pitch centre until the ball is first seen

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1

            xs = np.full(n_slots, np.nan)
            ys = np.full(n_slots, np.nan)
            speeds = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                xs[slot] = float(group[:c1]) * length
                ys[slot] = float(group[c1 + 1:c2]) * width
                speed = group[c2 + 1:]
                if speed[0] != MISSING:
                    speeds.append(float(speed))
            if speeds:
                out["player_speed"][frame] = np.median(speeds)

            ball = parts[2]
            if ball[0] != MISSING:
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                last_bx = float(ball[:c1]) * length
                last_by = float(ball[c1 + 1:c2]) * width
                out["ball_present"][frame] = True
                out["ball_x"][frame], out["ball_y"][frame] = last_bx, last_by

                dist2 = (xs - last_bx) ** 2 + (ys - last_by) ** 2
                if not np.all(np.isnan(dist2)):
                    out["nearest"][frame] = team_of_track[int(np.nanargmin(dist2))]

            if exclude_goalkeeper:
                home_zero = home_near_zero[0 if frame < second_half_start else 1]
                goal_x = {HOME: 0.0 if home_zero else length, AWAY: length if home_zero else 0.0}
            else:
                goal_x = None
            out["fallback"][frame] = _team_of_closer_group(
                xs, ys, last_bx, last_by, home_slots, away_slots, goal_x)
    return out


def _drop_goalkeeper(xs, slots, own_goal_x):
    """slots without the team's likely goalkeeper.

    Nobody carries a role, so the keeper is taken to be the team's player nearest its own goal
    line -- and only if he is inside penalty-area depth, so that when the keeper is not tracked at
    all the deepest defender is not thrown away in his place.
    """
    depth = np.abs(xs[slots] - own_goal_x)
    if np.all(np.isnan(depth)):
        return slots
    deepest = int(np.nanargmin(depth))
    if depth[deepest] > BOX_DEPTH_M:
        return slots
    return np.delete(slots, deepest)


def _team_of_closer_group(xs, ys, bx, by, home_slots, away_slots, goal_x):
    def group_distance(slots, team):
        if goal_x is not None:
            slots = _drop_goalkeeper(xs, slots, goal_x[team])
        gx, gy = xs[slots], ys[slots]
        dist = np.hypot(gx - bx, gy - by)
        valid = ~np.isnan(dist)
        if not valid.any():
            return np.nan
        order = np.argsort(dist[valid])[:TOP_N]
        return math.hypot(np.mean(gx[valid][order]) - bx, np.mean(gy[valid][order]) - by)

    home_d, away_d = group_distance(home_slots, HOME), group_distance(away_slots, AWAY)
    if np.isnan(home_d) and np.isnan(away_d):
        return NOBODY
    if np.isnan(away_d) or home_d < away_d:
        return HOME
    if np.isnan(home_d) or away_d < home_d:
        return AWAY
    return NOBODY


def runs_of(mask):
    """[(start, end)] of every maximal True run, end exclusive."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def centred_mean(values, half_window):
    """Mean over +/- half_window frames, ignoring NaN. NaN where the window holds nothing."""
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def features(parsed, events, n_frames):
    """The per-frame booleans the evidence is built from."""
    present = parsed["ball_present"]

    # Ball speed over one second, from positions: displacement between t-12 and t+12.
    half = FPS // 2
    speed = np.full(n_frames, np.nan)
    dx = parsed["ball_x"][2 * half:] - parsed["ball_x"][:-2 * half]
    dy = parsed["ball_y"][2 * half:] - parsed["ball_y"][:-2 * half]
    speed[half:-half] = np.hypot(dx, dy) / (2 * half / FPS)
    still = present & (speed < STILL_SPEED)

    walking = centred_mean(parsed["player_speed"], half) < WALK_SPEED

    # Ball-less stretches: fresh for the first FRESH_S, "long" if the whole gap is LONG_GAP_S+.
    fresh = np.zeros(n_frames, dtype=bool)
    long_gap = np.zeros(n_frames, dtype=bool)
    for start, end in runs_of(~present):
        if end - start >= LONG_GAP_S * FPS:
            long_gap[start:end] = True
        else:
            fresh[start:min(end, start + int(FRESH_S * FPS))] = True

    # The seconds leading up to each tagged restart. Kick-off tags are 5 s windows around the
    # restart itself; a kick-off after a goal is preceded by the whole celebration, which the
    # long-gap and walking evidence has to find on its own.
    before_restart = np.zeros(n_frames, dtype=bool)
    for event in events:
        if event["code"].upper() != "SET PIECES":
            continue
        start = event["start_frame"]
        lo = max(0, start - int(TAG_PRE_S * FPS))
        hi = min(n_frames, start + int(TAG_POST_S * FPS))
        before_restart[lo:hi] = True

    return {"still": still, "walking": walking, "fresh": fresh, "long_gap": long_gap,
            "before_restart": before_restart, "ball_speed": speed}


def transition_matrix():
    return np.array([
        [STAY, SWITCH_TEAM, TO_DEAD],
        [SWITCH_TEAM, STAY, TO_DEAD],
        [FROM_DEAD, FROM_DEAD, 1.0 - 2 * FROM_DEAD],
    ])


def evidence(parsed, feats, n_frames):
    """Log-likelihood of each frame under H, A and D. Zero means "no opinion".

    Each weight is set by how long that kind of evidence has to persist before it is allowed to
    change the state: a detour through another state and back costs the two transitions, so
    evidence worth `cost / (seconds * FPS)` per frame pays for the detour after `seconds`.
    """
    trans = transition_matrix()
    team_detour = 2 * math.log(STAY / SWITCH_TEAM)
    dead_detour = math.log(STAY / TO_DEAD) + math.log(trans[D, D] / FROM_DEAD)

    w_team = team_detour / (MIN_SPELL_S * FPS)       # ball visible, nearest player
    w_fallback = w_team / 2                          # ball just lost, top-N group -- weaker
    w_still = dead_detour / (MIN_STILL_S * FPS)
    w_gap = dead_detour / (LONG_GAP_S * FPS)
    w_tag = dead_detour / (MIN_TAG_S * FPS)
    w_walk = w_still                                 # walking only ever supports other evidence

    ll = np.zeros((n_frames, 3))
    present = parsed["ball_present"]

    def against(mask, state, weight):
        ll[mask, state] -= weight

    # Who is on the ball. A visible ball near a home player is evidence against A, and -- unless
    # it is lying still -- against D as well.
    for team, other in ((HOME, A), (AWAY, H)):
        seen = present & (parsed["nearest"] == team)
        against(seen, other, w_team)
        against(seen & ~feats["still"], D, w_team / 2)
        lost = feats["fresh"] & (parsed["fallback"] == team)
        against(lost, other, w_fallback)

    # Is the ball in play at all. Each of these counts against both teams.
    walking = feats["walking"]
    for mask, weight in ((feats["still"] & walking, w_still),
                         (feats["long_gap"], w_gap / 2),
                         (feats["long_gap"] & walking, w_gap / 2),
                         (feats["before_restart"], w_tag / 2),
                         (feats["before_restart"] & (walking | ~present), w_tag / 2),
                         (~present & walking & ~feats["long_gap"], w_walk / 2)):
        against(mask, H, weight)
        against(mask, A, weight)
    return ll


def viterbi(ll, log_trans):
    """Most likely state path for per-frame log-likelihoods ll[T, 3]. Pure Python, O(T x 9)."""
    lt = log_trans.tolist()
    rows = ll.tolist()
    delta = rows[0]
    psi = np.zeros((len(rows), 3), dtype=np.int8)
    for t in range(1, len(rows)):
        e = rows[t]
        new = [0.0, 0.0, 0.0]
        for j in range(3):
            best_i, best = 0, delta[0] + lt[0][j]
            for i in (1, 2):
                value = delta[i] + lt[i][j]
                if value > best:
                    best, best_i = value, i
            new[j] = best + e[j]
            psi[t, j] = best_i
        delta = new

    path = np.empty(len(rows), dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(len(rows) - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def decode(parsed, feats, n_frames, game=GAME):
    """Decode each half on its own, so nothing leaks across the interval. D outside the halves."""
    ll = evidence(parsed, feats, n_frames)
    log_trans = np.log(transition_matrix())
    path = np.full(n_frames, D, dtype=np.int8)
    for _, first, last in half_video_blocks(game):
        path[first:last + 1] = viterbi(ll[first:last + 1], log_trans)
    return path


def build(game=GAME, data_dir=None, exclude_goalkeeper=True, verbose=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_of_track = baseline.team_of_each_track(meta_path)
    events = load_smart_tagging.load(game, data_dir)
    second_half_start = half_video_blocks(game)[1][1]

    if verbose:
        print("teams: H = {}, A = {}".format(*baseline.team_names(meta_path)))
        print("reading ATD ...")
    parsed = parse_atd(raw_path, team_of_track, n_frames, GAMES[game]["pitch"],
                       second_half_start, exclude_goalkeeper)
    feats = features(parsed, events, n_frames)
    path = decode(parsed, feats, n_frames, game)

    if verbose:
        graded = np.zeros(n_frames, dtype=bool)
        for _, first, last in half_video_blocks(game):
            graded[first:last + 1] = True
        print("  graded frames: ball visible {:.1f}%  still {:.1f}%  long gap {:.1f}%  "
              "fresh gap {:.1f}%  walking {:.1f}%  before a tagged restart {:.1f}%".format(
                  *(100.0 * np.mean(x[graded]) for x in (
                      parsed["ball_present"], feats["still"], feats["long_gap"], feats["fresh"],
                      feats["walking"], feats["before_restart"]))))
        share = [100.0 * np.mean(path[graded] == s) for s in (H, A, D)]
        print("  decoded, graded frames: H {:.1f}%  A {:.1f}%  D {:.1f}%".format(*share))
    return LABELS[path]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=os.path.splitext(OUT_NAME)[0],
                        help="file name without .csv -- the team name, for the real submission")
    args = parser.parse_args()

    labels = build()
    out_name = args.name + ".csv"
    out_path = baseline.write_submission(os.path.join(RESULTS_DIR, out_name), labels.tolist())
    print("wrote", os.path.relpath(out_path, ROOT))
    print("validate and score:")
    print("  python evaluation/validate_submission.py evaluation/results/" + out_name)
    print("  python evaluation/run_evaluation.py")
