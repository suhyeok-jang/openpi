#!/usr/bin/env python3
"""
output_seed_fixed/openpi_gr1 하위의 모든 eval_tag / ckpt_run / ckpt_step 조합에 대해
24개 task의 success rate를 파싱하여 summary를 출력합니다.

디렉토리 구조 (eval_robocasa_gr1_sweep.sh가 만드는 layout):
  openpi_gr1/
    <eval_tag>/
      <ckpt_run>/                                       # e.g. gr1_v0
        <ckpt_step>/                                    # e.g. 56000, 59999
          gr1_unified_<TaskName>_GR1ArmsAndWaistFourierHands_Env/
            simulation_results.csv  (우선)
            rollout.log             (fallback)

데이터 소스 우선순위 (GR00T summary.py와 동일):
  1. simulation_results.csv (CSV의 success 컬럼)
  2. rollout.log 의 "Task ... done: X/Y success (Z%)" 줄

GR00T 버전과의 차이점은 디렉토리 layout 1개뿐:
  GR00T:  gr1/<eval_tag>/<ckpt_name>/<step>/gr1_unified/<task_name>/
  openpi: openpi_gr1/<eval_tag>/<ckpt_run>/<step>/gr1_unified_<task_name>/
"""

import csv
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / "openpi_gr1"

# Same 24 tasks as GR00T's summary.py — keep the order verbatim so the printed
# table aligns one-to-one with the GR00T summary output.
EXPECTED_TASKS = [
    "PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
]

# Match GR00T summary.py groupings:
#   PNP = the 18 PosttrainPnPNovel tasks  (out-of-distribution / generalization)
#   ART = the 6 PnP tasks                 (in-distribution)
PNP_TASKS = [t for t in EXPECTED_TASKS if t.startswith("Posttrain")]
ART_TASKS = [t for t in EXPECTED_TASKS if t not in PNP_TASKS]

TASK_DIR_PREFIX = "gr1_unified_"
EXPECTED_EPISODES = 50
N_ENVS = 5
EPISODES_PER_ENV = EXPECTED_EPISODES // N_ENVS  # 10


def parse_task_result(task_dir: Path):
    """CSV 우선, 없으면 rollout.log에서 success rate 파싱.

    Strict: env 5개 × episode 10개 = 50. 초과분은 잘라냄 (per-env 첫 10개만).
    Returns (success_rate, n_episodes_counted) or (None, 0) if no data.
    """
    csv_path = task_dir / "simulation_results.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None, 0

            # group by env_idx, then take first EPISODES_PER_ENV per env
            env_episodes: dict[int, list] = {}
            for r in rows:
                env_idx = int(r["env_idx"])
                env_episodes.setdefault(env_idx, []).append(r)

            # ensure deterministic order within each env (lowest episode_idx first)
            for eps in env_episodes.values():
                eps.sort(key=lambda x: int(x["episode_idx"]))

            trimmed = []
            for env_idx in range(N_ENVS):
                eps = env_episodes.get(env_idx, [])
                trimmed.extend(eps[:EPISODES_PER_ENV])

            if not trimmed:
                return None, 0

            successes = [bool(int(r["success"])) for r in trimmed]
            return sum(successes) / len(successes), len(successes)
        except Exception:
            return None, 0

    # fallback: rollout.log -- main.py prints "Task ... done: X/Y success (Z%)"
    log_path = task_dir / "rollout.log"
    if log_path.exists():
        try:
            text = log_path.read_text()
            m = re.search(r"Task .* done:\s+(\d+)/(\d+) success", text)
            if m:
                ok, total = int(m.group(1)), int(m.group(2))
                return ok / total if total else 0.0, total
        except Exception:
            pass

    return None, 0


def process_ckpt_step(ckpt_step_dir: Path) -> dict:
    """One ckpt-step directory → 24 task results."""
    results = {}
    for task_name in EXPECTED_TASKS:
        task_dir = ckpt_step_dir / f"{TASK_DIR_PREFIX}{task_name}"
        if task_dir.exists():
            sr, n_eps = parse_task_result(task_dir)
            if sr is not None:
                results[task_name] = (sr, n_eps)
    return results


def main():
    if not BASE_DIR.exists():
        print(f"[error] 디렉토리 없음: {BASE_DIR}")
        return

    # 자동 탐색: eval_tag / ckpt_run / ckpt_step
    entries = []
    for eval_tag_dir in sorted(BASE_DIR.iterdir()):
        if not eval_tag_dir.is_dir():
            continue
        for ckpt_run_dir in sorted(eval_tag_dir.iterdir()):
            if not ckpt_run_dir.is_dir():
                continue
            # Heuristic: a "ckpt_run" directory contains numeric step subdirs;
            # otherwise treat the eval_tag dir itself as containing step subdirs
            # (the smoke-test single-job sbatch writes layout
            #  output_seed_fixed/openpi_gr1/<tag>/<step>/<task>/).
            substeps = sorted(ckpt_run_dir.iterdir())
            looks_like_steps = all(p.is_dir() and p.name.isdigit() for p in substeps if p.is_dir())
            if looks_like_steps:
                ckpt_run = ckpt_run_dir.name
                for step_dir in substeps:
                    if step_dir.is_dir():
                        entries.append(
                            (eval_tag_dir.name, ckpt_run, step_dir.name, step_dir)
                        )
            else:
                # Older layout (smoke tests): <tag>/<step>/<task>/
                # Re-interpret ckpt_run_dir as the step dir, with empty ckpt_run.
                if ckpt_run_dir.name.isdigit():
                    entries.append(
                        (eval_tag_dir.name, "", ckpt_run_dir.name, ckpt_run_dir)
                    )
                # else: skip (unrecognized layout)

    if not entries:
        print("[warn] 처리할 결과가 없습니다")
        return

    print(
        f"{'EVAL_TAG':<20} {'CKPT_RUN':<20} {'STEP':<8} "
        f"{'ALL':>6} {'ART':>6} {'PNP':>6} {'DONE':>7}"
    )
    print("-" * 90)

    REQUIRED_TOTAL = EXPECTED_EPISODES * len(EXPECTED_TASKS)  # 50 * 24 = 1200

    for eval_tag, ckpt_run, ckpt_step, step_dir in entries:
        results = process_ckpt_step(step_dir)

        total_episodes = sum(n for _, n in results.values())
        missing = [t for t in EXPECTED_TASKS if t not in results]
        incomplete = {t: n for t, (_, n) in results.items() if n != EXPECTED_EPISODES}

        is_complete = (
            len(missing) == 0
            and len(incomplete) == 0
            and total_episodes == REQUIRED_TOTAL
        )

        art_rates = [results[t][0] for t in ART_TASKS if t in results]
        pnp_rates = [results[t][0] for t in PNP_TASKS if t in results]
        all_rates = [results[t][0] for t in EXPECTED_TASKS if t in results]

        avg_all = sum(all_rates) / len(all_rates) * 100 if all_rates else 0
        avg_art = sum(art_rates) / len(art_rates) * 100 if art_rates else 0
        avg_pnp = sum(pnp_rates) / len(pnp_rates) * 100 if pnp_rates else 0

        warn = "" if is_complete else f"  ⚠ {total_episodes}/{REQUIRED_TOTAL}"
        done = f"{len(results)}/{len(EXPECTED_TASKS)}"
        print(
            f"{eval_tag:<20} {ckpt_run:<20} {ckpt_step:<8} "
            f"{avg_all:>5.1f}% {avg_art:>5.1f}% {avg_pnp:>5.1f}% {done:>7}{warn}"
        )

        # Per-(eval_tag, ckpt_run, ckpt_step) summary file (overwrites the
        # per-task summary.txt that main.py wrote during the run).
        summary_path = step_dir / "summary.txt"
        with summary_path.open("w") as f:
            f.write(f"eval_tag: {eval_tag}\n")
            f.write(f"ckpt_run: {ckpt_run}\n")
            f.write(f"ckpt_step: {ckpt_step}\n")
            f.write(f"complete: {is_complete}\n")
            f.write(f"total_episodes: {total_episodes}/{REQUIRED_TOTAL}\n")
            f.write(f"tasks_completed: {len(results)}/{len(EXPECTED_TASKS)}\n")
            f.write(f"avg_success_rate: {avg_all:.2f}%\n")
            f.write(
                f"art_success_rate: {avg_art:.2f}% ({len(art_rates)}/{len(ART_TASKS)})\n"
            )
            f.write(
                f"pnp_success_rate: {avg_pnp:.2f}% ({len(pnp_rates)}/{len(PNP_TASKS)})\n"
            )
            f.write(f"\n{'TASK':<75} {'SR':>8} {'EPISODES':>10}\n")
            f.write("-" * 95 + "\n")
            for task_name in EXPECTED_TASKS:
                if task_name in results:
                    sr, n_eps = results[task_name]
                    flag = "" if n_eps == EXPECTED_EPISODES else " *"
                    f.write(
                        f"{task_name:<75} {sr * 100:>7.1f}% {n_eps:>10}{flag}\n"
                    )
                else:
                    f.write(f"{task_name:<75} {'N/A':>8} {'N/A':>10}\n")

    print("-" * 90)
    print(
        f"총 {len(entries)}개 (eval_tag, ckpt_run, ckpt_step) 조합 처리 완료. "
        f"최종 결과는 24 tasks × {EXPECTED_EPISODES} episodes = {REQUIRED_TOTAL} 완료 시에만 ⚠ 표시 사라집니다."
    )


if __name__ == "__main__":
    main()
