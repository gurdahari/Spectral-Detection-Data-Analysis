#!/usr/bin/env python3
"""Analysis pipeline for controlled and uncontrolled RF spectral detections.

This script reproduces the exploratory analysis and trigger-design workflow
used in the accompanying report. By default it loads CSV files from ./data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# seaborn is optional (used for boxplots)
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

CONTROLLED_FILE = "controlled.csv"
UNCONTROLLED_FILES = {
    "config_1": "uncontrolled_detections_export_config_1.csv",
    "config_2": "uncontrolled_detections_export_config_2.csv",
}

# Task-1 working point
GAMMA_WORK = 0.90

# Task-2 Neyman–Pearson constraint
ALPHA_BUDGET = 0.05
THRESH_GRID = np.linspace(0.50, 0.99, 50)

# Task-2 Trigger gates
TRIGGER_MIN_HITS = 2
TRIGGER_MIN_DUR_SEC = 1.0
TRIGGER_MIN_SINR_DB = 3.0      # set None to disable
TRIGGER_MIN_POWER_DBM = None   # set e.g. -95 to enable

# Runtime plot behavior
SHOW_PLOTS = True
SAVE_PLOTS_DIR: Path | None = None


def _finalize_plot(filename: str | None = None) -> None:
    if SAVE_PLOTS_DIR is not None and filename:
        SAVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(SAVE_PLOTS_DIR / filename, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =========================================================
# I/O
# =========================================================
def read_csv(path_or_dir: str | Path, filename: str) -> pd.DataFrame:
    base = Path(path_or_dir)
    file_path = Path(filename)
    path = file_path if file_path.is_file() else base / filename
    if not path.is_file():
        raise FileNotFoundError(f"Could not find file: {path}")
    return pd.read_csv(path)


# =========================================================
# PLOTTING HELPERS
# =========================================================
def plot_hist(series, bins=100, title="", xlabel="", ylabel="Density", filename=None):
    series = series.dropna()
    plt.figure()
    plt.hist(series, bins=bins, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _finalize_plot(filename)


def plot_line(x, y, title="", xlabel="", ylabel="", marker="o", filename=None):
    plt.figure()
    plt.plot(x, y, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _finalize_plot(filename)


def plot_binned_fpr(tbl, title="", xlabel="Bin midpoint", ylabel="FPR", filename=None):
    """tbl must include 'bin' (pandas Interval) and 'FPR'."""
    if tbl is None or tbl.empty:
        print("[plot_binned_fpr] Empty table, skipping plot.")
        return
    mids = [b.mid for b in tbl["bin"]]
    plt.figure()
    plt.plot(mids, tbl["FPR"], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _finalize_plot(filename)


# =========================================================
# PREP
# =========================================================
def prepare_uncontrolled(df):
    """Add derived fields: times, duration, center_freq, bandwidth."""
    df = df.copy()

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    if "start_time" in df.columns and "end_time" in df.columns:
        df["duration_sec"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    else:
        df["duration_sec"] = np.nan

    if "frequency_high" in df.columns and "frequency_low" in df.columns:
        df["center_freq"] = (df["frequency_high"] + df["frequency_low"]) / 2.0
        df["bandwidth"] = df["frequency_high"] - df["frequency_low"]
    else:
        df["center_freq"] = np.nan
        df["bandwidth"] = np.nan

    return df


# =========================================================
# TASK 1 — CONTROLLED
# =========================================================
def controlled_apply_gamma(df_controlled, gamma):
    df = df_controlled.copy()
    df["pred_final"] = np.where(df["pred scores"] >= gamma, df["pred labels"], 0)
    return df


def miss_detection_analysis_controlled(df_controlled):
    misses = df_controlled[(df_controlled["GT labels"] != 0) & (df_controlled["pred labels"] == 0)]
    miss_counts = misses["GT labels"].value_counts()
    total_gt_counts = df_controlled[df_controlled["GT labels"] != 0]["GT labels"].value_counts()
    miss_rate = (miss_counts / total_gt_counts) * 100

    print("\n--- Miss Detections Analysis (Controlled) ---")
    if miss_rate.empty:
        print("No misses found (or GT labels missing).")
        return

    for cat, rate in miss_rate.items():
        if not np.isnan(rate):
            print(f"Class {cat}: Miss Rate = {rate:.2f}%")


def pr_table_for_rbw(df_controlled, rbw0, gamma):
    df_cfg = df_controlled[df_controlled["rbw"] == rbw0].copy()
    df_cfg = controlled_apply_gamma(df_cfg, gamma)

    classes = sorted([c for c in df_cfg["GT labels"].dropna().unique() if c != 0])
    rows = []

    for target in classes:
        gt_is_target = df_cfg["GT labels"] == target
        pred_is_target = df_cfg["pred_final"] == target

        TP = (gt_is_target & pred_is_target).sum()
        FN = (gt_is_target & ~pred_is_target).sum()
        FP = (~gt_is_target & pred_is_target).sum()

        recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan

        rows.append({
            "class": int(target),
            "TP": int(TP),
            "FN": int(FN),
            "FP": int(FP),
            "recall": recall,
            "precision": precision,
        })

    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def task1_controlled(df_controlled):
    print("\n" + "=" * 80)
    print("TASK 1 — CONTROLLED DATASET")
    print("=" * 80)

    print("Shape:", df_controlled.shape)
    print("Columns:", list(df_controlled.columns))
    print("Total detections:", df_controlled.shape[0])

    if "GT labels" in df_controlled.columns:
        print("Unique GT labels (including 0):", df_controlled["GT labels"].nunique())
        print(
            "Unique signals in controlled (excluding 0):",
            df_controlled[df_controlled["GT labels"] != 0]["GT labels"].nunique(),
        )

    if "pred scores" in df_controlled.columns:
        plot_hist(
            df_controlled["pred scores"],
            bins=300,
            title="PDF of Detection Scores (Controlled)",
            xlabel="pred scores",
            filename="controlled_score_hist.png",
        )

    if "GT labels" in df_controlled.columns and "pred labels" in df_controlled.columns:
        miss_detection_analysis_controlled(df_controlled)

    if "rbw" in df_controlled.columns:
        for rbw0 in sorted(df_controlled["rbw"].dropna().unique()):
            tbl = pr_table_for_rbw(df_controlled, rbw0, GAMMA_WORK)
            print(f"\n--- Precision/Recall per class (rbw={rbw0}, gamma={GAMMA_WORK}) ---")
            print(tbl)


# =========================================================
# TASK 1 — UNCONTROLLED
# =========================================================
def fpr_curve_uncontrolled(df_uncontrolled, thresholds, score_col="score", category_col="category", noise_label=-1):
    is_noise = df_uncontrolled[category_col] == noise_label

    fprs = []
    for g in thresholds:
        accept = df_uncontrolled[score_col] > g
        reject = ~accept

        FP = (is_noise & accept).sum()
        TN = (is_noise & reject).sum()
        fpr = FP / (FP + TN) if (FP + TN) > 0 else np.nan
        fprs.append(fpr)

    return pd.DataFrame({"gamma": thresholds, "FPR": fprs})


def fpr_by_param_binned(df, param, gamma=0.9, bins=8, score_col="score", category_col="category", noise_label=-1):
    df = df.dropna(subset=[param, score_col, category_col]).copy()
    df["bin"] = pd.qcut(df[param], q=bins, duplicates="drop")

    rows = []
    for b in sorted(df["bin"].unique()):
        sub = df[df["bin"] == b]
        noise = sub[sub[category_col] == noise_label]
        FP = (noise[score_col] > gamma).sum()
        TN = (noise[score_col] <= gamma).sum()
        fpr = FP / (FP + TN) if (FP + TN) > 0 else np.nan

        rows.append({
            "bin": b,
            "N_noise": int(len(noise)),
            "FP": int(FP),
            "TN": int(TN),
            "FPR": fpr,
        })

    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def pr_table_uncontrolled_proxy(df_uncontrolled, gamma, score_col="score", category_col="category", noise_label=-1):
    """Compute a relative-recall proxy per class without full ground truth."""
    df = df_uncontrolled.copy()
    df["pred_final"] = np.where(df[score_col] >= gamma, df[category_col], noise_label)

    classes = sorted([c for c in df[category_col].dropna().unique() if c != noise_label])
    rows = []

    for c in classes:
        original_is_c = df[category_col] == c
        survived = (original_is_c & (df["pred_final"] == c)).sum()
        filtered = (original_is_c & (df["pred_final"] == noise_label)).sum()

        rel_recall = survived / (survived + filtered) if (survived + filtered) > 0 else np.nan
        rows.append({
            "class": int(c),
            "Pseudo_TP (Survived)": int(survived),
            "Pseudo_FN (Filtered)": int(filtered),
            "RelativeRecall": rel_recall,
        })

    noise_events = (df[category_col] == noise_label).sum()
    false_alarms = ((df[category_col] == noise_label) & (df[score_col] >= gamma)).sum()
    est_fpr = false_alarms / noise_events if noise_events > 0 else np.nan

    print(f"\n--- Proxy Trigger Eval (event-level) gamma={gamma} ---")
    print(f"False alarms (noise events passing): {false_alarms}")
    print(f"Estimated FPR (noise events): {est_fpr:.4f}")

    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def task1_uncontrolled(cfg_name, df_uncontrolled):
    print("\n" + "=" * 80)
    print(f"TASK 1 — UNCONTROLLED ANALYSIS: {cfg_name}")
    print("=" * 80)

    if "duration_sec" in df_uncontrolled.columns:
        print(f"Average duration: {df_uncontrolled['duration_sec'].mean():.2f} sec")

    if "signal_power" in df_uncontrolled.columns:
        sp = df_uncontrolled["signal_power"]
        print(f"Mean signal_power: {sp.mean():.2f} | Median: {sp.median():.2f}")
        plot_hist(sp, bins=300, title=f"PDF of Signal Power ({cfg_name})", xlabel="Power [dBm]", filename=f"{cfg_name}_signal_power_hist.png")

    if "sinr" in df_uncontrolled.columns:
        snr = df_uncontrolled["sinr"]
        print(f"Mean SINR: {snr.mean():.2f} | Median: {snr.median():.2f}")
        plot_hist(snr, bins=100, title=f"PDF of SINR ({cfg_name})", xlabel="SINR [dB]", filename=f"{cfg_name}_sinr_hist.png")

    if "center_freq" in df_uncontrolled.columns:
        plot_hist(df_uncontrolled["center_freq"], bins=300, title=f"PDF of Center Frequency ({cfg_name})", xlabel="Center Frequency [Hz]", filename=f"{cfg_name}_center_freq_hist.png")

    if "bandwidth" in df_uncontrolled.columns:
        plot_hist(df_uncontrolled["bandwidth"], bins=300, title=f"PDF of Bandwidth ({cfg_name})", xlabel="Bandwidth [Hz]", filename=f"{cfg_name}_bandwidth_hist.png")

    if "start_time" in df_uncontrolled.columns:
        df_time = df_uncontrolled.dropna(subset=["start_time"]).copy()
        if not df_time.empty:
            time_counts = df_time.set_index("start_time").resample("1min").size()
            plt.figure(figsize=(10, 5))
            plt.bar(time_counts.index, time_counts.values, width=0.0005)
            plt.title(f"Temporal Coverage Pattern λ(t) (detections/min) - {cfg_name}")
            plt.xlabel("Time")
            plt.ylabel("Detections per minute")
            plt.xticks(rotation=45)
            plt.tight_layout()
            _finalize_plot(f"{cfg_name}_temporal_coverage.png")

    if "category" in df_uncontrolled.columns:
        print("\nClass distribution (category):")
        print(df_uncontrolled["category"].value_counts())

    cols = [c for c in ["score", "signal_power", "sinr"] if c in df_uncontrolled.columns]
    if "category" in df_uncontrolled.columns and cols:
        print("\nMean metrics per category:")
        print(df_uncontrolled.groupby("category")[cols].mean())

        print("\nStd metrics per category:")
        print(df_uncontrolled.groupby("category")[cols].std())

    if HAS_SNS and "category" in df_uncontrolled.columns and "score" in df_uncontrolled.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="category", y="score", data=df_uncontrolled)
        plt.title(f"Score distribution by category - {cfg_name}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        _finalize_plot(f"{cfg_name}_score_boxplot.png")

    if "score" in df_uncontrolled.columns and "category" in df_uncontrolled.columns:
        curve = fpr_curve_uncontrolled(df_uncontrolled, THRESH_GRID)
        print("\nFPR curve (noise only) sample:")
        print(curve.head())

        plot_line(curve["gamma"], curve["FPR"], title=f"FPR vs Threshold γ ({cfg_name})", xlabel="γ", ylabel="FPR (noise only)", filename=f"{cfg_name}_fpr_vs_threshold.png")

        if "signal_power" in df_uncontrolled.columns:
            tbl_power = fpr_by_param_binned(df_uncontrolled, "signal_power", gamma=GAMMA_WORK, bins=8)
            print(f"\nFPR vs signal_power bins (gamma={GAMMA_WORK})")
            print(tbl_power[["bin", "N_noise", "FPR"]])
            plot_binned_fpr(tbl_power, title=f"{cfg_name}: FPR vs Signal Power (gamma={GAMMA_WORK})", xlabel="Signal Power bin midpoint", ylabel="FPR", filename=f"{cfg_name}_fpr_vs_signal_power.png")

        if "bandwidth" in df_uncontrolled.columns:
            tbl_bw = fpr_by_param_binned(df_uncontrolled, "bandwidth", gamma=GAMMA_WORK, bins=8)
            print(f"\nFPR vs bandwidth bins (gamma={GAMMA_WORK})")
            print(tbl_bw[["bin", "N_noise", "FPR"]])
            plot_binned_fpr(tbl_bw, title=f"{cfg_name}: FPR vs Bandwidth (gamma={GAMMA_WORK})", xlabel="Bandwidth bin midpoint", ylabel="FPR", filename=f"{cfg_name}_fpr_vs_bandwidth.png")

            print("\n[Debug] bandwidth nunique (all):", df_uncontrolled["bandwidth"].nunique())
            print(df_uncontrolled["bandwidth"].value_counts().head(10))
            noise_df = df_uncontrolled[df_uncontrolled["category"] == -1]
            print("[Debug] bandwidth nunique (noise):", noise_df["bandwidth"].nunique())
            print(noise_df["bandwidth"].value_counts().head(10))

        tbl_proxy = pr_table_uncontrolled_proxy(df_uncontrolled, gamma=GAMMA_WORK)
        print("\nProxy Relative Recall per class (uncontrolled):")
        print(tbl_proxy)


# =========================================================
# TASK 2 — TRIGGER DESIGN (NP + TRACK GATES)
# =========================================================
def neyman_pearson_threshold(df, alpha=0.05, grid=None, score_col="score", category_col="category", noise_label=-1):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 501)

    curve = fpr_curve_uncontrolled(df, grid, score_col=score_col, category_col=category_col, noise_label=noise_label)
    valid = curve.dropna()
    ok = valid[valid["FPR"] <= alpha]
    if ok.empty:
        return np.nan, curve
    gamma_star = float(ok.iloc[0]["gamma"])
    return gamma_star, curve


def track_table_uncontrolled(df, track_col="association_id", score_col="score", category_col="category"):
    if track_col not in df.columns:
        raise KeyError(f"Missing '{track_col}' column. Track-based trigger requires association_id.")

    g = df.groupby(track_col, dropna=False)

    def mode_or_first(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else x.iloc[0]

    tracks = g.agg(
        n_hits=(score_col, "size"),
        max_score=(score_col, "max"),
        med_sinr=("sinr", "median") if "sinr" in df.columns else (score_col, "size"),
        med_power=("signal_power", "median") if "signal_power" in df.columns else (score_col, "size"),
        start_time=("start_time", "min") if "start_time" in df.columns else (score_col, "size"),
        end_time=("end_time", "max") if "end_time" in df.columns else (score_col, "size"),
        track_label=(category_col, mode_or_first),
    ).reset_index()

    if "start_time" in tracks.columns and "end_time" in tracks.columns:
        tracks["track_duration_sec"] = (tracks["end_time"] - tracks["start_time"]).dt.total_seconds()
    else:
        tracks["track_duration_sec"] = np.nan

    if "sinr" not in df.columns:
        tracks["med_sinr"] = np.nan
    if "signal_power" not in df.columns:
        tracks["med_power"] = np.nan

    return tracks


def apply_trigger(tracks, gamma_star, min_hits=2, min_dur_sec=1.0, min_sinr_db=None, min_power_dbm=None):
    score_gate = tracks["max_score"] >= gamma_star
    persist_gate = (tracks["n_hits"] >= min_hits) | (tracks["track_duration_sec"] >= min_dur_sec)

    fired = score_gate & persist_gate

    if min_sinr_db is not None and "med_sinr" in tracks.columns:
        fired = fired & (tracks["med_sinr"] >= min_sinr_db)

    if min_power_dbm is not None and "med_power" in tracks.columns:
        fired = fired & (tracks["med_power"] >= min_power_dbm)

    return fired


def evaluate_trigger_tracks(tracks, fired_mask, noise_label=-1):
    noise_tracks = tracks["track_label"] == noise_label
    signal_tracks = ~noise_tracks

    fpr_track = fired_mask[noise_tracks].mean() if noise_tracks.sum() > 0 else np.nan
    rel_recall = fired_mask[signal_tracks].mean() if signal_tracks.sum() > 0 else np.nan

    return {
        "N_tracks": int(len(tracks)),
        "N_noise_tracks": int(noise_tracks.sum()),
        "N_signal_tracks": int(signal_tracks.sum()),
        "Triggered_tracks": int(fired_mask.sum()),
        "FPR_track": fpr_track,
        "RelativeRecall_track": rel_recall,
    }


def task2_trigger_design(cfg_name, df_uncontrolled):
    print("\n" + "=" * 80)
    print(f"TASK 2 — TRIGGER DESIGN (NP + gates): {cfg_name}")
    print("=" * 80)

    gamma_star, curve = neyman_pearson_threshold(
        df_uncontrolled,
        alpha=ALPHA_BUDGET,
        grid=THRESH_GRID,
        score_col="score",
        category_col="category",
        noise_label=-1,
    )

    if np.isnan(gamma_star):
        print(f"[NP] No gamma in THRESH_GRID satisfies FPR <= {ALPHA_BUDGET}. Using gamma_star=GAMMA_WORK.")
        gamma_star = GAMMA_WORK
    else:
        print(f"[NP] alpha={ALPHA_BUDGET} => gamma*={gamma_star:.3f}")

    plot_line(curve["gamma"], curve["FPR"], title=f"NP curve: FPR vs γ (noise only) - {cfg_name}", xlabel="γ", ylabel="FPR", filename=f"{cfg_name}_np_curve.png")

    if "signal_power" in df_uncontrolled.columns:
        tbl_power_np = fpr_by_param_binned(df_uncontrolled, "signal_power", gamma=gamma_star, bins=8)
        print(f"\n[Task2] FPR vs signal_power bins (gamma*={gamma_star:.3f})")
        print(tbl_power_np[["bin", "N_noise", "FPR"]])

    if "bandwidth" in df_uncontrolled.columns:
        tbl_bw_np = fpr_by_param_binned(df_uncontrolled, "bandwidth", gamma=gamma_star, bins=8)
        print(f"\n[Task2] FPR vs bandwidth bins (gamma*={gamma_star:.3f})")
        print(tbl_bw_np[["bin", "N_noise", "FPR"]])

    tracks = track_table_uncontrolled(df_uncontrolled)

    base_fired = tracks["max_score"] >= GAMMA_WORK
    base_metrics = evaluate_trigger_tracks(tracks, base_fired)

    np_only_fired = tracks["max_score"] >= gamma_star
    np_only_metrics = evaluate_trigger_tracks(tracks, np_only_fired)

    fired = apply_trigger(
        tracks,
        gamma_star=gamma_star,
        min_hits=TRIGGER_MIN_HITS,
        min_dur_sec=TRIGGER_MIN_DUR_SEC,
        min_sinr_db=TRIGGER_MIN_SINR_DB,
        min_power_dbm=TRIGGER_MIN_POWER_DBM,
    )
    trig_metrics = evaluate_trigger_tracks(tracks, fired)

    print("\n--- BASELINE (score-only) ---")
    print(f"gamma={GAMMA_WORK}")
    for k, v in base_metrics.items():
        print(f"{k}: {v}")

    print("\n--- NP-ONLY (score-only) ---")
    print(f"gamma*={gamma_star:.3f}")
    for k, v in np_only_metrics.items():
        print(f"{k}: {v}")

    print("\n--- FINAL TRIGGER (NP + gates) ---")
    print(f"gamma*={gamma_star:.3f}, min_hits={TRIGGER_MIN_HITS}, min_dur={TRIGGER_MIN_DUR_SEC}, min_sinr={TRIGGER_MIN_SINR_DB}, min_power={TRIGGER_MIN_POWER_DBM}")
    for k, v in trig_metrics.items():
        print(f"{k}: {v}")

    print("\n--- SUMMARY (Before -> After) ---")
    print(f"FPR_track: {base_metrics['FPR_track']:.3f} -> {trig_metrics['FPR_track']:.3f}")
    print(f"RelativeRecall_track: {base_metrics['RelativeRecall_track']:.3f} -> {trig_metrics['RelativeRecall_track']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF spectral detection analysis.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing controlled.csv and the uncontrolled CSV files.",
    )
    parser.add_argument(
        "--save-plots-dir",
        type=Path,
        default=None,
        help="Optional directory where plots will be saved as PNG files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows.",
    )
    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
def main():
    global SHOW_PLOTS, SAVE_PLOTS_DIR

    args = parse_args()
    SHOW_PLOTS = not args.no_show
    SAVE_PLOTS_DIR = args.save_plots_dir

    df_controlled = read_csv(args.data_dir, CONTROLLED_FILE)

    uncontrolled = {}
    for cfg, fn in UNCONTROLLED_FILES.items():
        df = read_csv(args.data_dir, fn)
        df = prepare_uncontrolled(df)
        uncontrolled[cfg] = df

    task1_controlled(df_controlled)
    for cfg_name, df_u in uncontrolled.items():
        task1_uncontrolled(cfg_name, df_u)

    for cfg_name, df_u in uncontrolled.items():
        if "association_id" not in df_u.columns:
            print(f"\n[WARNING] {cfg_name} missing association_id. Skipping Task 2 track trigger.")
            continue
        task2_trigger_design(cfg_name, df_u)


if __name__ == "__main__":
    main()
