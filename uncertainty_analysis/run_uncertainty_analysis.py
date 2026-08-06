#!/usr/bin/env python
"""Direct uncertainty, identifiability, and global-sensitivity workflow.

Methods:
1. Ten-start Latin-hypercube optimizer-stability analysis.
2. Heteroscedastic Gaussian parametric bootstrap (50 refits).
3. Profile likelihood with nuisance parameters re-optimized.
4. Direct conditional objective-function grids.
5. Morris elementary-effects global sensitivity.

No response-surface surrogate is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_LOCAL_MPLCONFIG = Path(__file__).resolve().parent / "outputs" / ".mplconfig"
_LOCAL_MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INVERSION = HERE.parent
ASSETS = HERE / "assets"
BASELINE_OUTPUT = ASSETS / "baseline_fit"
ELECTRIC_CHECKPOINT = ASSETS / "common_prefix" / "electric" / "checkpoint_electric.ckpt"
WATER_CHECKPOINT = ASSETS / "common_prefix" / "water" / "checkpoint_water.ckpt"
MODEL = HERE / "source" / "inverse_model_uq.py"
DATA = ASSETS / "data"
DATA_FILES = (
    "pH_calibration.csv",
    "TotalPb_calibration.csv",
    "pH_validation.csv",
    "TotalPb_validation.csv",
)
PARAMS = ("Keos", "RH", "pH50")
CASES = ("theta020", "theta047")
PARAMETER_BOUNDS = {
    "Keos": (0.20, 1.80),
    "RH": (8.0, 40.0),
    "pH50": (1.50, 8.0),
}
COLORS = {"Keos": "#1f4e79", "RH": "#d49a2a", "pH50": "#2e8b57"}
PROFILE_THRESHOLD_95 = 3.841458820694124
SURFACE_THRESHOLD_95 = 5.991464547107979


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def savefig(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def latest_baseline_summary():
    path = BASELINE_OUTPUT / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {path}")
    return path


def bootstrap_data(rng, target):
    """Draw one parametric-bootstrap value for every calibration observation.

    The fitted baseline prediction is the sampling center and the observation's
    measurement-SD column supplies its Gaussian standard deviation. Validation data
    are copied unchanged and remain out-of-fit.
    """
    frames = {name: pd.read_csv(DATA / name) for name in DATA_FILES}
    target.mkdir(parents=True, exist_ok=True)
    keys = [
        "case",
        "theta0",
        "psi_ic_m",
        "time_h",
        "time_day",
        "distance_cm",
    ]
    specifications = {
        "pH_calibration.csv": ("pH", "pH_obs", "pH_measurement_sd"),
        "TotalPb_calibration.csv": (
            "TotalPb",
            "TotalPb_obs_mol_m3_bulk",
            "TotalPb_measurement_sd",
        ),
    }
    metadata = {
        "scheme": "heteroscedastic Gaussian parametric bootstrap",
        "center": "fitted baseline prediction",
        "validation": "unchanged and excluded from fitting",
    }
    for name, (tag, value_column, sigma_column) in specifications.items():
        source = frames[name].copy()
        source["_uq_order"] = np.arange(len(source))
        predicted = []
        for case in CASES:
            prediction_path = BASELINE_OUTPUT / f"{case}_{tag}_calibration.csv"
            block = pd.read_csv(prediction_path)
            predicted.append(block[keys + ["prediction"]])
        predicted = pd.concat(predicted, ignore_index=True)
        merged = source.merge(
            predicted,
            on=keys,
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values("_uq_order")
        if merged["prediction"].isna().any() or len(merged) != len(source):
            raise RuntimeError(f"Baseline predictions do not align with {name}")
        sigma = merged[sigma_column].to_numpy(float)
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError(f"{sigma_column} must contain finite positive values")
        merged[value_column] = rng.normal(
            merged["prediction"].to_numpy(float), sigma
        )
        merged.drop(columns=["prediction", "_uq_order"]).to_csv(
            target / name, index=False
        )
        metadata[f"{tag}_n"] = int(len(merged))
    for name in ("pH_validation.csv", "TotalPb_validation.csv"):
        shutil.copy2(DATA / name, target / name)
    return metadata


def metric_summary(summary):
    metrics = summary["metrics"]
    likelihood = summary["observation_likelihood"]
    best_guard = summary.get("best_guard", {})
    stop = summary.get("stage_stop_status", {}).get("joint", {})
    return {
        **summary["parameters"],
        "calibration_neg2loglik": likelihood["calibration"]["neg2loglik"],
        "validation_neg2loglik": likelihood["validation"]["neg2loglik"],
        "calibration_chi_square": likelihood["calibration"]["chi_square"],
        "calibration_reduced_chi_square": likelihood["calibration"]["reduced_chi_square"],
        "calibration_pH_neg2loglik": likelihood["calibration"]["components"]["pH"],
        "calibration_Pb_neg2loglik": likelihood["calibration"]["components"]["TotalPb"],
        "pH_validation_RMSE": float(
            np.mean([metrics[f"{case}_pH_validation_RMSE"] for case in CASES])
        ),
        "Pb_validation_RMSE": float(
            np.mean([metrics[f"{case}_TotalPb_validation_RMSE"] for case in CASES])
        ),
        "physics_objective": summary["final_physics_objective"],
        "best_iteration": best_guard.get("iteration"),
        "best_selection_score": best_guard.get("score"),
        "stop_iteration": stop.get("iteration"),
        "stop_reason": stop.get("reason"),
        "numerical_recoveries": stop.get("numerical_recoveries", 0),
    }


def fit_command(args, data_dir, job_root, seed, starts, fixed=None, protocol=None):
    fixed = fixed or {}
    protocol = protocol or args.protocol
    cfg = args.baseline_config
    cmd = [
        sys.executable,
        str(MODEL),
        "--data-dir",
        str(data_dir),
        "--output-root",
        str(job_root),
        "--summary-only",
        "--seed",
        str(seed),
        "--init-Keos",
        str(starts["Keos"]),
        "--init-RH",
        str(starts["RH"]),
        "--init-pH50",
        str(starts["pH50"]),
        "--ph50-lower",
        str(cfg["ph50_lower"]),
        "--ph50-upper",
        str(cfg["ph50_upper"]),
        "--width",
        str(cfg["width"]),
        "--depth",
        str(cfg["depth"]),
        "--data-stride",
        str(protocol["data_stride"]),
        "--n-res",
        str(protocol["n_res"]),
        "--n-face",
        str(protocol["n_face"]),
        "--n-ic",
        str(protocol["n_ic"]),
        "--mass-nx",
        str(protocol["mass_grid"]),
        "--mass-nt",
        str(protocol["mass_grid"]),
        "--start-stage",
        str(protocol["start_stage"]),
        "--electric-iters",
        str(protocol["electric_iters"]),
        "--water-iters",
        str(protocol["water_iters"]),
        "--acid-iters",
        str(protocol["acid_iters"]),
        "--pb-iters",
        str(protocol["pb_iters"]),
        "--joint-iters",
        str(protocol["joint_iters"]),
        "--water-state-pretrain-iters",
        str(protocol["water_state_pretrain"]),
        "--acid-state-pretrain-iters",
        str(protocol["acid_state_pretrain"]),
        "--pb-state-pretrain-iters",
        str(protocol["pb_state_pretrain"]),
        "--joint-state-pretrain-iters",
        str(protocol["joint_state_pretrain"]),
        "--alternating-block-size",
        str(protocol["block_size"]),
        "--joint-state-steps",
        "1",
        "--joint-parameter-steps",
        "1",
        "--joint-learning-rate",
        str(cfg["joint_learning_rate"]),
        "--joint-parameter-learning-rate",
        str(cfg["joint_parameter_learning_rate"]),
        "--warm-parameter-learning-rate",
        str(cfg["warm_parameter_learning_rate"]),
        "--keos-parameter-learning-rate",
        str(cfg["keos_parameter_learning_rate"]),
        "--rh-parameter-learning-rate",
        str(cfg["rh_parameter_learning_rate"]),
        "--ph50-parameter-learning-rate",
        str(cfg["ph50_parameter_learning_rate"]),
        "--warm-lr-decay-steps",
        str(cfg["warm_lr_decay_steps"]),
        "--coupled-lr-decay-steps",
        str(cfg["coupled_lr_decay_steps"]),
        "--lr-decay-rate",
        str(cfg["lr_decay_rate"]),
        "--adaptive-stop",
        "--monitor-every",
        str(protocol["monitor_every"]),
        "--stability-window",
        str(protocol["stability_window"]),
        "--min-electric-iters",
        str(protocol["min_electric_iters"]),
        "--min-water-iters",
        str(protocol["min_water_iters"]),
        "--min-acid-iters",
        str(protocol["min_acid_iters"]),
        "--min-pb-iters",
        str(protocol["min_pb_iters"]),
        "--min-joint-iters",
        str(protocol["min_joint_iters"]),
        "--stability-score-rtol",
        str(cfg["stability_score_rtol"]),
        "--stability-parameter-rtol",
        str(cfg["stability_parameter_rtol"]),
        "--max-numerical-recoveries",
        str(cfg["max_numerical_recoveries"]),
        "--acid-relative-weight",
        str(cfg["acid_relative_weight"]),
        "--acid-relative-floor",
        str(cfg["acid_relative_floor"]),
        "--pH-weight",
        str(cfg["pH_weight"]),
        "--Pb-weight",
        str(cfg["Pb_weight"]),
        "--faraday-weight",
        str(cfg["faraday_weight"]),
        "--acid-bc-weight",
        str(cfg["acid_bc_weight"]),
        "--mass-weight",
        str(cfg["mass_weight"]),
        "--adsorption-slope-n",
        str(cfg["adsorption_slope_n"]),
        "--selection-physics-weight",
        str(cfg["selection_physics_weight"]),
        "--selection-burn-in-iters",
        str(protocol["selection_burn_in"]),
        "--selection-data-mode",
        "gaussian-likelihood",
        "--pb-patience",
        "0",
        "--joint-patience",
        "0",
        "--run-role",
        "diagnostic",
        "--run-label",
        "formal_uq",
    ]
    cmd += ["--resume-checkpoint", str(protocol["resume_checkpoint"])]
    if protocol.get("resume_network_only", False):
        cmd.append("--resume-network-only")
    if protocol.get("balance_scales_json"):
        cmd += ["--balance-scales-json", str(protocol["balance_scales_json"])]
    flags = {"Keos": "--fixed-keos", "RH": "--fixed-rh", "pH50": "--fixed-ph50"}
    for name, value in fixed.items():
        cmd += [flags[name], str(value)]
    if len(fixed) < len(PARAMS):
        cmd.append("--block-coordinate-curriculum")
    return cmd


def run_fit(args, label, data_dir, seed, starts, fixed=None, extra=None, protocol=None):
    job_root = args.run_dir / "fits" / label
    summaries = list(job_root.glob("*/summary.json"))
    if not summaries:
        job_root.mkdir(parents=True, exist_ok=True)
        child_environment = os.environ.copy()
        child_environment["DDE_BACKEND"] = "tensorflow.compat.v1"
        child_environment["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
        child_environment["PYTHONUNBUFFERED"] = "1"
        log_path = job_root / "fit.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n=== {datetime.now().isoformat()} starting {label} ===\n"
            )
            stream.flush()
            proc = subprocess.run(
                fit_command(args, data_dir, job_root, seed, starts, fixed, protocol),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_environment,
            )
            stream.write(
                f"\n=== {datetime.now().isoformat()} exit code {proc.returncode} ===\n"
            )
        summaries = list(job_root.glob("*/summary.json"))
        if proc.returncode or not summaries:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1600:]
            row = {"label": label, "status": "failed", "error": log_tail}
            row.update(extra or {})
            return row
    summary_path = max(summaries, key=lambda path: path.stat().st_mtime)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {"label": label, "status": "ok", **metric_summary(summary)}
    row.update({f"initial_{name}": starts[name] for name in PARAMS})
    row.update(extra or {})
    return row


def execute_jobs(args, jobs, csv_path):
    def execute(job):
        row = run_fit(
            args,
            job["label"],
            job.get("data_dir", DATA),
            job["seed"],
            job["starts"],
            fixed=job.get("fixed"),
            extra=job.get("extra"),
            protocol=job.get("protocol"),
        )
        return job["order"], row

    completed = {}
    workers = max(1, min(args.workers, len(jobs))) if jobs else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {pool.submit(execute, job): job for job in jobs}
        for done, future in enumerate(as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                order, row = future.result()
            except Exception as exc:
                order = job["order"]
                row = {"label": job["label"], "status": "failed", "error": repr(exc)}
                row.update(job.get("extra", {}))
            completed[order] = row
            save_csv([completed[index] for index in sorted(completed)], csv_path)
            print(f"[{done}/{len(jobs)}] {job['label']} -> {row['status']}", flush=True)
    return pd.DataFrame([completed[index] for index in sorted(completed)])


def latin_hypercube_starts(count, seed, required_start):
    """Return one required start plus a reproducible space-filling design."""
    if count < 2:
        return [dict(required_start)]
    rng = np.random.default_rng(seed)
    design_count = count - 1
    unit = np.empty((design_count, len(PARAMS)), dtype=float)
    for column in range(len(PARAMS)):
        unit[:, column] = (
            rng.permutation(design_count) + rng.random(design_count)
        ) / design_count
    return [dict(required_start)] + [scale_point(point) for point in unit]


def summarize_multistart(results, baseline, outdir):
    """Summarize dependence of the fitted solution on physical-parameter starts."""
    good = results[results["status"] == "ok"].copy()
    if good.empty:
        raise RuntimeError("All multi-start stability fits failed")
    good = good.sort_values("start_id").reset_index(drop=True)
    good.to_csv(outdir / "multistart_stability_successful.csv", index=False)
    rows = []
    for parameter in PARAMS:
        values = good[parameter].to_numpy(float)
        lower, upper = PARAMETER_BOUNDS[parameter]
        rows.append(
            {
                "parameter": parameter,
                "n_successful": int(len(values)),
                "baseline_estimate": float(baseline[parameter]),
                "mean": float(np.mean(values)),
                "standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "range_fraction_of_parameter_bounds": float(
                    (np.max(values) - np.min(values)) / (upper - lower)
                ),
                "maximum_absolute_deviation_from_baseline": float(
                    np.max(np.abs(values - baseline[parameter]))
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "multistart_stability_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), constrained_layout=True)
    for axis, parameter in zip(axes[:3], PARAMS):
        axis.plot(
            good["start_id"], good[parameter], "o-", color=COLORS[parameter], lw=1.2
        )
        axis.axhline(
            baseline[parameter], color=".2", ls=":", lw=1.2, label="baseline estimate"
        )
        axis.set(xlabel="start ID", ylabel="fitted value", title=parameter)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    axes[3].plot(
        good["start_id"], good["calibration_neg2loglik"], "o-", color="#555555", lw=1.2
    )
    axes[3].set(xlabel="start ID", ylabel=r"$-2\log L$", title="Calibration objective")
    axes[3].grid(alpha=0.2)
    fig.suptitle("Multi-start optimizer stability")
    savefig(fig, outdir / "figures" / "FigF0_multistart_stability")
    return summary


def best_by_group(frame, group_columns):
    good = frame[frame["status"] == "ok"].copy()
    if good.empty:
        return good
    indices = good.groupby(group_columns)["calibration_neg2loglik"].idxmin()
    return good.loc[indices].sort_values(group_columns).reset_index(drop=True)


def bootstrap_summary(best, mle, outdir):
    rows = []
    for parameter in PARAMS:
        values = best[parameter].to_numpy(float)
        q025, q975 = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "parameter": parameter,
                "n_successful": len(values),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "standard_error": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "percentile_95_lower": float(q025),
                "percentile_95_upper": float(q975),
                "basic_95_lower": float(2.0 * mle[parameter] - q975),
                "basic_95_upper": float(2.0 * mle[parameter] - q025),
                "best_fit": mle[parameter],
            }
        )
    save_csv(rows, outdir / "bootstrap_intervals.csv")
    if len(best) >= 2:
        best[list(PARAMS)].corr(method="pearson").to_csv(
            outdir / "bootstrap_parameter_correlation_pearson.csv"
        )
        best[list(PARAMS)].corr(method="spearman").to_csv(
            outdir / "bootstrap_parameter_correlation_spearman.csv"
        )
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), constrained_layout=True)
        for ax, parameter in zip(axes, PARAMS):
            ax.hist(
                best[parameter],
                bins=min(14, max(4, len(best) // 5)),
                color=COLORS[parameter],
                alpha=0.82,
            )
            ax.axvline(mle[parameter], color=".15", ls=":", lw=1.3, label="best fit")
            ax.set(xlabel=parameter, ylabel="bootstrap count", title=parameter)
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Parametric bootstrap for three-replicate mean data")
        savefig(fig, outdir / "figures" / "FigF1_parametric_bootstrap")
    return pd.DataFrame(rows)


def interpolate_crossing(x0, y0, x1, y1, threshold):
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def profile_interval(profile, parameter, threshold=PROFILE_THRESHOLD_95):
    data = profile[profile["profile_parameter"] == parameter].sort_values("fixed_value").copy()
    data["delta_neg2loglik"] = data["calibration_neg2loglik"] - data["calibration_neg2loglik"].min()
    x = data["fixed_value"].to_numpy(float)
    delta = data["delta_neg2loglik"].to_numpy(float)
    minimum = int(np.argmin(delta))
    left = minimum
    while left > 0 and delta[left - 1] <= threshold:
        left -= 1
    right = minimum
    while right + 1 < len(delta) and delta[right + 1] <= threshold:
        right += 1
    if left == 0 and delta[left] <= threshold:
        lower, lower_status = float(x[0]), "open_at_grid_boundary"
    else:
        lower = interpolate_crossing(x[left - 1], delta[left - 1], x[left], delta[left], threshold)
        lower_status = "crossed"
    if right == len(delta) - 1 and delta[right] <= threshold:
        upper, upper_status = float(x[-1]), "open_at_grid_boundary"
    else:
        upper = interpolate_crossing(x[right], delta[right], x[right + 1], delta[right + 1], threshold)
        upper_status = "crossed"
    return data, {
        "parameter": parameter,
        "profile_minimum": float(x[minimum]),
        "neg2loglik_min": float(data["calibration_neg2loglik"].min()),
        "ci95_lower": lower,
        "ci95_upper": upper,
        "lower_status": lower_status,
        "upper_status": upper_status,
        "threshold_delta_neg2loglik": threshold,
    }


def summarize_profiles(best, mle, outdir):
    intervals, plotted = [], {}
    for parameter in PARAMS:
        data, interval = profile_interval(best, parameter)
        intervals.append(interval)
        plotted[parameter] = data
    save_csv(intervals, outdir / "profile_likelihood_intervals.csv")
    pd.concat(plotted.values(), ignore_index=True).to_csv(
        outdir / "profile_likelihood_best.csv", index=False
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), constrained_layout=True)
    for ax, parameter in zip(axes, PARAMS):
        data = plotted[parameter]
        ax.plot(data["fixed_value"], data["delta_neg2loglik"], "o-", color=COLORS[parameter])
        ax.axhline(PROFILE_THRESHOLD_95, color="#b22222", ls="--", lw=1.2, label="95% threshold")
        ax.axvline(mle[parameter], color=".2", ls=":", lw=1.2, label="best fit")
        ax.set(xlabel=f"fixed {parameter}", ylabel=r"$\Delta(-2\log L)$", title=parameter)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Profile likelihood with nuisance parameters re-optimized")
    savefig(fig, outdir / "figures" / "FigF2_profile_likelihood")
    return pd.DataFrame(intervals)


def scale_point(unit_point):
    return {
        parameter: float(
            PARAMETER_BOUNDS[parameter][0]
            + unit_point[index]
            * (PARAMETER_BOUNDS[parameter][1] - PARAMETER_BOUNDS[parameter][0])
        )
        for index, parameter in enumerate(PARAMS)
    }


def morris_design(levels, trajectories, seed):
    """Return randomized Morris trajectories on an even-level unit grid."""
    if levels < 4 or levels % 2:
        raise ValueError("Morris levels must be an even integer >= 4")
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, levels)
    delta = levels / (2.0 * (levels - 1.0))
    points, transitions = [], []
    for trajectory in range(trajectories):
        directions = rng.choice((-1.0, 1.0), size=len(PARAMS))
        unit = np.empty(len(PARAMS), dtype=float)
        for index, direction in enumerate(directions):
            valid = (
                grid[grid <= 1.0 - delta + 1e-12]
                if direction > 0
                else grid[grid >= delta - 1e-12]
            )
            unit[index] = float(rng.choice(valid))
        order = rng.permutation(len(PARAMS))
        points.append({"trajectory": trajectory, "step": 0, "unit": unit.copy()})
        for step, parameter_index in enumerate(order, 1):
            before = unit.copy()
            unit[parameter_index] += directions[parameter_index] * delta
            points.append({"trajectory": trajectory, "step": step, "unit": unit.copy()})
            transitions.append(
                {
                    "trajectory": trajectory,
                    "from_step": step - 1,
                    "to_step": step,
                    "parameter": PARAMS[parameter_index],
                    "delta_unit": float(unit[parameter_index] - before[parameter_index]),
                }
            )
    return points, pd.DataFrame(transitions)


def summarize_morris(results, transitions, outdir, resamples=2000, seed=20260843):
    responses = ("calibration_neg2loglik", "pH_validation_RMSE", "Pb_validation_RMSE")
    indexed = results[results["status"] == "ok"].set_index(["trajectory", "step"])
    elementary = []
    for transition in transitions.to_dict("records"):
        before_key = (transition["trajectory"], transition["from_step"])
        after_key = (transition["trajectory"], transition["to_step"])
        if before_key not in indexed.index or after_key not in indexed.index:
            continue
        before = indexed.loc[before_key]
        after = indexed.loc[after_key]
        for response in responses:
            elementary.append(
                {
                    **transition,
                    "response": response,
                    "elementary_effect": float(
                        (after[response] - before[response]) / transition["delta_unit"]
                    ),
                }
            )
    effects = pd.DataFrame(elementary)
    if effects.empty:
        raise RuntimeError("No complete Morris elementary effects were available")
    effects.to_csv(outdir / "morris_elementary_effects.csv", index=False)
    rng = np.random.default_rng(seed)
    rows = []
    for (response, parameter), block in effects.groupby(["response", "parameter"], sort=False):
        values = block["elementary_effect"].to_numpy(float)
        boot_mu_star = []
        for _ in range(resamples):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_mu_star.append(float(np.mean(np.abs(sample))))
        rows.append(
            {
                "response": response,
                "parameter": parameter,
                "n_trajectories": len(values),
                "mu": float(np.mean(values)),
                "mu_star": float(np.mean(np.abs(values))),
                "sigma": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "mu_star_ci95_lower": float(np.quantile(boot_mu_star, 0.025)),
                "mu_star_ci95_upper": float(np.quantile(boot_mu_star, 0.975)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "morris_indices.csv", index=False)
    response_list = list(responses)
    fig, axes = plt.subplots(
        1, len(response_list), figsize=(12.0, 3.5), constrained_layout=True
    )
    for ax, response in zip(axes, response_list):
        block = summary[summary["response"] == response].set_index("parameter").reindex(PARAMS)
        x = np.arange(len(PARAMS))
        lower = block["mu_star"] - block["mu_star_ci95_lower"]
        upper = block["mu_star_ci95_upper"] - block["mu_star"]
        ax.bar(x, block["mu_star"], color=[COLORS[name] for name in PARAMS], alpha=0.85)
        ax.errorbar(
            x,
            block["mu_star"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            color=".15",
            capsize=3,
        )
        ax.set_xticks(x, PARAMS)
        ax.set(ylabel=r"Morris $\mu^*$", title=response)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Morris elementary-effects global sensitivity")
    savefig(fig, outdir / "figures" / "FigF3_morris_sensitivity")
    return summary


def grid_with_center(parameter, levels, center):
    values = np.linspace(*PARAMETER_BOUNDS[parameter], levels)
    values[int(np.argmin(np.abs(values - center)))] = center
    return np.sort(values)


def summarize_surfaces(surface, mle, outdir):
    good = surface[surface["status"] == "ok"].copy()
    if good.empty:
        raise RuntimeError("All direct objective-surface fits failed")
    rows = []
    pairs = good["surface_pair"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(4.1 * len(pairs), 3.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, pair in zip(axes.flat, pairs):
        block = good[good["surface_pair"] == pair].copy()
        x_name = str(block["x_parameter"].iloc[0])
        y_name = str(block["y_parameter"].iloc[0])
        minimum = float(block["calibration_neg2loglik"].min())
        block["delta_neg2loglik"] = block["calibration_neg2loglik"] - minimum
        rows.extend(block.to_dict("records"))
        pivot = block.pivot(index="y_value", columns="x_value", values="delta_neg2loglik")
        xx, yy = np.meshgrid(pivot.columns.to_numpy(float), pivot.index.to_numpy(float))
        zz = pivot.to_numpy(float)
        contour = ax.contourf(xx, yy, zz, levels=18, cmap="viridis")
        if np.nanmin(zz) <= SURFACE_THRESHOLD_95 <= np.nanmax(zz):
            ax.contour(
                xx,
                yy,
                zz,
                levels=[SURFACE_THRESHOLD_95],
                colors="white",
                linewidths=1.4,
            )
        ax.scatter(
            mle[x_name], mle[y_name], marker="x", s=60, color="white", lw=1.8, label="best fit"
        )
        nuisance = str(block["nuisance_parameter"].iloc[0])
        nuisance_value = float(block["nuisance_value"].iloc[0])
        ax.set(
            xlabel=x_name,
            ylabel=y_name,
            title=f"{x_name} × {y_name}; {nuisance}={nuisance_value:.3g}",
        )
        ax.grid(alpha=0.15)
        fig.colorbar(contour, ax=ax, label=r"direct $\Delta(-2\log L)$")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Direct conditional objective-function surfaces")
    savefig(fig, outdir / "figures" / "FigF4_direct_objective_surfaces")
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "direct_objective_surfaces.csv", index=False)
    return result


def full_stage_protocol(mode, start_stage, resume_checkpoint):
    """Construct the staged-refitting protocol."""
    smoke = mode == "smoke"
    if smoke:
        caps = {"water": 30, "acid": 40, "pb": 40, "joint": 40}
        sampling = {
            "n_res": 96,
            "n_face": 32,
            "n_ic": 32,
            "mass_grid": 7,
            "data_stride": 25,
        }
        monitor = 10
        window = 2
        minima = {"electric": 0, "water": 30, "acid": 40, "pb": 40, "joint": 40}
        pretrain = {"water": 10, "acid": 10, "pb": 10, "joint": 10}
        block_size = 10
        selection_burn_in = 40
    else:
        caps = {"water": 6000, "acid": 12000, "pb": 6000, "joint": 2000}
        sampling = {
            "n_res": 3000,
            "n_face": 500,
            "n_ic": 500,
            "mass_grid": 21,
            "data_stride": 1,
        }
        monitor = 100
        window = 5
        minima = {
            "electric": 500,
            "water": 800,
            "acid": 2500,
            "pb": 1000,
            "joint": 300,
        }
        pretrain = {"water": 500, "acid": 1000, "pb": 500, "joint": 200}
        block_size = 100
        selection_burn_in = 500

    order = ("electric", "water", "acid", "pb", "joint")
    start_index = order.index(start_stage)
    iterations = {
        "electric_iters": 0,
        "water_iters": caps["water"] if start_index <= order.index("water") else 0,
        "acid_iters": caps["acid"] if start_index <= order.index("acid") else 0,
        "pb_iters": caps["pb"] if start_index <= order.index("pb") else 0,
        "joint_iters": caps["joint"],
    }
    return {
        "start_stage": start_stage,
        "resume_checkpoint": str(Path(resume_checkpoint).resolve()),
        "resume_network_only": False,
        **iterations,
        "water_state_pretrain": pretrain["water"],
        "acid_state_pretrain": pretrain["acid"],
        "pb_state_pretrain": pretrain["pb"],
        "joint_state_pretrain": pretrain["joint"],
        "block_size": block_size,
        **sampling,
        "monitor_every": monitor,
        "stability_window": window,
        "min_electric_iters": minima["electric"],
        "min_water_iters": minima["water"],
        "min_acid_iters": minima["acid"],
        "min_pb_iters": minima["pb"],
        "min_joint_iters": minima["joint"],
        "selection_burn_in": selection_burn_in,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal", "full"), default="formal")
    parser.add_argument(
        "--sections", default="stability,bootstrap,profile,morris,surface"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--control-csv", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    presets = {
        "smoke": {
            "stability_starts": 2,
            "bootstrap_reps": 1,
            "profile_levels": 2,
            "morris_levels": 4,
            "morris_trajectories": 1,
            "surface_levels": 2,
            "protocol": {
                "joint_iters": 40,
                "state_pretrain": 10,
                "block_size": 20,
                "n_res": 96,
                "n_face": 32,
                "n_ic": 32,
                "mass_grid": 7,
                "data_stride": 25,
                "monitor_every": 20,
                "stability_window": 2,
                "min_joint_iters": 40,
                "selection_burn_in": 40,
            },
            "fixed_protocol": {
                "joint_iters": 30,
                "state_pretrain": 10,
                "block_size": 15,
                "n_res": 96,
                "n_face": 32,
                "n_ic": 32,
                "mass_grid": 7,
                "data_stride": 25,
                "monitor_every": 15,
                "stability_window": 2,
                "min_joint_iters": 30,
                "selection_burn_in": 30,
            },
        },
        "formal": {
            "stability_starts": 10,
            "bootstrap_reps": 50,
            "profile_levels": 21,
            "morris_levels": 6,
            "morris_trajectories": 10,
            "surface_levels": 9,
            "protocol": {
                "joint_iters": 640,
                "state_pretrain": 80,
                "block_size": 40,
                "n_res": 384,
                "n_face": 96,
                "n_ic": 96,
                "mass_grid": 13,
                "data_stride": 10,
                "monitor_every": 40,
                "stability_window": 3,
                "min_joint_iters": 240,
                "selection_burn_in": 160,
            },
            "fixed_protocol": {
                "joint_iters": 150,
                "state_pretrain": 0,
                "block_size": 30,
                "n_res": 384,
                "n_face": 96,
                "n_ic": 96,
                "mass_grid": 13,
                "data_stride": 10,
                "monitor_every": 25,
                "stability_window": 3,
                "min_joint_iters": 100,
                "selection_burn_in": 100,
            },
        },
        "full": {
            "stability_starts": 20,
            "bootstrap_reps": 100,
            "profile_levels": 31,
            "morris_levels": 8,
            "morris_trajectories": 20,
            "surface_levels": 13,
            "protocol": {
                "joint_iters": 800,
                "state_pretrain": 100,
                "block_size": 50,
                "n_res": 768,
                "n_face": 192,
                "n_ic": 192,
                "mass_grid": 17,
                "data_stride": 5,
                "monitor_every": 50,
                "stability_window": 3,
                "min_joint_iters": 300,
                "selection_burn_in": 200,
            },
            "fixed_protocol": {
                "joint_iters": 250,
                "state_pretrain": 0,
                "block_size": 40,
                "n_res": 768,
                "n_face": 192,
                "n_ic": 192,
                "mass_grid": 17,
                "data_stride": 5,
                "monitor_every": 40,
                "stability_window": 3,
                "min_joint_iters": 160,
                "selection_burn_in": 160,
            },
        },
    }
    preset = presets[args.mode]
    preset["protocol"] = full_stage_protocol(args.mode, "acid", WATER_CHECKPOINT)
    preset["profile_keos_protocol"] = full_stage_protocol(
        args.mode, "water", ELECTRIC_CHECKPOINT
    )
    preset["fixed_protocol"] = full_stage_protocol(
        args.mode, "water", ELECTRIC_CHECKPOINT
    )
    preset["stability_protocol"] = full_stage_protocol(
        args.mode, "water", ELECTRIC_CHECKPOINT
    )
    preset["stability_protocol"]["resume_network_only"] = True
    args.protocol = preset["protocol"]
    args.fixed_protocol = preset["fixed_protocol"]
    sections = {section.strip() for section in args.sections.split(",") if section.strip()}
    unknown = sections - {
        "control", "stability", "bootstrap", "profile", "morris", "surface"
    }
    if unknown:
        parser.error(f"Unknown sections: {sorted(unknown)}")

    baseline_path = (args.baseline_summary or latest_baseline_summary()).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    args.baseline_config_path = baseline_path.parent / "analysis_config.json"
    args.baseline_checkpoint = baseline_path.parent / "model.ckpt"
    args.baseline_config = json.loads(args.baseline_config_path.read_text(encoding="utf-8"))
    args.workers = max(1, args.workers)
    args.run_dir = (
        args.resume
        or (HERE / "outputs" / f"standard_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}")
    ).resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "figures").mkdir(exist_ok=True)
    mle = {parameter: float(baseline["parameters"][parameter]) for parameter in PARAMS}
    original_start = {"Keos": 0.5, "RH": 35.0, "pH50": 7.0}
    water_branch_start = {"Keos": mle["Keos"], "RH": 35.0, "pH50": 7.0}

    manifest = {
        "analysis_class": "direct_standard_uq_three_replicate_mean",
        "mode": args.mode,
        "sections": sorted(sections),
        "baseline_summary": str(baseline_path),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "control_csv": None if args.control_csv is None else str(args.control_csv.resolve()),
        "parameter_bounds": PARAMETER_BOUNDS,
        "observation_model": "heteroscedastic Gaussian likelihood using observation SD columns",
        "data_representation": (
            "arithmetic mean of three independent experimental replicates "
            "per case-location-time"
        ),
        "measurement_sd_definition": (
            "sample standard deviation across the three independent experimental replicates"
        ),
        "optimizer_stability": (
            "ten starts in formal mode: required original start plus a seeded "
            "Latin-hypercube design; common state-network checkpoint restored network-only"
        ),
        "bootstrap_method": "heteroscedastic Gaussian parametric bootstrap",
        "bootstrap_center": "fitted baseline predictions",
        "bootstrap_scale": "the measurement SD attached to each observation",
        "profile_threshold": {
            "df": 1,
            "confidence": 0.95,
            "delta_neg2loglik": PROFILE_THRESHOLD_95,
        },
        "surface_threshold": {
            "df": 2,
            "confidence": 0.95,
            "delta_neg2loglik": SURFACE_THRESHOLD_95,
        },
        "global_sensitivity": "direct Morris elementary effects",
        "surface_method": "direct conditional grid; third physical parameter fixed at best fit",
        "original_physical_start": original_start,
        "shared_water_branch_start": water_branch_start,
        "baseline_parameters": mle,
        "refit_initialization": (
            "shared deterministic electric/water prefix followed by complete "
            "data-dependent staged refitting"
        ),
        "preset": preset,
    }
    (args.run_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    (args.run_dir / "best_fit.json").write_text(json.dumps(mle, indent=2), encoding="utf-8")
    save_csv(
        [{"status": "baseline_estimate", **mle, "baseline_summary": str(baseline_path)}],
        args.run_dir / "baseline_estimate.csv",
    )

    if "stability" in sections:
        starts_list = latin_hypercube_starts(
            preset["stability_starts"], 20260811, original_start
        )
        jobs = []
        for start_id, starts in enumerate(starts_list):
            jobs.append(
                {
                    "order": start_id,
                    "label": f"stability_start_{start_id:02d}",
                    "seed": 20267000,
                    "starts": starts,
                    "protocol": preset["stability_protocol"],
                    "extra": {
                        "start_id": start_id,
                        "design": (
                            "required_original_start" if start_id == 0 else "latin_hypercube"
                        ),
                    },
                }
            )
        stability_all = execute_jobs(
            args, jobs, args.run_dir / "multistart_stability_all.csv"
        )
        summarize_multistart(stability_all, mle, args.run_dir)

    if "bootstrap" in sections:
        rng = np.random.default_rng(20260820)
        jobs, order = [], 0
        data_root = args.run_dir / "bootstrap_data"
        starts_list = [dict(water_branch_start)]
        for bootstrap_id in range(1, preset["bootstrap_reps"] + 1):
            target = data_root / f"bootstrap_{bootstrap_id:04d}"
            bootstrap_model = bootstrap_data(rng, target)
            for start_id, starts in enumerate(starts_list):
                jobs.append(
                    {
                        "order": order,
                        "label": f"bootstrap_{bootstrap_id:04d}_start_{start_id}",
                        "data_dir": target,
                        "seed": 20262000 + start_id,
                        "starts": starts,
                        "extra": {
                            "bootstrap_id": bootstrap_id,
                            "start_id": start_id,
                            "bootstrap_model": json.dumps(bootstrap_model),
                        },
                    }
                )
                order += 1
        bootstrap_all = execute_jobs(
            args, jobs, args.run_dir / "bootstrap_refits.csv"
        )
        bootstrap_best = best_by_group(bootstrap_all, ["bootstrap_id"])
        bootstrap_best.to_csv(args.run_dir / "bootstrap_best.csv", index=False)
        bootstrap_summary(bootstrap_best, mle, args.run_dir)

    if "profile" in sections:
        jobs, order = [], 0
        for parameter in PARAMS:
            starts_list = [
                dict(original_start if parameter == "Keos" else water_branch_start)
            ]
            profile_protocol = (
                preset["profile_keos_protocol"] if parameter == "Keos" else args.protocol
            )
            values = grid_with_center(
                parameter, preset["profile_levels"], mle[parameter]
            )
            for grid_id, fixed_value in enumerate(values):
                for start_id, starts in enumerate(starts_list):
                    jobs.append(
                        {
                            "order": order,
                            "label": f"profile_{parameter}_{grid_id:02d}_start_{start_id}",
                            "seed": 20266000 + start_id,
                            "starts": starts,
                            "fixed": {parameter: float(fixed_value)},
                            "protocol": profile_protocol,
                            "extra": {
                                "profile_parameter": parameter,
                                "grid_id": grid_id,
                                "fixed_value": float(fixed_value),
                                "start_id": start_id,
                            },
                        }
                    )
                    order += 1
        profile_all = execute_jobs(
            args, jobs, args.run_dir / "profile_refits.csv"
        )
        profile_best = best_by_group(profile_all, ["profile_parameter", "grid_id"])
        summarize_profiles(profile_best, mle, args.run_dir)

    if "morris" in sections:
        points, transitions = morris_design(
            preset["morris_levels"], preset["morris_trajectories"], 20260840
        )
        transitions.to_csv(args.run_dir / "morris_transitions.csv", index=False)
        jobs = []
        for order, point in enumerate(points):
            requested = scale_point(point["unit"])
            jobs.append(
                {
                    "order": order,
                    "label": f"morris_t{point['trajectory']:03d}_s{point['step']:02d}",
                    "seed": 20266800,
                    "starts": original_start,
                    "fixed": requested,
                    "protocol": args.fixed_protocol,
                    "extra": {
                        "trajectory": point["trajectory"],
                        "step": point["step"],
                        **{f"requested_{name}": value for name, value in requested.items()},
                    },
                }
            )
        morris_all = execute_jobs(args, jobs, args.run_dir / "morris_runs.csv")
        summarize_morris(morris_all, transitions, args.run_dir)

    if "surface" in sections:
        pairs = (("Keos", "RH"), ("Keos", "pH50"), ("RH", "pH50"))
        jobs, order = [], 0
        for x_name, y_name in pairs:
            nuisance = next(name for name in PARAMS if name not in (x_name, y_name))
            x_values = grid_with_center(
                x_name, preset["surface_levels"], mle[x_name]
            )
            y_values = grid_with_center(
                y_name, preset["surface_levels"], mle[y_name]
            )
            for grid_i, x_value in enumerate(x_values):
                for grid_j, y_value in enumerate(y_values):
                    fixed = dict(mle)
                    fixed[x_name] = float(x_value)
                    fixed[y_name] = float(y_value)
                    jobs.append(
                        {
                            "order": order,
                            "label": f"surface_{x_name}_{y_name}_{grid_i:02d}_{grid_j:02d}",
                            "seed": 20266900,
                            "starts": original_start,
                            "fixed": fixed,
                            "protocol": args.fixed_protocol,
                            "extra": {
                                "surface_pair": f"{x_name}_{y_name}",
                                "x_parameter": x_name,
                                "y_parameter": y_name,
                                "x_value": float(x_value),
                                "y_value": float(y_value),
                                "nuisance_parameter": nuisance,
                                "nuisance_value": float(mle[nuisance]),
                                "grid_i": grid_i,
                                "grid_j": grid_j,
                            },
                        }
                    )
                    order += 1
        surface_all = execute_jobs(
            args, jobs, args.run_dir / "direct_surface_runs.csv"
        )
        summarize_surfaces(surface_all, mle, args.run_dir)

    print(f"Standard UQ outputs: {args.run_dir}")


if __name__ == "__main__":
    main()
