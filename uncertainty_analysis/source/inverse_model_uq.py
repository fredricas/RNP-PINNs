#!/usr/bin/env python
"""Joint pH50 / Keos / RH PINN inversion for the current KunsatSe3 model.

The electric, water, acid/base and Pb subnetworks share one TF1 graph/session.
They are warmed up sequentially and finally optimized through one fully coupled
loss.  The three physical parameters are shared trainable variables for the
entire lifetime of that graph.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_MODEL = ROOT / "model" / "quanouhe1_KunsatSe3.ipynb"


def load_base_equations():
    """Load physical-model definitions from the notebook."""
    nb = json.loads(BASE_MODEL.read_text(encoding="utf-8-sig"))
    marker = "# 8) Build layers & BC samples"
    chunks = []
    found = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        if marker in text:
            chunks.append(text.split(marker, 1)[0])
            found = True
            break
        chunks.append(text)
    if not found:
        raise RuntimeError(f"Definition marker not found in {BASE_MODEL}")
    exec(compile("\n".join(chunks), str(BASE_MODEL), "exec"), globals())


load_base_equations()


CASES = {
    "theta020": {"psi_ic_m": -166.9363086307076, "fixed_saturated": False},
    "theta047": {"psi_ic_m": -16.33394573035923, "fixed_saturated": False},
}
CASE_NAMES = tuple(CASES)


def save_parameter_trajectory(history, outdir, final_values):
    """Save a phase-aware parameter-versus-training-step plot for every run."""
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history).copy()
    stage_order = [
        stage for stage in ("electric", "water", "acid", "pb", "joint")
        if stage in set(frame["stage"])
    ]
    offsets = {}
    offset = 0
    for stage in stage_order:
        offsets[stage] = offset
        offset += int(frame.loc[frame["stage"] == stage, "iteration"].max())
    frame["cumulative_iteration"] = [
        offsets[stage] + iteration
        for stage, iteration in zip(frame["stage"], frame["iteration"])
    ]
    boundaries = [offsets[stage] for stage in stage_order] + [offset]
    colors = {"Keos": "#2864b4", "RH": "#d47c16", "pH50": "#24855b"}
    final_map = dict(zip(("Keos", "RH", "pH50"), map(float, final_values)))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    for axis, parameter in zip(axes, ("Keos", "RH", "pH50")):
        axis.plot(
            frame["cumulative_iteration"], frame[parameter],
            color=colors[parameter], marker="o", markersize=2.8, linewidth=1.6,
        )
        axis.axhline(final_map[parameter], color="#333333", linestyle=":", linewidth=1.2,
                     label="selected final")
        for boundary in boundaries[1:-1]:
            axis.axvline(boundary, color="#aaaaaa", linewidth=0.8)
        axis.set_xlabel("cumulative training step")
        axis.set_ylabel(parameter)
        axis.set_title(parameter)
        axis.grid(alpha=0.22)
    for left, right, stage in zip(boundaries[:-1], boundaries[1:], stage_order):
        axes[0].text(
            (left + right) / 2, 1.02, stage,
            transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7, rotation=22,
        )
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Inverse parameters versus cumulative training step", fontsize=11)
    fig.savefig(outdir / "parameter_trajectory.png", dpi=220)
    fig.savefig(outdir / "parameter_trajectory.svg")
    plt.close(fig)


def finite(z):
    """Fail fast on invalid physics instead of turning NaN/Inf into zero residual."""
    return tf.debugging.check_numerics(z, "non-finite value in inverse model")


def mse(z):
    return tf.reduce_mean(tf.square(finite(z)))


def bounded_parameter(name, initial, low, high):
    if not low < initial < high:
        raise ValueError(f"{name}: require {low} < {initial} < {high}")
    q = (initial - low) / (high - low)
    raw0 = math.log(q / (1.0 - q))
    raw = tf.Variable(raw0, dtype=tf.float32, trainable=True, name=f"raw_{name}")
    value = tf.constant(low, tf.float32) + tf.constant(high - low, tf.float32) * tf.sigmoid(raw)
    return raw, tf.identity(value, name=name)


def inversion_parameter(name, initial, low, high, fixed=None):
    """Return one physical parameter and its optional trainable raw variable."""
    if fixed is None:
        raw, value = bounded_parameter(name, initial, low, high)
        return value, [raw]
    if not low <= fixed <= high:
        raise ValueError(f"{name}: fixed value must be within [{low}, {high}]")
    return tf.constant(float(fixed), tf.float32, name=name), []


def make_optimizer(loss, var_list, learning_rate, clip_norm, name, clip_each=False,
                   decay_steps=0, decay_rate=1.0):
    with tf.compat.v1.variable_scope(name):
        if not var_list:
            return tf.no_op(), tf.constant(0.0, tf.float32), []
        optimizer_step = tf.compat.v1.get_variable(
            "optimizer_step", shape=(), dtype=tf.int64,
            initializer=tf.zeros_initializer(), trainable=False,
        )
        if int(decay_steps) > 0 and float(decay_rate) < 1.0:
            effective_learning_rate = tf.compat.v1.train.exponential_decay(
                float(learning_rate), optimizer_step, int(decay_steps),
                float(decay_rate), staircase=True,
            )
        else:
            effective_learning_rate = tf.constant(float(learning_rate), tf.float32)
        optimizer = tf.compat.v1.train.AdamOptimizer(effective_learning_rate)
        pairs = [(g, v) for g, v in optimizer.compute_gradients(loss, var_list=var_list) if g is not None]
        if not pairs:
            return tf.no_op(), tf.constant(0.0, tf.float32), []
        grads, variables = zip(*pairs)
        checked = [tf.debugging.check_numerics(g, f"non-finite gradient in {name}") for g in grads]
        norm = tf.linalg.global_norm(checked)
        # During the fully coupled step, one large state-network gradient must
        # not suppress every shared physical-parameter gradient.  Clip tensors
        # separately there; retain global clipping for the warm-up stages.
        if clip_each:
            safe = [tf.clip_by_norm(g, float(clip_norm)) for g in checked]
        else:
            safe, _ = tf.clip_by_global_norm(checked, float(clip_norm))
        return optimizer.apply_gradients(
            list(zip(safe, variables)), global_step=optimizer_step
        ), norm, list(variables)


class CaseGraph:
    def __init__(self, name, meta, layers, params, data, args, evaluation_data=None):
        self.name = name
        self.meta = meta
        self.args = args
        self.Keos, self.RH, self.pH50 = params
        self.psi_ic = float(meta["psi_ic_m"])
        self.fixed_saturated = bool(meta["fixed_saturated"])
        self.xmin, self.xmax = 0.0, 0.2
        self.tmin, self.tmax = 0.0, 5.0
        self.L = self.xmax - self.xmin

        with tf.compat.v1.variable_scope(name):
            self.t_res = tf.compat.v1.placeholder(tf.float32, [None, 1], name="t_res")
            self.x_res = tf.compat.v1.placeholder(tf.float32, [None, 1], name="x_res")
            self.t_ic = tf.compat.v1.placeholder(tf.float32, [None, 1], name="t_ic")
            self.x_ic = tf.compat.v1.placeholder(tf.float32, [None, 1], name="x_ic")
            self.t_left = tf.compat.v1.placeholder(tf.float32, [None, 1], name="t_left")
            self.x_left = tf.compat.v1.placeholder(tf.float32, [None, 1], name="x_left")
            self.t_right = tf.compat.v1.placeholder(tf.float32, [None, 1], name="t_right")
            self.x_right = tf.compat.v1.placeholder(tf.float32, [None, 1], name="x_right")

            with tf.compat.v1.variable_scope("electric"):
                self.electric_mlp = DenseMLP(layers, NETC["act_scale"], NETC["trainable_layer_gain"])
            if not self.fixed_saturated:
                with tf.compat.v1.variable_scope("water"):
                    self.water_mlp = DenseMLP(layers, NETC["act_scale"], NETC["trainable_layer_gain"])
            else:
                self.water_mlp = None
            with tf.compat.v1.variable_scope("acid"):
                self.acid_mlp = DenseMLP(layers, NETC["act_scale"], NETC["trainable_layer_gain"])
            pb_layers = list(layers)
            pb_layers[-1] = 2
            with tf.compat.v1.variable_scope("pb"):
                self.pb_mlp = DenseMLP(pb_layers, NETC["act_scale"], NETC["trainable_layer_gain"])

        self.phi_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=f"{name}/electric")
        self.water_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=f"{name}/water")
        self.acid_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=f"{name}/acid")
        self.pb_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=f"{name}/pb")

        self._build_data_tensors(data, evaluation_data)
        self._build_losses()
        self._build_profile_tensors()

    def phi(self, t, x):
        return self.electric_mlp.forward(tf.concat([t, x], 1))

    def head(self, t, x):
        if self.fixed_saturated:
            return tf.zeros_like(t)
        X = tf.concat([t, x], 1)
        return water_head_from_raw(self.water_mlp.forward(X), t, x)

    def theta(self, t, x):
        return theta_function(self.head(t, x))

    def water_flux(self, t, x):
        h = self.head(t, x)
        theta = theta_function(h)
        q_hyd = -K_function(h) * grad0(h, x)
        phi_x = grad0(self.phi(t, x), x)
        q_eo = self.Keos * keo_of_theta(theta) * phi_x
        return finite(q_hyd), finite(q_eo), finite(q_hyd + q_eo)

    def acid_a(self, t, x):
        raw = tf.clip_by_value(self.acid_mlp.forward(tf.concat([t, x], 1)), -8.0, 8.0)
        gate_tau = tf.constant(float(H_PAR.get("a_ic_gate_tau", 0.02)), tf.float32)
        gate = tf.where(t <= self.tmin, tf.zeros_like(t), 1.0 - tf.exp(-(t - self.tmin) / tf.maximum(gate_tau, 1e-12)))
        a_ic = tf.constant(float(self.a_ic), tf.float32)
        return finite(a_ic + gate * tf.constant(float(H_PAR.get("a_raw_scale", 20.0)), tf.float32) * raw)

    @staticmethod
    def acid_species_from_a(a):
        s = tf.sqrt(tf.square(a) + 4.0 * Kw_const)
        cH = tf.where(a >= 0.0, 0.5 * (a + s), (2.0 * Kw_const) / tf.maximum(s - a, 1e-30))
        cOH = tf.where(a <= 0.0, 0.5 * (-a + s), (2.0 * Kw_const) / tf.maximum(s + a, 1e-30))
        return finite(cH), finite(cOH)

    def acid_species(self, t, x):
        return self.acid_species_from_a(self.acid_a(t, x))

    def pH(self, t, x):
        cH, _ = self.acid_species(t, x)
        value = -tf.math.log(tf.maximum(cH, 1e-30) / 1000.0) / tf.constant(np.log(10.0), tf.float32)
        return tf.clip_by_value(finite(value), float(H_PAR.get("pH_min", 0.0)), float(H_PAR.get("pH_max", 14.0)))

    def acid_fluxes(self, t, x):
        cH, cOH = self.acid_species(t, x)
        theta = self.theta(t, x)
        _, _, q = self.water_flux(t, x)
        phi_x = grad0(self.phi(t, x), x)
        Deff_H, u_H = diffusion_pieces(theta, q, DL_H, Dw_H, z_H)
        Deff_OH, u_OH = diffusion_pieces(theta, q, DL_OH, Dw_OH, z_OH)
        cHx, cOHx = grad0(cH, x), grad0(cOH, x)
        JH = q * cH - Deff_H * cHx - u_H * cH * phi_x
        JOH = q * cOH - Deff_OH * cOHx - u_OH * cOH * phi_x
        JH_nonadv = -Deff_H * cHx - u_H * cH * phi_x
        JOH_nonadv = -Deff_OH * cOHx - u_OH * cOH * phi_x
        return finite(JH), finite(JOH), finite(JH_nonadv), finite(JOH_nonadv)

    def pb_total(self, t, x):
        raw = self.pb_mlp.forward(tf.concat([t, x], 1))[:, 0:1]
        scale = tf.constant(float(PB_PAR.get("psi_raw_scale", 20.0)), tf.float32)
        raw = scale * tf.tanh(raw / tf.maximum(scale, 1e-6))
        initial = tf.ones_like(t) * tf.constant(float(SURF_INIT["SOPb0"] + SURF_INIT.get("Pp0", 0.0)), tf.float32)
        eta0 = tf.math.log(tf.math.expm1(tf.maximum(initial, 1e-6)) + 1e-12)
        tau = tf.constant(float(PB_PAR.get("psi_ic_gate_tau", 0.02)), tf.float32)
        gate = tf.where(t <= self.tmin, tf.zeros_like(t), 1.0 - tf.exp(-(t - self.tmin) / tf.maximum(tau, 1e-6)))
        total = tf.nn.softplus(eta0 + gate * raw)
        site_total = float(SURF_INIT["SOH0"] + SURF_INIT["SOPb0"] + SURF_INIT["SOH2_0"] + SURF_INIT["SOm0"])
        return tf.clip_by_value(finite(total), 0.0, site_total + 1.0e4)

    def pb_aux_flux(self, t, x):
        raw = self.pb_mlp.forward(tf.concat([t, x], 1))[:, 1:2]
        scale = tf.constant(float(PB_PAR.get("flux_raw_scale", 2.0)), tf.float32)
        raw = scale * tf.tanh(raw / tf.maximum(scale, 1e-6))
        return finite((x - self.xmin) / self.L * raw)

    def pb_species(self, t, x, total=None):
        total = self.pb_total(t, x) if total is None else total
        theta = tf.clip_by_value(finite(self.theta(t, x)), 1e-6, 1.0)
        cH, cOH = self.acid_species(t, x)
        cH = tf.clip_by_value(finite(cH), 1e-12, 1e3)
        cOH = tf.clip_by_value(finite(cOH), 1e-10, 1e4)
        pH = self.pH(t, x)

        n_ads = tf.constant(float(self.args.adsorption_slope_n), tf.float32)
        exponent = tf.clip_by_value(tf.constant(np.log(10.0), tf.float32) * n_ads * (self.pH50 - pH), -60.0, 60.0)
        f_ads = tf.clip_by_value(1.0 / (1.0 + tf.exp(exponent)), 0.0, 1.0)

        c0 = tf.constant(float(SURF_INIT["SOPb0"] + SURF_INIT.get("Pp0", 0.0)), tf.float32)
        site_total = tf.constant(float(SURF_INIT["SOH0"] + SURF_INIT["SOPb0"] + SURF_INIT["SOH2_0"] + SURF_INIT["SOm0"]), tf.float32)
        native_pool = tf.minimum(total, c0)
        # Absolute target based on initial inventory: this is the non-zero-Pb history rule.
        locked = tf.minimum(native_pool, f_ads * c0)
        released = tf.maximum(native_pool - locked, 0.0)
        incoming = tf.maximum(total - native_pool, 0.0)
        vacant = tf.maximum(site_total - locked, 0.0)
        f_reactive = tf.clip_by_value(f_ads, 0.0, 1.0 - 1e-6)
        odds = f_reactive / tf.maximum(1.0 - f_reactive, 1e-6)

        ads_no_precip = tf.minimum(vacant, f_reactive * incoming)
        aq_no_precip = tf.maximum(released + incoming - ads_no_precip, 0.0)
        ksp = tf.constant(float(CFG["FIELDS"]["PB"]["CHEM"].get("Ksp_PbOH2", 1.43e-11)), tf.float32)
        m_precip = tf.constant(float(CFG["FIELDS"]["PB"]["CHEM"].get("m", 2.0)), tf.float32)
        aq_sat = tf.minimum(theta * ksp / tf.maximum(tf.pow(cOH, m_precip), 1e-30), released + incoming)
        precip_gate = aq_no_precip > (1.0 + 1e-6) * tf.maximum(aq_sat, 1e-30)
        ads_precip = tf.minimum(vacant, tf.minimum(incoming, odds * aq_sat))
        aq_bulk = tf.where(precip_gate, aq_sat, aq_no_precip)
        ads = locked + tf.where(precip_gate, ads_precip, ads_no_precip)
        precip = tf.nn.relu(total - ads - aq_bulk)
        aq_bulk = total - ads - precip
        cPb = tf.clip_by_value(aq_bulk / tf.maximum(theta, 1e-12), 0.0, 1e9)

        k_pr = tf.constant(float(CFG["FIELDS"]["PB"]["CHEM"]["k_pr_f"] / CFG["FIELDS"]["PB"]["CHEM"]["k_pr_b"]), tf.float32)
        k_dpr = tf.constant(float(CFG["FIELDS"]["PB"]["CHEM"]["k_dpr_f"] / CFG["FIELDS"]["PB"]["CHEM"]["k_dpr_b"]), tf.float32)
        cH_bulk = tf.maximum(theta * cH, 1e-12)
        aH, bH = k_pr * cH_bulk, k_dpr / cH_bulk
        free_sites = tf.maximum(site_total - ads, 0.0)
        SOH = free_sites / tf.maximum(1.0 + aH + bH, 1e-12)
        SOH2, SOm = aH * SOH, bH * SOH
        return tuple(finite(v) for v in (cPb, ads, SOH, SOH2, SOm, precip, cOH))

    def pb_fluxes(self, t, x):
        total = self.pb_total(t, x)
        cPb, *_ = self.pb_species(t, x, total)
        cPb_x = grad0(cPb, x)
        theta = self.theta(t, x)
        _, _, q = self.water_flux(t, x)
        phi_x = grad0(self.phi(t, x), x)
        Dstar = Dstar_of_theta(theta, Dw_Pb)
        Deff = Dstar + DL_Pb * tf.abs(q)
        z = tf.constant(float(PB_PAR.get("transport_z", PB_PAR.get("z", 2.0))), tf.float32)
        u = z * F_c / (R_c * T_c) * Dstar
        J_adv, J_diff, J_em = q * cPb, -Deff * cPb_x, -u * cPb * phi_x
        J_aux = self.pb_aux_flux(t, x)
        return tuple(finite(v) for v in (J_aux, J_adv, J_diff, J_em))

    def pb_physical_flux(self, t, x):
        _, adv, diff, em = self.pb_fluxes(t, x)
        return finite(adv + diff + em)

    def acid_residual_terms(self, t, x, coupled):
        cH, cOH = self.acid_species(t, x)
        JH, JOH, _, _ = self.acid_fluxes(t, x)
        theta = self.theta(t, x)
        storage = self.RH * (theta * grad0(cH, t) + grad0(theta, t) * cH) \
                  - (theta * grad0(cOH, t) + grad0(theta, t) * cOH)
        source = tf.zeros_like(t)
        if coupled:
            _, ads, _, SOH2, SOm, precip, _ = self.pb_species(t, x)
            source = grad0(-SOH2 + SOm + ads + 2.0 * precip, t)
            # Picard-style coupling: use the current Pb source value in the
            # acid residual, but do not backpropagate through the nonsmooth
            # min/where species partition.  The source is recomputed on every
            # update, so the forward coupling is retained without undefined
            # gradients at phase-switch surfaces.
            source = tf.stop_gradient(source)
        transport_source = grad0(JH - JOH, x) - source
        residual = storage + transport_source
        return finite(storage), finite(transport_source), finite(residual)

    def acid_residual(self, t, x, coupled):
        return self.acid_residual_terms(t, x, coupled)[2]

    def _build_data_tensors(self, data, evaluation_data=None):
        if data is None:
            self.ph_cal_df = self.pb_cal_df = self.ph_val_df = self.pb_val_df = pd.DataFrame()
            self.ph_cal = self.pb_cal = self.ph_val = self.pb_val = None
            self.ph_cal_pred = self.pb_cal_pred = self.ph_val_pred = self.pb_val_pred = None
            return

        def tensors(df, value_col, sigma_col, prefix):
            t = tf.constant(df["time_day"].to_numpy(np.float32).reshape(-1, 1), name=f"{prefix}_t")
            x = tf.constant((df["distance_cm"].to_numpy(np.float32) / 100.0).reshape(-1, 1), name=f"{prefix}_x")
            y = tf.constant(df[value_col].to_numpy(np.float32).reshape(-1, 1), name=f"{prefix}_y")
            sigma = tf.constant(df[sigma_col].to_numpy(np.float32).reshape(-1, 1), name=f"{prefix}_sigma")
            return t, x, y, sigma

        self.ph_cal_df = data["ph_cal"].reset_index(drop=True)
        self.pb_cal_df = data["pb_cal"].reset_index(drop=True)
        self.ph_val_df = data["ph_val"].reset_index(drop=True)
        self.pb_val_df = data["pb_val"].reset_index(drop=True)
        self.ph_cal = tensors(self.ph_cal_df, "pH_obs", "pH_measurement_sd", "ph_cal")
        self.pb_cal = tensors(self.pb_cal_df, "TotalPb_obs_mol_m3_bulk", "TotalPb_measurement_sd", "pb_cal")
        self.ph_val = tensors(self.ph_val_df, "pH_obs", "pH_measurement_sd", "ph_val")
        self.pb_val = tensors(self.pb_val_df, "TotalPb_obs_mol_m3_bulk", "TotalPb_measurement_sd", "pb_val")
        self.ph_cal_pred = self.pH(self.ph_cal[0], self.ph_cal[1])
        self.pb_cal_pred = self.pb_total(self.pb_cal[0], self.pb_cal[1])
        self.ph_val_pred = self.pH(self.ph_val[0], self.ph_val[1])
        self.pb_val_pred = self.pb_total(self.pb_val[0], self.pb_val[1])

        evaluation_data = evaluation_data or data
        self.ph_cal_full_df = evaluation_data["ph_cal"].reset_index(drop=True)
        self.pb_cal_full_df = evaluation_data["pb_cal"].reset_index(drop=True)
        self.ph_val_full_df = evaluation_data["ph_val"].reset_index(drop=True)
        self.pb_val_full_df = evaluation_data["pb_val"].reset_index(drop=True)
        self.ph_cal_full = tensors(self.ph_cal_full_df, "pH_obs", "pH_measurement_sd", "full_ph_cal")
        self.pb_cal_full = tensors(
            self.pb_cal_full_df, "TotalPb_obs_mol_m3_bulk",
            "TotalPb_measurement_sd", "full_pb_cal"
        )
        self.ph_val_full = tensors(self.ph_val_full_df, "pH_obs", "pH_measurement_sd", "full_ph_val")
        self.pb_val_full = tensors(
            self.pb_val_full_df, "TotalPb_obs_mol_m3_bulk",
            "TotalPb_measurement_sd", "full_pb_val"
        )
        self.ph_cal_full_pred = self.pH(self.ph_cal_full[0], self.ph_cal_full[1])
        self.pb_cal_full_pred = self.pb_total(self.pb_cal_full[0], self.pb_cal_full[1])
        self.ph_val_full_pred = self.pH(self.ph_val_full[0], self.ph_val_full[1])
        self.pb_val_full_pred = self.pb_total(self.pb_val_full[0], self.pb_val_full[1])

    def _mass_integral_loss(self):
        nx, nt = self.args.mass_nx, self.args.mass_nt
        t_values = np.linspace(self.tmin, self.tmax, nt, dtype=np.float32)
        x_values = np.linspace(self.xmin, self.xmax, nx, dtype=np.float32)
        tt, xx = np.meshgrid(t_values, x_values, indexing="ij")
        t = tf.constant(tt.reshape(-1, 1), tf.float32)
        x = tf.constant(xx.reshape(-1, 1), tf.float32)
        total = tf.reshape(self.pb_total(t, x), [nt, nx])
        dx = tf.constant(self.L / (nx - 1), tf.float32)
        xw = tf.constant(np.r_[0.5, np.ones(nx - 2), 0.5].reshape(1, -1), tf.float32)
        mass = dx * tf.reduce_sum(total * xw, axis=1, keepdims=True)
        tc = tf.constant(t_values.reshape(-1, 1), tf.float32)
        left = self.pb_physical_flux(tc, tf.ones_like(tc) * self.xmin)
        right = self.pb_physical_flux(tc, tf.ones_like(tc) * self.xmax)
        dt = tc[1:] - tc[:-1]
        cumulative = tf.concat([tf.zeros([1, 1]), tf.cumsum(0.5 * (right[1:] - left[1:] + right[:-1] - left[:-1]) * dt, axis=0)], 0)
        return mse((mass - mass[0:1] + cumulative) / tf.constant(5.0 * self.L, tf.float32))

    def _build_profile_tensors(self):
        self.profile_x_cm = np.linspace(0.0, 20.0, 201, dtype=np.float32)
        t = tf.ones([self.profile_x_cm.size, 1], tf.float32) * 5.0
        x = tf.constant((self.profile_x_cm / 100.0).reshape(-1, 1), tf.float32)
        total = self.pb_total(t, x)
        cPb, ads, _, _, _, precip, _ = self.pb_species(t, x, total)
        theta = self.theta(t, x)
        aq_bulk = theta * cPb
        self.profile_tensors = (self.pH(t, x), total, cPb, aq_bulk, ads, precip,
                                total - aq_bulk - ads - precip)

    def _build_losses(self):
        phi_res = grad0(sigma_eff_of_theta(self.theta(self.t_res, self.x_res)) * grad0(self.phi(self.t_res, self.x_res), self.x_res), self.x_res)
        phi_res_pre = grad0(grad0(self.phi(self.t_res, self.x_res), self.x_res), self.x_res)
        phi_bc = mse((self.phi(self.t_left, self.x_left) - float(ELEC_BD["phi_anode"])) / 5.0) \
                 + mse((self.phi(self.t_right, self.x_right) - float(ELEC_BD["phi_cathode"])) / 5.0)
        self.loss_electric_pre = mse(phi_res_pre) + 100.0 * phi_bc
        self.loss_electric = mse(phi_res / tf.constant(10.0, tf.float32)) + 100.0 * phi_bc

        theta = self.theta(self.t_res, self.x_res)
        _, _, q = self.water_flux(self.t_res, self.x_res)
        water_res = grad0(theta, self.t_res) + grad0(q, self.x_res)
        theta_ic = theta_function(tf.ones_like(self.t_ic) * self.psi_ic)
        water_ic = mse((self.theta(self.t_ic, self.x_ic) - theta_ic) / 0.5)
        tau = float(CFG["FIELDS"]["WATER"]["PARAM"].get("head_transition_tau", 0.02))
        gate_l = 1.0 - tf.exp(-self.t_left / max(tau, 1e-6))
        gate_r = 1.0 - tf.exp(-self.t_right / max(tau, 1e-6))
        h_left_target = self.psi_ic + gate_l * (0.0 - self.psi_ic)
        h_right_target = self.psi_ic + gate_r * (0.0 - self.psi_ic)
        head_scale = max(abs(self.psi_ic), 1.0)
        water_bc = mse((self.head(self.t_left, self.x_left) - h_left_target) / head_scale) \
                   + mse((self.head(self.t_right, self.x_right) - h_right_target) / head_scale)
        self.loss_water = mse(water_res / 0.5) + 500.0 * water_ic + 50.0 * water_bc

        ja_ref = max(float(H_BC.get("current_efficiency", 0.19)) * float(ELEC_PAR.get("sigma_sat", 0.5)) * 5.0 / self.L / float(CFG["GLOBAL"]["CONSTANTS"]["F"]) * SEC_PER_DAY, 1e-8)
        res_scale = tf.constant(ja_ref / self.L, tf.float32)
        ramp_l = 1.0 - tf.exp(-self.t_left / max(float(H_PAR.get("faraday_ramp_tau", 0.01)), 1e-12))
        ramp_r = 1.0 - tf.exp(-self.t_right / max(float(H_PAR.get("faraday_ramp_tau", 0.01)), 1e-12))
        sigma_l = sigma_eff_of_theta(self.theta(self.t_left, self.x_left))
        sigma_r = sigma_eff_of_theta(self.theta(self.t_right, self.x_right))
        j_l = float(H_BC.get("current_efficiency", 0.19)) * tf.abs(-sigma_l * grad0(self.phi(self.t_left, self.x_left), self.x_left)) / F_c * SEC_PER_DAY
        j_r = float(H_BC.get("current_efficiency", 0.19)) * tf.abs(-sigma_r * grad0(self.phi(self.t_right, self.x_right), self.x_right)) / F_c * SEC_PER_DAY
        _, _, jh_l, joh_l = self.acid_fluxes(self.t_left, self.x_left)
        _, _, jh_r, joh_r = self.acid_fluxes(self.t_right, self.x_right)
        flux_loss = mse((jh_l - j_l * ramp_l) / ja_ref) + mse(joh_l / ja_ref) \
                    + mse(jh_r / ja_ref) + mse((joh_r + j_r * ramp_r) / ja_ref)
        acid_bc_loss = flux_loss
        acid_bc_multiplier = float(self.args.faraday_weight)
        if str(H_BC.get("left_BC", "")).lower() == "dirichlet" and str(H_BC.get("right_BC", "")).lower() == "dirichlet":
            a_left_target, a_right_target = self._dirichlet_a_targets()
            a_ref = max(abs(float(self._a_from_pH_scalar(float(H_BC.get("pH_left", 0.0))))),
                        abs(float(self._a_from_pH_scalar(float(H_BC.get("pH_right", 14.0))))), 1e-8)
            acid_bc_loss = mse((self.acid_a(self.t_left, self.x_left) - a_left_target) / a_ref) \
                           + mse((self.acid_a(self.t_right, self.x_right) - a_right_target) / a_ref)
            acid_bc_multiplier = float(self.args.acid_bc_weight)
        self.loss_pH_data = (mse((self.ph_cal_pred - self.ph_cal[2]) / self.ph_cal[3])
                             if self.ph_cal is not None else tf.constant(0.0, tf.float32))
        # Keep the acid PDE, boundary, and observation terms separate.
        warm_storage, warm_transport_source, warm_acid_residual = self.acid_residual_terms(
            self.t_res, self.x_res, False
        )
        coupled_storage, coupled_transport_source, coupled_acid_residual = self.acid_residual_terms(
            self.t_res, self.x_res, True
        )
        relative_floor = tf.constant(float(self.args.acid_relative_floor), tf.float32) * res_scale
        warm_relative_residual = warm_acid_residual / tf.stop_gradient(
            tf.abs(warm_storage) + tf.abs(warm_transport_source) + relative_floor
        )
        coupled_relative_residual = coupled_acid_residual / tf.stop_gradient(
            tf.abs(coupled_storage) + tf.abs(coupled_transport_source) + relative_floor
        )
        self.loss_acid_pde_warm = mse(warm_acid_residual / res_scale)
        self.loss_acid_pde = mse(coupled_acid_residual / res_scale)
        self.loss_acid_relative_pde_warm = (
            float(self.args.acid_relative_weight) * mse(warm_relative_residual)
        )
        self.loss_acid_relative_pde = (
            float(self.args.acid_relative_weight) * mse(coupled_relative_residual)
        )
        self.loss_acid_bc = acid_bc_multiplier * acid_bc_loss
        self.loss_pH_weighted = float(self.args.pH_weight) * self.loss_pH_data
        self.loss_acid_warm = (
            self.loss_acid_pde_warm + self.loss_acid_relative_pde_warm
            + self.loss_acid_bc + self.loss_pH_weighted
        )
        self.loss_acid = (
            self.loss_acid_pde + self.loss_acid_relative_pde
            + self.loss_acid_bc + self.loss_pH_weighted
        )

        total = self.pb_total(self.t_res, self.x_res)
        j_aux, j_adv, j_diff, j_em = self.pb_fluxes(self.t_res, self.x_res)
        pb_res = grad0(total, self.t_res) + grad0(j_aux, self.x_res)
        constitutive = j_aux - (j_adv + j_diff + j_em)
        left_phys = self.pb_physical_flux(self.t_left, self.x_left)
        right_phys = self.pb_physical_flux(self.t_right, self.x_right)
        left_aux, la, ld, le = self.pb_fluxes(self.t_left, self.x_left)
        right_aux, ra, rd, re = self.pb_fluxes(self.t_right, self.x_right)
        pb_ic = mse((self.pb_total(self.t_ic, self.x_ic) - 5.0) / 5.0)
        self.loss_Pb_data = (mse((self.pb_cal_pred - self.pb_cal[2]) / self.pb_cal[3])
                             if self.pb_cal is not None else tf.constant(0.0, tf.float32))
        self.loss_pb_transport = 5.0 * mse(pb_res / 10.0) + 5.0 * mse(constitutive)
        self.loss_pb_ic = 20.0 * pb_ic
        self.loss_pb_boundary = (
            100.0 * mse(left_phys) + 100.0 * mse(tf.nn.relu(-right_phys))
            + 100.0 * mse(left_aux - (la + ld + le)) + 30.0 * mse(right_aux - (ra + rd + re))
        )
        self.loss_pb_mass = float(self.args.mass_weight) * self._mass_integral_loss()
        self.loss_Pb_weighted = float(self.args.Pb_weight) * self.loss_Pb_data
        self.loss_pb_physics = (
            self.loss_pb_transport + self.loss_pb_ic + self.loss_pb_boundary + self.loss_pb_mass
        )
        self.loss_pb = self.loss_pb_physics + self.loss_Pb_weighted

        self.loss_total = self.loss_electric + self.loss_water + self.loss_acid + self.loss_pb
        self.diagnostics = {
            "electric": self.loss_electric, "water": self.loss_water,
            "acid": self.loss_acid,
            "acid_pde": self.loss_acid_pde,
            "acid_relative_pde": self.loss_acid_relative_pde,
            "acid_boundary": self.loss_acid_bc,
            "pH_data": self.loss_pH_data,
            "pH_data_weighted": self.loss_pH_weighted,
            "pb": self.loss_pb,
            "pb_transport": self.loss_pb_transport,
            "pb_initial": self.loss_pb_ic,
            "pb_boundary": self.loss_pb_boundary,
            "pb_mass": self.loss_pb_mass,
            "Pb_data": self.loss_Pb_data,
            "Pb_data_weighted": self.loss_Pb_weighted,
        }

    @property
    def a_ic(self):
        cH = 1000.0 * 10.0 ** -7.0
        return cH - float(CFG["GLOBAL"]["CHEM"]["Kw"]) / cH

    @staticmethod
    def _a_from_pH_scalar(pH):
        cH = 1000.0 * 10.0 ** (-float(pH))
        return cH - float(CFG["GLOBAL"]["CHEM"]["Kw"]) / max(cH, 1e-30)

    @staticmethod
    def _a_from_pH_tensor(pH):
        cH = 1000.0 * tf.exp(-tf.constant(np.log(10.0), tf.float32) * pH)
        return cH - Kw_const / tf.maximum(cH, 1e-30)

    def _dirichlet_a_targets(self):
        pH0 = float(H_INIT.get("pH_ic", 7.0))
        pH_left = float(H_BC.get("pH_left", pH0))
        pH_right = float(H_BC.get("pH_right", pH0))
        tau = max(float(H_BC.get("reservoir_pH_ramp_tau", H_PAR.get("faraday_ramp_tau", 0.05))), 1e-12)
        ramp_l = 1.0 - tf.exp(-tf.maximum(self.t_left - self.tmin, 0.0) / tau)
        ramp_r = 1.0 - tf.exp(-tf.maximum(self.t_right - self.tmin, 0.0) / tau)
        return (self._a_from_pH_tensor(pH0 + ramp_l * (pH_left - pH0)),
                self._a_from_pH_tensor(pH0 + ramp_r * (pH_right - pH0)))


def read_case_data(data_dir, case, stride=1):
    def select(name):
        df = pd.read_csv(data_dir / name)
        df = df[df["case"] == case].copy()
        if df.empty:
            raise ValueError(f"No rows for case={case!r} in {data_dir / name}")
        if name.startswith("pH_"):
            local_stride = max(int(stride), 1)
            if local_stride > 1:
                group_columns = ["distance_cm"]
                selected_indices = []
                for _, block in df.groupby(group_columns, sort=False):
                    selected_indices.extend(block.index[::local_stride].tolist())
                df = df.loc[selected_indices]
        return df.reset_index(drop=True)
    return {
        "ph_cal": select("pH_calibration.csv"),
        "pb_cal": select("TotalPb_calibration.csv"),
        "ph_val": select("pH_validation.csv"),
        "pb_val": select("TotalPb_validation.csv"),
    }


def gaussian_observation_likelihood(ph_df, ph_prediction, pb_df, pb_prediction):
    """Heteroscedastic Gaussian likelihood using supplied observation SDs."""
    channels = (
        (
            "pH", ph_df, ph_prediction,
            "pH_obs", "pH_measurement_sd",
        ),
        (
            "TotalPb", pb_df, pb_prediction,
            "TotalPb_obs_mol_m3_bulk", "TotalPb_measurement_sd",
        ),
    )
    components = {}
    chi_square_components = {}
    n_observations = 0
    for name, frame, prediction, observation_column, sigma_column in channels:
        observation = frame[observation_column].to_numpy(float)
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        sigma = frame[sigma_column].to_numpy(float)
        if len(observation) != len(prediction):
            raise ValueError(f"{name} observation/prediction length mismatch")
        if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise ValueError(f"{name} standard deviations must be finite and positive")
        standardized_residual = (observation - prediction) / sigma
        chi_square = float(np.sum(standardized_residual ** 2))
        components[name] = float(
            chi_square + np.sum(np.log(2.0 * np.pi * sigma ** 2))
        )
        chi_square_components[name] = chi_square
        n_observations += len(observation)
    total = float(sum(components.values()))
    total_chi_square = float(sum(chi_square_components.values()))
    return {
        "neg2loglik": total,
        "mean_neg2loglik": total / max(n_observations, 1),
        "chi_square": total_chi_square,
        "reduced_chi_square": total_chi_square / max(n_observations, 1),
        "n_observations": int(n_observations),
        "components": components,
        "chi_square_components": chi_square_components,
        "error_model": "independent Gaussian errors with observation SDs supplied in the input tables",
    }


def iterations(args):
    if args.smoke:
        return {"electric": 3, "water": 3, "acid": 5, "pb": 5, "joint": 5}
    return {"electric": args.electric_iters, "water": args.water_iters,
            "acid": args.acid_iters, "pb": args.pb_iters, "joint": args.joint_iters}


def sample_feed(cases, rng, n_res, n_face, n_ic):
    feed = {}
    for case in cases:
        feed[case.t_res] = rng.uniform(0.0, 5.0, (n_res, 1)).astype(np.float32)
        feed[case.x_res] = rng.uniform(0.0, 0.2, (n_res, 1)).astype(np.float32)
        feed[case.t_ic] = np.zeros((n_ic, 1), np.float32)
        feed[case.x_ic] = rng.uniform(0.0, 0.2, (n_ic, 1)).astype(np.float32)
        tb = rng.uniform(0.0, 5.0, (n_face, 1)).astype(np.float32)
        feed[case.t_left], feed[case.x_left] = tb, np.zeros_like(tb)
        feed[case.t_right], feed[case.x_right] = tb, np.full_like(tb, 0.2)
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--output-root", type=Path, default=HERE / "outputs")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-res", type=int, default=3000)
    ap.add_argument("--n-face", type=int, default=500)
    ap.add_argument("--n-ic", type=int, default=500)
    ap.add_argument("--data-stride", type=int, default=1)
    ap.add_argument("--electric-iters", type=int, default=500)
    ap.add_argument("--water-iters", type=int, default=800)
    ap.add_argument("--acid-iters", type=int, default=2500)
    ap.add_argument("--pb-iters", type=int, default=3000)
    ap.add_argument("--joint-iters", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--joint-learning-rate", type=float, default=2e-5,
                    help="Learning rate for state-network updates in the joint stage.")
    ap.add_argument("--pb-acid-learning-rate", type=float, default=5e-5,
                    help="Learning rate for acid-state adaptation during the coupled chemistry stage.")
    ap.add_argument("--joint-parameter-learning-rate", type=float, default=1e-4,
                    help="Learning rate for the three physical parameters during alternating joint updates.")
    ap.add_argument("--warm-lr-decay-steps", type=int, default=2000,
                    help="Optimizer updates between staircase learning-rate decays in warm-up stages; 0 disables decay.")
    ap.add_argument("--coupled-lr-decay-steps", type=int, default=2000,
                    help="Optimizer updates between staircase learning-rate decays in Pb/joint stages; 0 disables decay.")
    ap.add_argument("--lr-decay-rate", type=float, default=0.5,
                    help="Multiplicative staircase learning-rate decay factor.")
    ap.add_argument("--block-coordinate-curriculum", action="store_true",
                    help="Alternate state-only and physics-parameter-only blocks to limit compensation.")
    ap.add_argument("--warm-parameter-learning-rate", type=float, default=1e-4,
                    help="Learning rate for isolated Keos/RH updates in the water/acid curriculum.")
    ap.add_argument("--keos-parameter-learning-rate", type=float,
                    help="Override the isolated Keos learning rate.")
    ap.add_argument("--rh-parameter-learning-rate", type=float,
                    help="Override the isolated RH learning rate.")
    ap.add_argument("--ph50-parameter-learning-rate", type=float,
                    help="Override the isolated pH50 learning rate.")
    ap.add_argument("--water-state-pretrain-iters", type=int, default=500)
    ap.add_argument("--acid-state-pretrain-iters", type=int, default=1000)
    ap.add_argument("--pb-state-pretrain-iters", type=int, default=500)
    ap.add_argument("--joint-state-pretrain-iters", type=int, default=200)
    ap.add_argument("--alternating-block-size", type=int, default=100)
    ap.add_argument("--max-numerical-recoveries", type=int, default=5,
                    help="Bad collocation batches skipped after safe-state rollback before aborting a stage.")
    ap.add_argument("--joint-state-steps", type=int, default=1)
    ap.add_argument("--joint-parameter-steps", type=int, default=1)
    ap.add_argument("--joint-loss-mode", choices=("scaled", "balanced", "raw"), default="scaled",
                    help="Use frozen-scale component balancing, legacy log1p balancing, or the raw sum.")
    ap.add_argument("--balance-scale-floor", type=float, default=1e-3,
                    help="Minimum frozen warm-up scale for a joint physics component.")
    ap.add_argument("--balance-scales-json", type=Path,
                    help="Reuse frozen physics scales from a baseline run for comparable diagnostic scores.")
    ap.add_argument("--pH-weight", type=float, default=10.0)
    ap.add_argument("--Pb-weight", type=float, default=5.0)
    ap.add_argument("--faraday-weight", type=float, default=100.0)
    ap.add_argument("--acid-bc-weight", type=float, default=float(CFG["FIELDS"]["HPLUS"]["BOUNDARY"].get("pH_dirichlet_weight", 30.0)))
    ap.add_argument("--acid-relative-weight", type=float, default=2.0,
                    help="Weight on the scale-free acid balance residual used to strengthen RH sensitivity.")
    ap.add_argument("--acid-relative-floor", type=float, default=0.05,
                    help="Denominator floor as a fraction of the acid residual reference scale.")
    ap.add_argument("--mass-weight", type=float, default=50.0)
    ap.add_argument("--mass-nx", type=int, default=21)
    ap.add_argument("--mass-nt", type=int, default=21)
    ap.add_argument("--guard-every", type=int, default=150)
    ap.add_argument("--pb-patience", type=int, default=5)
    ap.add_argument("--pb-min-delta", type=float, default=1e-4)
    ap.add_argument("--joint-patience", type=int, default=5)
    ap.add_argument("--joint-min-delta", type=float, default=1e-4)
    ap.add_argument("--adaptive-stop", action="store_true",
                    help="Stop a stage only after its fixed-monitor objective and active parameters are stable.")
    ap.add_argument("--monitor-every", type=int, default=100)
    ap.add_argument("--stability-window", type=int, default=5)
    ap.add_argument("--stability-score-rtol", type=float, default=2e-3)
    ap.add_argument("--stability-parameter-rtol", type=float, default=2e-3)
    ap.add_argument("--min-electric-iters", type=int, default=500)
    ap.add_argument("--min-water-iters", type=int, default=800)
    ap.add_argument("--min-acid-iters", type=int, default=2500)
    ap.add_argument("--min-pb-iters", type=int, default=1000)
    ap.add_argument("--min-joint-iters", type=int, default=100)
    ap.add_argument("--selection-physics-weight", type=float, default=0.5,
                    help="Weight on the fixed-collocation, physics-only scaled score during checkpoint selection.")
    ap.add_argument(
        "--selection-burn-in-iters", type=int, default=0,
        help=(
            "Do not admit a resumed stage-entry state or early updates into the best-checkpoint "
            "guard until this many stage iterations have completed."
        ),
    )
    ap.add_argument(
        "--selection-data-mode",
        choices=("standardized-rmse", "gaussian-likelihood"),
        default="standardized-rmse",
        help=(
            "Checkpoint data score. Gaussian likelihood evaluates every calibration "
            "row using the observation SD supplied in the input table."
        ),
    )
    ap.add_argument("--adsorption-slope-n", type=float, default=4.0 * 0.27 / np.log(10.0))
    ap.add_argument("--init-pH50", type=float, default=7.0)
    ap.add_argument("--init-Keos", type=float, default=0.5)
    ap.add_argument("--init-RH", type=float, default=35.0)
    ap.add_argument("--ph50-lower", type=float, default=1.5)
    ap.add_argument("--ph50-upper", type=float, default=10.0)
    ap.add_argument("--fixed-keos", type=float)
    ap.add_argument("--fixed-rh", type=float)
    ap.add_argument("--fixed-ph50", type=float)
    ap.add_argument("--run-role", choices=("inversion", "diagnostic"), default="inversion")
    ap.add_argument("--run-label", default="")
    ap.add_argument("--resume-checkpoint", type=Path,
                    help="Restore model variables from a TensorFlow checkpoint prefix before training.")
    ap.add_argument(
        "--resume-network-only", action="store_true",
        help=(
            "Restore only state-network variables from --resume-checkpoint and retain "
            "the requested physical-parameter initial values. Required for genuine "
            "multi-start inverse diagnostics."
        ),
    )
    ap.add_argument("--start-stage", choices=("electric", "water", "acid", "pb", "joint"),
                    default="electric")
    ap.add_argument("--profile-rh-values", default="",
                    help="Comma-separated fixed-state RH objective slice evaluated after training.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Write metrics/history only; skip checkpoints and prediction tables.")
    args = ap.parse_args()

    if args.selection_physics_weight < 0.0:
        ap.error("--selection-physics-weight must be non-negative")
    if args.selection_burn_in_iters < 0:
        ap.error("--selection-burn-in-iters must be non-negative")
    if args.balance_scale_floor <= 0.0:
        ap.error("--balance-scale-floor must be positive")
    if args.acid_relative_weight < 0.0 or args.acid_relative_floor <= 0.0:
        ap.error("acid relative weight must be non-negative and its floor must be positive")
    if args.ph50_lower >= args.ph50_upper:
        ap.error("--ph50-lower must be smaller than --ph50-upper")
    if args.monitor_every < 1 or args.stability_window < 2:
        ap.error("monitor interval must be >= 1 and stability window must be >= 2")
    if args.stability_score_rtol < 0.0 or args.stability_parameter_rtol < 0.0:
        ap.error("stability tolerances must be non-negative")
    if args.joint_state_steps < 1 or args.joint_parameter_steps < 0:
        ap.error("joint state steps must be >= 1 and parameter steps must be >= 0")
    if args.warm_lr_decay_steps < 0 or args.coupled_lr_decay_steps < 0:
        ap.error("learning-rate decay steps must be non-negative")
    if not 0.0 < args.lr_decay_rate <= 1.0:
        ap.error("--lr-decay-rate must be in (0, 1]")
    curriculum_pretrains = (
        args.water_state_pretrain_iters, args.acid_state_pretrain_iters,
        args.pb_state_pretrain_iters, args.joint_state_pretrain_iters,
    )
    if min(curriculum_pretrains) < 0 or args.alternating_block_size < 1:
        ap.error("curriculum pretraining iterations must be non-negative and block size >= 1")
    if args.max_numerical_recoveries < 0:
        ap.error("--max-numerical-recoveries must be non-negative")
    parameter_learning_rates = (
        args.warm_parameter_learning_rate,
        args.keos_parameter_learning_rate,
        args.rh_parameter_learning_rate,
        args.ph50_parameter_learning_rate,
    )
    if args.warm_parameter_learning_rate <= 0.0 or any(
            value is not None and value <= 0.0 for value in parameter_learning_rates[1:]):
        ap.error("curriculum parameter learning rates must be positive")

    keos_parameter_learning_rate = (
        args.keos_parameter_learning_rate or args.warm_parameter_learning_rate
    )
    rh_parameter_learning_rate = (
        args.rh_parameter_learning_rate or args.warm_parameter_learning_rate
    )
    ph50_parameter_learning_rate = (
        args.ph50_parameter_learning_rate or args.warm_parameter_learning_rate
    )

    if args.smoke:
        args.width, args.depth = 8, 2
        args.n_res, args.n_face, args.n_ic = 64, 32, 32
        args.mass_nx, args.mass_nt = 7, 7
        args.data_stride = max(args.data_stride, 25)

    for name in ("pH_calibration.csv", "TotalPb_calibration.csv", "pH_validation.csv", "TotalPb_validation.csv"):
        if not (args.data_dir / name).exists():
            raise FileNotFoundError(f"Observation file not found: {args.data_dir / name}")

    CFG["DOMAIN"]["tmax"] = 5.0
    DOMAIN["tmax"] = 5.0
    NETC["width"], NETC["hidden_layers"] = args.width, args.depth
    tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)

    with tf.compat.v1.variable_scope("inverse_parameters"):
        Keos, keos_vars = inversion_parameter("Keos", args.init_Keos, 0.05, 3.0, args.fixed_keos)
        RH, rh_vars = inversion_parameter("RH", args.init_RH, 1.0, 70.0, args.fixed_rh)
        pH50, ph50_vars = inversion_parameter(
            "pH50", args.init_pH50, args.ph50_lower, args.ph50_upper, args.fixed_ph50
        )
    parameter_vars = keos_vars + rh_vars + ph50_vars
    parameter_tensors = [Keos, RH, pH50]

    layers = [2] + [args.width] * args.depth + [1]
    cases = [
        CaseGraph(
            name, meta, layers, (Keos, RH, pH50),
            read_case_data(args.data_dir, name, args.data_stride), args,
            evaluation_data=read_case_data(args.data_dir, name, 1),
        )
        for name, meta in CASES.items()
    ]

    electric_loss = tf.add_n([c.loss_electric_pre for c in cases])
    water_loss = tf.add_n([c.loss_water for c in cases])
    acid_loss = tf.add_n([c.loss_acid_warm for c in cases])
    # Coupled chemistry alternates two safe updates.  The acid state follows the
    # coupled acid residual, while the Pb state plus RH/pH50 use the stable Pb +
    # uncoupled-acid objective.  This avoids differentiating the nonsmooth Pb
    # mass integral through the acid network (a source of NaN gradients), while
    # still allowing the acid state to adapt between parameter updates.
    pb_loss = tf.add_n([c.loss_pb + c.loss_acid_warm for c in cases])
    pb_coupled_acid_loss = tf.add_n([c.loss_acid for c in cases])
    joint_loss = tf.add_n([c.loss_total for c in cases])
    joint_loss_balanced = tf.add_n([
        tf.math.log1p(tf.maximum(component, 0.0))
        for case in cases
        for component in (case.loss_electric, case.loss_water, case.loss_acid, case.loss_pb)
    ])

    physics_components = {}
    data_components = {}
    for case in cases:
        physics_components.update({
            f"{case.name}.electric": case.loss_electric,
            f"{case.name}.water": case.loss_water,
            f"{case.name}.acid_pde": case.loss_acid_pde,
            f"{case.name}.acid_relative_pde": case.loss_acid_relative_pde,
            f"{case.name}.acid_boundary": case.loss_acid_bc,
            f"{case.name}.pb_transport": case.loss_pb_transport,
            f"{case.name}.pb_initial": case.loss_pb_ic,
            f"{case.name}.pb_boundary": case.loss_pb_boundary,
            f"{case.name}.pb_mass": case.loss_pb_mass,
        })
        # These are already standardized mean-square errors, so a target scale
        # of one has a direct statistical interpretation.
        data_components.update({
            f"{case.name}.pH_data": case.loss_pH_data,
            f"{case.name}.Pb_data": case.loss_Pb_data,
        })

    balance_scale_vars = {}
    balance_scale_inputs = {}
    balance_scale_assignments = []
    with tf.compat.v1.variable_scope("joint_balance_scales"):
        for key in physics_components:
            safe_key = key.replace(".", "_")
            scale = tf.compat.v1.get_variable(
                safe_key, shape=(), dtype=tf.float32,
                initializer=tf.compat.v1.constant_initializer(1.0), trainable=False,
            )
            scale_input = tf.compat.v1.placeholder(tf.float32, shape=(), name=f"{safe_key}_input")
            balance_scale_vars[key] = scale
            balance_scale_inputs[key] = scale_input
            balance_scale_assignments.append(scale.assign(scale_input))

    scaled_physics_terms = [
        tf.math.log1p(tf.maximum(component, 0.0) / balance_scale_vars[key])
        for key, component in physics_components.items()
    ]
    scaled_data_terms = [
        tf.math.log1p(tf.maximum(component, 0.0))
        for component in data_components.values()
    ]
    joint_loss_scaled = tf.add_n(scaled_physics_terms + scaled_data_terms) / float(
        len(scaled_physics_terms) + len(scaled_data_terms)
    )
    joint_physics_scaled = tf.add_n(scaled_physics_terms) / float(len(scaled_physics_terms))
    pb_acid_state_terms = []
    for case in cases:
        pb_acid_state_terms.extend([
            tf.math.log1p(
                tf.maximum(case.loss_acid_pde, 0.0)
                / balance_scale_vars[f"{case.name}.acid_pde"]
            ),
            tf.math.log1p(
                tf.maximum(case.loss_acid_relative_pde, 0.0)
                / balance_scale_vars[f"{case.name}.acid_relative_pde"]
            ),
            tf.math.log1p(
                tf.maximum(case.loss_acid_bc, 0.0)
                / balance_scale_vars[f"{case.name}.acid_boundary"]
            ),
            tf.math.log1p(tf.maximum(case.loss_pH_data, 0.0)),
        ])
    pb_acid_state_objective = tf.add_n(pb_acid_state_terms) / float(len(pb_acid_state_terms))
    joint_objective = {
        "scaled": joint_loss_scaled,
        "balanced": joint_loss_balanced,
        "raw": joint_loss,
    }[args.joint_loss_mode]
    electric_vars = sum((c.phi_vars for c in cases), []) + parameter_vars
    water_state_vars = sum((c.water_vars for c in cases), [])
    acid_state_vars = sum((c.acid_vars for c in cases), [])
    water_vars = water_state_vars + parameter_vars
    acid_vars = acid_state_vars + parameter_vars
    pb_acid_state_vars = sum((c.acid_vars for c in cases), [])
    pb_state_vars = sum((c.pb_vars for c in cases), [])
    pb_transport_vars = pb_state_vars + rh_vars + ph50_vars
    state_vars = sum((c.phi_vars + c.water_vars + c.acid_vars + c.pb_vars for c in cases), [])
    all_model_vars = state_vars + parameter_vars

    ops = {}
    ops["electric"] = make_optimizer(
        electric_loss, electric_vars, args.learning_rate, 5.0, "opt_electric",
        decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
    )
    curriculum_state_ops = {}
    curriculum_parameter_ops = {}
    if args.block_coordinate_curriculum:
        curriculum_state_ops["water"] = make_optimizer(
            water_loss, water_state_vars, args.learning_rate, 5.0, "opt_water_state",
            decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        curriculum_parameter_ops["water"] = make_optimizer(
            water_loss, keos_vars, keos_parameter_learning_rate, 1.0,
            "opt_water_parameter", clip_each=True,
            decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        curriculum_state_ops["acid"] = make_optimizer(
            acid_loss, acid_state_vars, args.learning_rate, 2.0, "opt_acid_state",
            decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        # Keos has already been isolated by the water equations.  Only RH is
        # released here; pH50 remains reserved for the Pb-observation stage.
        curriculum_parameter_ops["acid"] = make_optimizer(
            acid_loss, rh_vars, rh_parameter_learning_rate, 1.0,
            "opt_acid_parameter", clip_each=True,
            decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        ops["water"] = curriculum_state_ops["water"]
        ops["acid"] = curriculum_state_ops["acid"]
    else:
        ops["water"] = make_optimizer(
            water_loss, water_vars, args.learning_rate, 5.0, "opt_water",
            decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        ops["acid"] = make_optimizer(
            acid_loss, acid_vars, args.learning_rate, 2.0, "opt_acid",
            decay_steps=args.warm_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
    pb_acid_state_op = make_optimizer(
        pb_acid_state_objective, pb_acid_state_vars, args.pb_acid_learning_rate, 0.5,
        "opt_pb_acid_state", clip_each=True,
        decay_steps=args.coupled_lr_decay_steps, decay_rate=args.lr_decay_rate,
    )
    if args.block_coordinate_curriculum:
        pb_state_op = make_optimizer(
            pb_loss, pb_state_vars, args.learning_rate, 1.0,
            "opt_pb_state", clip_each=True,
            decay_steps=args.coupled_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        # RH is learned from the dense pH history.  The seven spatial locations
        # contain only four Pb calibration locations per case, all at day 5, so
        # the sparse Pb stage releases pH50 alone to prevent RH-pH50 compensation.
        pb_parameter_op = make_optimizer(
            pb_loss, ph50_vars, ph50_parameter_learning_rate, 1.0,
            "opt_pb_parameter", clip_each=True,
            decay_steps=args.coupled_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
        pb_transport_op = None
    else:
        pb_state_op = None
        pb_parameter_op = None
        pb_transport_op = make_optimizer(
            pb_loss, pb_transport_vars, args.learning_rate, 1.0,
            "opt_pb_transport", clip_each=True,
            decay_steps=args.coupled_lr_decay_steps, decay_rate=args.lr_decay_rate,
        )
    joint_state_op = make_optimizer(
        joint_objective, state_vars, args.joint_learning_rate, 1.0,
        "opt_joint_state", clip_each=True,
        decay_steps=args.coupled_lr_decay_steps, decay_rate=args.lr_decay_rate,
    )
    joint_parameter_op = make_optimizer(
        joint_objective, parameter_vars, args.joint_parameter_learning_rate, 1.0,
        "opt_joint_parameters", clip_each=True,
        decay_steps=args.coupled_lr_decay_steps, decay_rate=args.lr_decay_rate,
    )

    parameter_names = {v.name for v in parameter_vars}
    stage_parameter_gradients = {
        stage: [v.name for v in active_vars if v.name in parameter_names]
        for stage, (_, _, active_vars) in ops.items()
    }
    if args.block_coordinate_curriculum:
        stage_parameter_gradients["water"] = [
            v.name for v in curriculum_parameter_ops["water"][2]
            if v.name in parameter_names
        ]
        stage_parameter_gradients["acid"] = [
            v.name for v in curriculum_parameter_ops["acid"][2]
            if v.name in parameter_names
        ]
        stage_parameter_gradients["pb"] = [
            v.name for v in pb_parameter_op[2] if v.name in parameter_names
        ]
    else:
        stage_parameter_gradients["pb"] = [
            v.name for v in pb_transport_op[2] if v.name in parameter_names
        ]
    stage_parameter_gradients["joint"] = [
        v.name for v in joint_parameter_op[2] if v.name in parameter_names
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    outdir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=False)
    run_config = dict(vars(args))
    run_config["evaluation_data_stride"] = 1
    run_config["inversion_version"] = "baseline_3param_v5"
    run_config["effective_iterations"] = iterations(args)
    run_config["shared_trainable_parameters"] = [v.name for v in parameter_vars]
    run_config["stage_parameter_gradients"] = stage_parameter_gradients
    run_config["pb_stage"] = (
        "block-coordinate acid/Pb state updates alternating with isolated pH50 updates; "
        "Keos and RH frozen" if args.block_coordinate_curriculum else
        "alternating coupled-acid state and Pb+acid-warm transport/parameter updates; "
        "RH and pH50 trainable; Keos fixed"
    )
    run_config["acid_Pb_source_coupling"] = "Picard value coupling with stopped source gradient"
    run_config["pb_acid_state_objective"] = (
        "scaled acid PDE + scaled relative acid balance + scaled acid boundary + standardized pH data"
    )
    run_config["rh_sensitivity_strengthening"] = (
        "relative full acid balance residual with stopped adaptive denominator; original PDE zero set retained"
    )
    run_config["stopping_rule"] = (
        "adaptive fixed-monitor objective plus active-parameter stability with safety caps"
        if args.adaptive_stop else "iteration caps plus legacy PB/joint patience"
    )
    run_config["parameter_compensation_control"] = (
        "process curriculum plus state/parameter block-coordinate optimization; "
        "water->Keos, acid/pH->RH, sparse day-5 Pb->pH50, then low-rate joint refinement"
        if args.block_coordinate_curriculum else "joint state/parameter optimization"
    )
    run_config["effective_isolated_parameter_learning_rates"] = {
        "Keos": keos_parameter_learning_rate,
        "RH": rh_parameter_learning_rate,
        "pH50": ph50_parameter_learning_rate,
    }
    run_config["checkpoint_selection"] = "calibration_plus_fixed_collocation_scaled_physics_only"
    run_config["validation_usage"] = "final reporting only"
    (outdir / "run_config.json").write_text(json.dumps(run_config, default=str, indent=2), encoding="utf-8")
    model_saver = tf.compat.v1.train.Saver(var_list=all_model_vars, max_to_keep=6)
    network_saver = tf.compat.v1.train.Saver(var_list=state_vars)
    saver = None if args.summary_only else model_saver
    rng = np.random.default_rng(args.seed)
    history = []
    rh_objective_slice = []
    stage_iters = iterations(args)
    n_res = args.n_res
    min_stage_iters = {
        "electric": args.min_electric_iters,
        "water": args.min_water_iters,
        "acid": args.min_acid_iters,
        "pb": args.min_pb_iters,
        "joint": args.min_joint_iters,
    }
    if args.block_coordinate_curriculum:
        # A stage may not be declared stable while it is still in state-only
        # pretraining.  Require a full stability window after parameters have
        # first been released.
        curriculum_pretrain_limits = {
            "water": args.water_state_pretrain_iters,
            "acid": args.acid_state_pretrain_iters,
            "pb": args.pb_state_pretrain_iters,
            "joint": args.joint_state_pretrain_iters,
        }
        for stage, pretrain_iterations in curriculum_pretrain_limits.items():
            min_stage_iters[stage] = max(
                int(min_stage_iters[stage]),
                int(pretrain_iterations)
                + int(args.stability_window) * int(args.monitor_every),
            )
    stage_parameter_indices = {
        "electric": (),
        "water": (0,) if keos_vars else (),
        "acid": tuple(index for index, variables in enumerate((keos_vars, rh_vars, ph50_vars))
                      if (index == 1 if args.block_coordinate_curriculum else index < 2) and variables),
        "pb": tuple(index for index, variables in enumerate((keos_vars, rh_vars, ph50_vars))
                    if (index == 2 if args.block_coordinate_curriculum else index in (1, 2)) and variables),
        "joint": tuple(index for index, variables in enumerate((keos_vars, rh_vars, ph50_vars))
                       if variables),
    }
    stability_records = {stage: [] for stage in ("electric", "water", "acid", "pb", "joint")}
    stage_stop_status = {}
    warm_best = {
        stage: {"score": np.inf, "iteration": None, "values": None, "parameters": None}
        for stage in ("electric", "water", "acid")
    }

    def update_stability(stage, iteration, score, parameter_values):
        records = stability_records[stage]
        records.append((float(score), np.asarray(parameter_values, dtype=float)))
        records[:] = records[-int(args.stability_window):]
        stats = {"stable": False, "score_relative_span": None, "parameter_relative_span": None}
        if not args.adaptive_stop or iteration < int(min_stage_iters[stage]):
            return False, stats
        if len(records) < int(args.stability_window):
            return False, stats
        scores = np.asarray([record[0] for record in records], dtype=float)
        score_scale = max(abs(float(np.mean(scores))), 1e-12)
        score_span = float(np.ptp(scores) / score_scale)
        indices = stage_parameter_indices[stage]
        parameter_span = 0.0
        if indices:
            values = np.asarray([record[1][list(indices)] for record in records], dtype=float)
            scales = np.maximum(np.abs(np.mean(values, axis=0)), 1.0)
            parameter_span = float(np.max(np.ptp(values, axis=0) / scales))
        stable = (
            score_span <= float(args.stability_score_rtol)
            and parameter_span <= float(args.stability_parameter_rtol)
        )
        stats.update({
            "stable": bool(stable),
            "score_relative_span": score_span,
            "parameter_relative_span": parameter_span,
        })
        return bool(stable), stats

    with tf.compat.v1.Session(config=TF_CONFIG) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if args.resume_checkpoint is not None:
            checkpoint_prefix = str(args.resume_checkpoint)
            if not Path(checkpoint_prefix + ".index").exists():
                raise FileNotFoundError(f"Checkpoint prefix not found: {checkpoint_prefix}")
            (network_saver if args.resume_network_only else model_saver).restore(
                sess, checkpoint_prefix
            )
            print(json.dumps({"restored_training_checkpoint": checkpoint_prefix,
                              "network_only": bool(args.resume_network_only),
                              "start_stage": args.start_stage}))
        guard_feed = sample_feed(
            cases, np.random.default_rng(args.seed + 2), n_res, args.n_face, args.n_ic
        )
        best = {"score": np.inf, "stage": None, "iteration": None, "values": None, "parameters": None, "metrics": None}
        balance_scales = {}

        def initialize_balance_scales():
            """Freeze warm-up physics scales before any Pb checkpoint is ranked."""
            if args.balance_scales_json is not None:
                loaded = json.loads(args.balance_scales_json.read_text(encoding="utf-8"))
                loaded = loaded.get("balance_scales", loaded)
                missing = sorted(set(physics_components) - set(loaded))
                unsupported_missing = [key for key in missing if not key.endswith(".acid_relative_pde")]
                if unsupported_missing:
                    raise ValueError(
                        f"Missing balance scales in {args.balance_scales_json}: {unsupported_missing}"
                    )
                balance_scales.update({
                    key: max(float(loaded[key]), float(args.balance_scale_floor))
                    for key in physics_components if key in loaded
                })
                if missing:
                    missing_values = sess.run(
                        [physics_components[key] for key in missing], guard_feed
                    )
                    balance_scales.update({
                        key: max(float(value), float(args.balance_scale_floor))
                        for key, value in zip(missing, missing_values)
                    })
                scale_source = str(args.balance_scales_json)
            else:
                component_values = sess.run(list(physics_components.values()), guard_feed)
                balance_scales.update({
                    key: max(float(value), float(args.balance_scale_floor))
                    for key, value in zip(physics_components, component_values)
                })
                scale_source = "current_run_after_acid_warmup"
            assign_feed = {
                balance_scale_inputs[key]: value for key, value in balance_scales.items()
            }
            sess.run(balance_scale_assignments, assign_feed)
            run_config["balance_scales"] = balance_scales
            run_config["balance_scale_source"] = scale_source
            (outdir / "run_config.json").write_text(
                json.dumps(run_config, default=str, indent=2), encoding="utf-8"
            )
            print(json.dumps({"joint_balance_scales_frozen": True,
                              "source": scale_source, **balance_scales}))

        def calibration_metrics_and_score():
            """Calibration plus fixed physics score; validation remains untouched."""
            metrics_now = {}
            ph_frames, ph_predictions = [], []
            pb_frames, pb_predictions = [], []
            for case in cases:
                ph_pred, pb_pred = sess.run(
                    [case.ph_cal_full_pred, case.pb_cal_full_pred]
                )
                ph_err = np.asarray(ph_pred).reshape(-1) - case.ph_cal_full_df["pH_obs"].to_numpy(float)
                pb_err = np.asarray(pb_pred).reshape(-1) - case.pb_cal_full_df["TotalPb_obs_mol_m3_bulk"].to_numpy(float)
                ph_measurement_sd = case.ph_cal_full_df["pH_measurement_sd"].to_numpy(float)
                pb_measurement_sd = case.pb_cal_full_df["TotalPb_measurement_sd"].to_numpy(float)
                metrics_now[f"{case.name}_pH_calibration_RMSE"] = float(np.sqrt(np.mean(ph_err ** 2)))
                metrics_now[f"{case.name}_TotalPb_calibration_RMSE"] = float(np.sqrt(np.mean(pb_err ** 2)))
                metrics_now[f"{case.name}_pH_calibration_standardized_RMSE"] = float(
                    np.sqrt(np.mean((ph_err / ph_measurement_sd) ** 2))
                )
                metrics_now[f"{case.name}_TotalPb_calibration_standardized_RMSE"] = float(
                    np.sqrt(np.mean((pb_err / pb_measurement_sd) ** 2))
                )
                ph_frames.append(case.ph_cal_full_df)
                ph_predictions.append(np.asarray(ph_pred).reshape(-1))
                pb_frames.append(case.pb_cal_full_df)
                pb_predictions.append(np.asarray(pb_pred).reshape(-1))
            likelihood = gaussian_observation_likelihood(
                pd.concat(ph_frames, ignore_index=True), np.concatenate(ph_predictions),
                pd.concat(pb_frames, ignore_index=True), np.concatenate(pb_predictions),
            )
            metrics_now["calibration_neg2loglik"] = likelihood["neg2loglik"]
            metrics_now["calibration_mean_neg2loglik"] = likelihood["mean_neg2loglik"]
            ph_score = np.mean([v for k, v in metrics_now.items() if k.endswith("pH_calibration_standardized_RMSE")])
            pb_score = np.mean([v for k, v in metrics_now.items() if k.endswith("TotalPb_calibration_standardized_RMSE")])
            data_score = (
                float(likelihood["mean_neg2loglik"])
                if args.selection_data_mode == "gaussian-likelihood"
                else float(0.5 * (ph_score + pb_score))
            )
            physics_score = float(sess.run(joint_physics_scaled, guard_feed))
            metrics_now["selection_data_score"] = data_score
            metrics_now["selection_physics_objective"] = physics_score
            total_score = data_score + float(args.selection_physics_weight) * physics_score
            return metrics_now, total_score

        def remember_best(stage, iteration, min_delta=0.0):
            metrics_now, score = calibration_metrics_and_score()
            improved = False
            if score < best["score"] - float(min_delta):
                best.update({
                    "score": score,
                    "stage": stage,
                    "iteration": int(iteration),
                    "values": sess.run(all_model_vars),
                    "parameters": [float(v) for v in sess.run(parameter_tensors)],
                    "metrics": metrics_now,
                })
                print(json.dumps({"best_guard": True, "stage": stage, "iteration": int(iteration),
                                  "selection_basis": "calibration_plus_scaled_physics_only",
                                  "score": score, **metrics_now}))
                improved = True
            return improved, metrics_now, score

        def restore_best():
            if best["values"] is None:
                return
            sess.run([v.assign(value) for v, value in zip(all_model_vars, best["values"])])
            print(json.dumps({"restored_best_guard": True, "stage": best["stage"],
                              "iteration": best["iteration"], "score": best["score"],
                              "parameters": best["parameters"], "metrics": best["metrics"]}))

        def remember_warm_best(stage, iteration, score):
            """Keep the lowest fixed-monitor warm-up state, independent of random batches."""
            record = warm_best[stage]
            if not np.isfinite(score) or score >= record["score"]:
                return False
            record.update({
                "score": float(score),
                "iteration": int(iteration),
                "values": sess.run(all_model_vars),
                "parameters": [float(v) for v in sess.run(parameter_tensors)],
            })
            if saver is not None:
                saver.save(
                    sess, str(outdir / f"checkpoint_{stage}_best.ckpt"),
                    write_meta_graph=False,
                )
            print(json.dumps({
                "warm_best": True, "stage": stage, "iteration": int(iteration),
                "fixed_monitor_objective": float(score),
                "parameters": record["parameters"],
            }))
            return True

        def restore_warm_best(stage, reason):
            record = warm_best[stage]
            if record["values"] is None:
                return False
            sess.run([v.assign(value) for v, value in zip(all_model_vars, record["values"])])
            print(json.dumps({
                "restored_warm_best": True, "stage": stage, "reason": reason,
                "iteration": record["iteration"], "score": record["score"],
                "parameters": record["parameters"],
            }))
            return True

        stage_sequence = ("electric", "water", "acid", "pb", "joint")
        stage_sequence = stage_sequence[stage_sequence.index(args.start_stage):]
        curriculum_pretrain = {
            "water": args.water_state_pretrain_iters,
            "acid": args.acid_state_pretrain_iters,
            "pb": args.pb_state_pretrain_iters,
            "joint": args.joint_state_pretrain_iters,
        }
        for stage in stage_sequence:
            if stage in ("pb", "joint") and not balance_scales:
                initialize_balance_scales()
            # A continuous run already carries the best Pb guard into the joint
            # stage. A resumed run starts with an empty in-memory guard, so seed
            # it from the restored checkpoint before applying even one update.
            # This keeps resumed Pb/joint refinement monotone under the fixed
            # calibration-plus-physics selection score.
            if (stage in ("pb", "joint") and best["values"] is None
                    and args.selection_burn_in_iters == 0):
                remember_best(f"{stage}_entry", 0)
            if args.block_coordinate_curriculum and stage in ("water", "acid"):
                active_vars = (
                    list(curriculum_state_ops[stage][2])
                    + list(curriculum_parameter_ops[stage][2])
                )
            elif args.block_coordinate_curriculum and stage == "pb":
                active_vars = (
                    list(pb_acid_state_op[2]) + list(pb_state_op[2])
                    + list(pb_parameter_op[2])
                )
            elif stage == "joint":
                active_vars = list(joint_state_op[2]) + list(joint_parameter_op[2])
            elif stage == "pb":
                active_vars = list(pb_acid_state_op[2]) + list(pb_transport_op[2])
            else:
                _, _, active_vars = ops[stage]
            count = stage_iters[stage]
            print(f"\n== {stage.upper()} | iters={count} | active={len(active_vars)} | "
                  f"trainable shared parameters={len(stage_parameter_gradients[stage])} ==")
            raw_loss_tensor = {"electric": electric_loss, "water": water_loss, "acid": acid_loss,
                               "pb": pb_loss + pb_coupled_acid_loss, "joint": joint_loss}[stage]
            objective_tensor = joint_objective if stage == "joint" else raw_loss_tensor
            no_improve = 0
            stage_stopped_stable = False
            stage_numerical_abort = False
            numerical_recovery_count = 0
            last_iteration = 0
            stage_entry_values = sess.run(all_model_vars)
            monitor_interval = (
                max(1, int(args.monitor_every)) if args.adaptive_stop
                else max(1, count // 10)
            )
            for it in range(count):
                last_iteration = it + 1
                feed = sample_feed(cases, rng, n_res, args.n_face, args.n_ic)
                curriculum_phase = "joint_update"
                if args.block_coordinate_curriculum and stage in curriculum_pretrain:
                    if it < int(curriculum_pretrain[stage]):
                        curriculum_phase = "state_pretrain"
                    else:
                        block_index = (
                            (it - int(curriculum_pretrain[stage]))
                            // int(args.alternating_block_size)
                        )
                        curriculum_phase = "parameter" if block_index % 2 == 0 else "state"
                    # In profile fits the parameter assigned to this isolated
                    # stage may be fixed.  Use that block for state training
                    # instead of spending half the staged refit on a no-op.
                    if curriculum_phase == "parameter":
                        if (
                            stage in ("water", "acid")
                            and not curriculum_parameter_ops[stage][2]
                        ):
                            curriculum_phase = "state"
                        elif stage == "pb" and not pb_parameter_op[2]:
                            curriculum_phase = "state"
                        elif stage == "joint" and not joint_parameter_op[2]:
                            curriculum_phase = "state"
                try:
                    if args.block_coordinate_curriculum and stage in ("water", "acid"):
                        selected_op = (
                            curriculum_parameter_ops[stage]
                            if curriculum_phase == "parameter"
                            else curriculum_state_ops[stage]
                        )
                        sess.run(selected_op[0], feed)
                    elif args.block_coordinate_curriculum and stage == "pb":
                        if curriculum_phase == "parameter":
                            sess.run(pb_parameter_op[0], feed)
                        else:
                            sess.run(pb_acid_state_op[0], feed)
                            sess.run(pb_state_op[0], feed)
                    elif args.block_coordinate_curriculum and stage == "joint":
                        if curriculum_phase == "parameter":
                            for _ in range(args.joint_parameter_steps):
                                sess.run(joint_parameter_op[0], feed)
                        else:
                            for _ in range(args.joint_state_steps):
                                sess.run(joint_state_op[0], feed)
                    elif stage == "joint":
                        for _ in range(args.joint_state_steps):
                            sess.run(joint_state_op[0], feed)
                        for _ in range(args.joint_parameter_steps):
                            sess.run(joint_parameter_op[0], feed)
                    elif stage == "pb":
                        sess.run(pb_acid_state_op[0], feed)
                        sess.run(pb_transport_op[0], feed)
                    else:
                        sess.run(ops[stage][0], feed)
                except tf.errors.InvalidArgumentError as exc:
                    if not args.adaptive_stop:
                        raise
                    recovered_from = None
                    if stage in warm_best and restore_warm_best(stage, "non_finite_gradient"):
                        recovered_from = "warm_stage_best"
                    elif stage in ("pb", "joint") and best["values"] is not None:
                        restore_best()
                        recovered_from = "coupled_guard_best"
                    else:
                        sess.run([v.assign(value) for v, value in zip(all_model_vars, stage_entry_values)])
                        recovered_from = "stage_entry"
                    print(json.dumps({
                        "numerical_recovery": True, "stage": stage,
                        "iteration": it + 1, "recovered_from": recovered_from,
                        "recovery_count": numerical_recovery_count + 1,
                        "error": str(exc).splitlines()[0],
                    }))
                    numerical_recovery_count += 1
                    if numerical_recovery_count <= int(args.max_numerical_recoveries):
                        # A check-numerics failure occurs before Adam applies the
                        # update.  Roll back the model, discard this random batch,
                        # and keep training so one pathological collocation draw
                        # cannot truncate parameter exploration.
                        continue
                    print(json.dumps({
                        "numerical_recovery_exhausted": True, "stage": stage,
                        "iteration": it + 1,
                        "max_numerical_recoveries": int(args.max_numerical_recoveries),
                    }))
                    stage_numerical_abort = True
                    break
                log_now = it == 0 or (it + 1) % monitor_interval == 0 or it + 1 == count
                if log_now:
                    if stage == "joint":
                        if args.block_coordinate_curriculum:
                            selected_op = (
                                joint_parameter_op if curriculum_phase == "parameter"
                                else joint_state_op
                            )
                            objective_value, loss_value, active_grad_value, values = sess.run(
                                [objective_tensor, raw_loss_tensor, selected_op[1], parameter_tensors], feed
                            )
                            state_grad_value = (
                                0.0 if curriculum_phase == "parameter" else active_grad_value
                            )
                            parameter_grad_value = (
                                active_grad_value if curriculum_phase == "parameter" else 0.0
                            )
                        else:
                            objective_value, loss_value, state_grad_value, parameter_grad_value, values = sess.run(
                                [objective_tensor, raw_loss_tensor, joint_state_op[1],
                                 joint_parameter_op[1], parameter_tensors], feed
                            )
                        grad_value = float(math.hypot(state_grad_value, parameter_grad_value))
                    elif stage == "pb":
                        if args.block_coordinate_curriculum:
                            if curriculum_phase == "parameter":
                                objective_value, loss_value, parameter_grad_value, values = sess.run(
                                    [objective_tensor, raw_loss_tensor, pb_parameter_op[1], parameter_tensors], feed
                                )
                                state_grad_value = 0.0
                            else:
                                objective_value, loss_value, acid_grad_value, pb_grad_value, values = sess.run(
                                    [objective_tensor, raw_loss_tensor, pb_acid_state_op[1],
                                     pb_state_op[1], parameter_tensors], feed
                                )
                                state_grad_value = float(math.hypot(acid_grad_value, pb_grad_value))
                                parameter_grad_value = 0.0
                        else:
                            objective_value, loss_value, acid_grad_value, transport_grad_value, values = sess.run(
                                [objective_tensor, raw_loss_tensor, pb_acid_state_op[1],
                                 pb_transport_op[1], parameter_tensors], feed
                            )
                            state_grad_value, parameter_grad_value = acid_grad_value, transport_grad_value
                        grad_value = float(math.hypot(state_grad_value, parameter_grad_value))
                    else:
                        if args.block_coordinate_curriculum and stage in ("water", "acid"):
                            selected_op = (
                                curriculum_parameter_ops[stage]
                                if curriculum_phase == "parameter"
                                else curriculum_state_ops[stage]
                            )
                            objective_value, loss_value, grad_value, values = sess.run(
                                [objective_tensor, raw_loss_tensor, selected_op[1], parameter_tensors], feed
                            )
                            state_grad_value = (
                                0.0 if curriculum_phase == "parameter" else float(grad_value)
                            )
                            parameter_grad_value = (
                                float(grad_value) if curriculum_phase == "parameter" else 0.0
                            )
                        else:
                            objective_value, loss_value, grad_value, values = sess.run(
                                [objective_tensor, raw_loss_tensor, ops[stage][1], parameter_tensors], feed
                            )
                            state_grad_value, parameter_grad_value = float(grad_value), 0.0
                    if (not np.isfinite(objective_value) or not np.isfinite(loss_value)
                            or not np.isfinite(values).all()):
                        raise FloatingPointError(
                            f"Non-finite state at {stage} iteration {it + 1}: "
                            f"objective={objective_value}, loss={loss_value}, parameters={values}"
                        )
                    row = {"stage": stage, "iteration": it + 1, "loss": float(loss_value),
                           "optimization_loss": float(objective_value),
                           "grad_norm": float(grad_value), "Keos": float(values[0]),
                           "RH": float(values[1]), "pH50": float(values[2])}
                    if stage == "joint":
                        row["state_grad_norm"] = float(state_grad_value)
                        row["parameter_grad_norm"] = float(parameter_grad_value)
                    elif stage == "pb":
                        row["acid_state_grad_norm"] = float(state_grad_value)
                        row["pb_transport_grad_norm"] = float(parameter_grad_value)
                    if args.block_coordinate_curriculum and stage != "electric":
                        row["curriculum_phase"] = curriculum_phase
                        row["state_grad_norm"] = float(state_grad_value)
                        row["parameter_grad_norm"] = float(parameter_grad_value)
                    history.append(row)
                    print(json.dumps(row))
                    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
                    if stage == "joint":
                        selection_eligible = (
                            it + 1 >= int(args.selection_burn_in_iters)
                            and (
                                not args.block_coordinate_curriculum
                                or curriculum_phase == "state"
                            )
                        )
                        if selection_eligible:
                            improved, _, selection_score = remember_best(
                                stage, it + 1, args.joint_min_delta
                            )
                        else:
                            _, selection_score = calibration_metrics_and_score()
                            improved = False
                            print(json.dumps({
                                "selection_deferred": True, "stage": stage,
                                "iteration": it + 1,
                                "eligible_after_iteration": int(args.selection_burn_in_iters),
                                "curriculum_phase": curriculum_phase,
                                "selection_score": float(selection_score),
                            }))
                        if improved:
                            no_improve = 0
                        else:
                            no_improve += 1
                        if selection_eligible:
                            stability_score = float(best["score"])
                            stability_parameters = np.asarray(best["parameters"], dtype=float)
                            stable, stability_stats = update_stability(
                                stage, it + 1, stability_score, stability_parameters
                            )
                        else:
                            stable = False
                            stability_stats = {"stability_ready": False}
                        if args.adaptive_stop:
                            print(json.dumps({
                                "adaptive_monitor": True, "stage": stage,
                                "iteration": it + 1, "selection_score": selection_score,
                                "guarded_best_score": (
                                    None if best["values"] is None else float(best["score"])
                                ),
                                **stability_stats,
                            }))
                        if args.adaptive_stop and stable:
                            print(json.dumps({
                                "stable_stop": True, "stage": stage, "iteration": it + 1,
                                "selection_basis": "calibration_plus_scaled_physics_only",
                                **stability_stats,
                            }))
                            stage_stopped_stable = True
                            break
                        if (not args.adaptive_stop and args.joint_patience > 0
                                and no_improve >= args.joint_patience):
                            print(json.dumps({"early_stop": True, "stage": stage, "iteration": it + 1,
                                              "selection_basis": "calibration_plus_scaled_physics_only",
                                              "best_stage": best["stage"], "best_iteration": best["iteration"],
                                              "best_score": best["score"]}))
                            break
                    elif stage in ("electric", "water", "acid") and args.adaptive_stop:
                        fixed_monitor_objective = float(sess.run(objective_tensor, guard_feed))
                        remember_warm_best(stage, it + 1, fixed_monitor_objective)
                        stable, stability_stats = update_stability(
                            stage, it + 1, fixed_monitor_objective, values
                        )
                        print(json.dumps({
                            "adaptive_monitor": True, "stage": stage,
                            "iteration": it + 1,
                            "fixed_monitor_objective": fixed_monitor_objective,
                            **stability_stats,
                        }))
                        if stable:
                            print(json.dumps({
                                "stable_stop": True, "stage": stage, "iteration": it + 1,
                                "selection_basis": "fixed_monitor_objective_plus_parameter_stability",
                                **stability_stats,
                            }))
                            stage_stopped_stable = True
                            break
                pb_guard_interval = (
                    max(1, int(args.monitor_every)) if args.adaptive_stop
                    else max(1, int(args.guard_every))
                )
                if stage == "pb" and ((it + 1) % pb_guard_interval == 0 or it + 1 == count):
                    improved, _, selection_score = remember_best(stage, it + 1, args.pb_min_delta)
                    if improved:
                        no_improve = 0
                    else:
                        no_improve += 1
                    values = sess.run(parameter_tensors)
                    stable, stability_stats = update_stability(
                        stage, it + 1, selection_score, values
                    )
                    if args.adaptive_stop:
                        print(json.dumps({
                            "adaptive_monitor": True, "stage": stage,
                            "iteration": it + 1, "selection_score": selection_score,
                            **stability_stats,
                        }))
                    if args.adaptive_stop and stable:
                        print(json.dumps({
                            "stable_stop": True, "stage": stage, "iteration": it + 1,
                            "selection_basis": "calibration_plus_scaled_physics_only",
                            **stability_stats,
                        }))
                        stage_stopped_stable = True
                        break
                    if (not args.adaptive_stop and args.pb_patience > 0
                            and no_improve >= args.pb_patience):
                        print(json.dumps({"early_stop": True, "stage": stage, "iteration": it + 1,
                                          "selection_basis": "calibration_plus_scaled_physics_only",
                                          "best_stage": best["stage"], "best_iteration": best["iteration"],
                                          "best_score": best["score"]}))
                        break
            if args.adaptive_stop and stage in warm_best:
                restore_warm_best(
                    stage,
                    "stable_stop" if stage_stopped_stable
                    else "numerical_abort" if stage_numerical_abort
                    else "safety_cap",
                )
            if (args.adaptive_stop and count > 0 and not stage_stopped_stable
                    and not stage_numerical_abort):
                print(json.dumps({
                    "safety_cap_reached": True, "stage": stage, "iteration": count,
                    "message": "stage reached its safety cap before satisfying stability criteria",
                }))
            stage_stop_status[stage] = {
                "iteration": int(last_iteration),
                "reason": (
                    "stable" if stage_stopped_stable
                    else "numerical_abort" if stage_numerical_abort
                    else "safety_cap" if args.adaptive_stop and count > 0
                    else "zero_iterations" if count == 0
                    else "completed_or_legacy_patience"
                ),
                "numerical_recoveries": int(numerical_recovery_count),
            }
            pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
            if stage == "pb":
                restore_best()
            if saver is not None:
                saver.save(sess, str(outdir / f"checkpoint_{stage}.ckpt"), write_meta_graph=(stage == "electric"))

        if best["values"] is None:
            remember_best("restored", 0)
        restore_best()
        final_values = sess.run(parameter_tensors)
        final_feed = sample_feed(cases, np.random.default_rng(args.seed + 1), n_res, args.n_face, args.n_ic)
        final_joint_loss, final_joint_objective, final_physics_objective = [float(v) for v in sess.run(
            [joint_loss, joint_objective, joint_physics_scaled], final_feed
        )]
        final_loss_breakdown = {}
        for case in cases:
            names = tuple(case.diagnostics)
            values = sess.run([case.diagnostics[name] for name in names], final_feed)
            final_loss_breakdown[case.name] = {name: float(value) for name, value in zip(names, values)}
        if saver is not None:
            saver.save(sess, str(outdir / "model.ckpt"))
        metrics = {}
        likelihood_inputs = {
            "calibration": {"ph_frames": [], "ph_predictions": [], "pb_frames": [], "pb_predictions": []},
            "validation": {"ph_frames": [], "ph_predictions": [], "pb_frames": [], "pb_predictions": []},
        }
        for case in cases:
            ph_cal_pred, pb_cal_pred, ph_val_pred, pb_val_pred = sess.run(
                [case.ph_cal_full_pred, case.pb_cal_full_pred,
                 case.ph_val_full_pred, case.pb_val_full_pred]
            )
            for df, pred, name, obs_col, sigma_col in (
                (case.ph_cal_full_df, ph_cal_pred, "pH_calibration", "pH_obs", "pH_measurement_sd"),
                (case.pb_cal_full_df, pb_cal_pred, "TotalPb_calibration", "TotalPb_obs_mol_m3_bulk", "TotalPb_measurement_sd"),
                (case.ph_val_full_df, ph_val_pred, "pH_validation", "pH_obs", "pH_measurement_sd"),
                (case.pb_val_full_df, pb_val_pred, "TotalPb_validation", "TotalPb_obs_mol_m3_bulk", "TotalPb_measurement_sd"),
            ):
                result = df.copy()
                result["prediction"] = np.asarray(pred).reshape(-1)
                if not args.summary_only:
                    result.to_csv(outdir / f"{case.name}_{name}.csv", index=False)
                error = result["prediction"] - result[obs_col]
                metrics[f"{case.name}_{name}_RMSE"] = float(np.sqrt(np.mean(error ** 2)))
                metrics[f"{case.name}_{name}_standardized_RMSE"] = float(
                    np.sqrt(np.mean((error / result[sigma_col]) ** 2))
                )
                split = "calibration" if name.endswith("calibration") else "validation"
                channel = "ph" if name.startswith("pH_") else "pb"
                likelihood_inputs[split][f"{channel}_frames"].append(df)
                likelihood_inputs[split][f"{channel}_predictions"].append(
                    np.asarray(pred).reshape(-1)
                )

            profile = sess.run(case.profile_tensors)
            profile_df = pd.DataFrame({
                "time_day": 5.0,
                "distance_cm": case.profile_x_cm,
                "pH": profile[0].reshape(-1),
                "TotalPb_mol_m3_bulk": profile[1].reshape(-1),
                "Pb_aqueous_mol_m3_water": profile[2].reshape(-1),
                "Pb_aqueous_mol_m3_bulk": profile[3].reshape(-1),
                "Pb_adsorbed_mol_m3_bulk": profile[4].reshape(-1),
                "Pb_precipitated_mol_m3_bulk": profile[5].reshape(-1),
                "mass_gap_mol_m3_bulk": profile[6].reshape(-1),
            })
            if not args.summary_only:
                profile_df.to_csv(outdir / f"{case.name}_Pb_species_day5.csv", index=False)
            metrics[f"{case.name}_adsorbed_Pb_day5_min"] = float(profile_df["Pb_adsorbed_mol_m3_bulk"].min())
            metrics[f"{case.name}_mass_gap_day5_abs_max"] = float(profile_df["mass_gap_mol_m3_bulk"].abs().max())

        observation_likelihood = {}
        for split, values in likelihood_inputs.items():
            observation_likelihood[split] = gaussian_observation_likelihood(
                pd.concat(values["ph_frames"], ignore_index=True),
                np.concatenate(values["ph_predictions"]),
                pd.concat(values["pb_frames"], ignore_index=True),
                np.concatenate(values["pb_predictions"]),
            )

        if args.profile_rh_values:
            if len(rh_vars) != 1:
                raise ValueError("--profile-rh-values requires trainable RH")
            original_raw_rh = float(sess.run(rh_vars[0]))
            for requested_rh in (float(value) for value in args.profile_rh_values.split(",")):
                if not 1.0 < requested_rh < 70.0:
                    raise ValueError("RH profile values must satisfy 1 < RH < 70")
                q = (requested_rh - 1.0) / 69.0
                sess.run(rh_vars[0].assign(math.log(q / (1.0 - q))))
                profile_metrics, profile_score = calibration_metrics_and_score()
                rh_objective_slice.append({
                    "RH": requested_rh,
                    "selection_score": float(profile_score),
                    "selection_data_score": float(profile_metrics["selection_data_score"]),
                    "selection_physics_objective": float(profile_metrics["selection_physics_objective"]),
                })
            sess.run(rh_vars[0].assign(original_raw_rh))

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    if rh_objective_slice:
        pd.DataFrame(rh_objective_slice).to_csv(outdir / "rh_objective_slice.csv", index=False)
    summary = {
        "inversion_version": "baseline_3param_v5",
        "run_role": args.run_role,
        "run_label": args.run_label,
        "fixed_parameters": {
            "Keos": args.fixed_keos,
            "RH": args.fixed_rh,
            "pH50": args.fixed_ph50,
        },
        "parameters": {"Keos": float(final_values[0]), "RH": float(final_values[1]), "pH50": float(final_values[2])},
        "calibration_points_cm": sorted(pd.read_csv(args.data_dir / "pH_calibration.csv")["distance_cm"].unique().tolist()),
        "validation_points_cm": sorted(pd.read_csv(args.data_dir / "pH_validation.csv")["distance_cm"].unique().tolist()),
        "metrics": metrics,
        "observation_likelihood": observation_likelihood,
        "metrics_evaluation_data_stride": 1,
        "final_joint_loss": final_joint_loss,
        "final_joint_objective": final_joint_objective,
        "final_physics_objective": final_physics_objective,
        "joint_loss_mode": args.joint_loss_mode,
        "balance_scales": balance_scales,
        "rh_objective_slice": rh_objective_slice,
        "final_loss_breakdown": final_loss_breakdown,
        "best_guard": {
            "selection_basis": (
                "full_calibration_gaussian_likelihood_plus_fixed_collocation_scaled_physics"
                if args.selection_data_mode == "gaussian-likelihood" else
                "full_calibration_standardized_RMSE_plus_fixed_collocation_scaled_physics"
            ),
            "stage": best["stage"],
            "iteration": best["iteration"],
            "score": best["score"],
            "parameters": best["parameters"],
            "metrics": best["metrics"],
        },
        "stage_parameter_gradients": stage_parameter_gradients,
        "stage_stop_status": stage_stop_status,
        "smoke": bool(args.smoke),
        "output": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        save_parameter_trajectory(history, outdir, final_values)
    except Exception as exc:
        print(json.dumps({"parameter_trajectory_warning": str(exc)}))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
