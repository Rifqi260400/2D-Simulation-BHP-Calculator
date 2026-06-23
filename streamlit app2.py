"""
Caprock Integrity & Reservoir Fracture - BHP Max Predictor (v2, 2D axisymmetric)
Streamlit app. Deploy via share.streamlit.io with a requirements.txt containing:
    streamlit, numpy, scipy, matplotlib, pandas

Scope/assumptions: single well, infinite-acting OR bounded reservoir (set via the
effective reservoir radius Rmax), single-phase weakly-compressible fluid, constant
permeability, caprock failure evaluated at the reservoir-caprock interface (z=0).
This is a screening tool; validate against full coupled simulation (e.g. CMG)
before operational decisions.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import brentq

st.set_page_config(page_title="Caprock & Reservoir BHP Max", layout="wide")

# ============================================================================
# CONSTANTS
# ============================================================================
C1 = 2.6368e-4
FT2HR_TO_M2YR = 0.092903 * 8760

def k_to_cond_m2yr(k_md, mu_cp, cf_psi):
    return C1 * k_md / (mu_cp * cf_psi) * FT2HR_TO_M2YR

# ============================================================================
# FAILURE CRITERIA (regime-agnostic)
# ============================================================================
def sort_principal_stresses(Sv, SHmax, Shmin):
    items = sorted([("Sv", Sv), ("SHmax", SHmax), ("Shmin", Shmin)], key=lambda x: -x[1])
    (n1, s1), (n2, s2), (n3, s3) = items
    if n1 == "Sv":
        regime = "Normal faulting"
    elif n3 == "Sv":
        regime = "Thrust / reverse faulting"
    else:
        regime = "Strike-slip faulting"
    return dict(sigma1=(n1, s1), sigma2=(n2, s2), sigma3=(n3, s3), regime=regime)

def mohr_coulomb_shear(sigma1, sigma3, cohesion, mu):
    phi = np.arctan(mu)
    Nphi = np.tan(np.pi / 4 + phi / 2) ** 2
    Co = 2 * cohesion * np.sqrt(Nphi)
    return (Nphi * sigma3 + Co - sigma1) / (Nphi - 1)

def tensile_failure(sigma3, st_):
    return sigma3 + st_

# ============================================================================
# CONNECTING-LAYER SLICING
# ============================================================================
def get_connecting_layers(column_df, zCaprock, zWell):
    out = []
    for _, row in column_df.iterrows():
        top, thick = row["top_depth"], row["thickness"]
        bottom = top + thick
        seg_top = max(top, zCaprock)
        seg_bottom = min(bottom, zWell)
        if seg_bottom > seg_top:
            out.append((seg_bottom - seg_top, row["k_mD"], row["porosity"]))
    return out[::-1]

# ============================================================================
# 2D AXISYMMETRIC (r,z) DIFFUSION MODEL
# (validated: proper Dirichlet elimination; steady-state ratio -> 1.0;
#  monotonic in time and BHP; 1D limit recovered as Rmax -> small)
# ============================================================================
def build_radial_model(layers, mu_cp, cf_psi, rw, Rmax, Nr=30, Nz_per_layer=None):
    if Nz_per_layer is None:
        Nz_per_layer = [max(3, int(l[0] / 20)) for l in layers]
    z_edges = [0.0]; k_z, phi_z = [], []
    for (thick, k, phi), nz in zip(layers, Nz_per_layer):
        dzL = thick / nz
        for _ in range(nz):
            z_edges.append(z_edges[-1] + dzL); k_z.append(k); phi_z.append(phi)
    z_edges = np.array(z_edges); Nz = len(k_z)
    dz = np.diff(z_edges)
    r_edges = np.exp(np.linspace(np.log(rw), np.log(Rmax), Nr + 1))
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    cond_z = np.array([k_to_cond_m2yr(k, mu_cp, cf_psi) for k in k_z])
    phi_arr = np.array(phi_z)
    return dict(Nr=Nr, Nz=Nz, r_edges=r_edges, r_centers=r_centers,
                z_edges=z_edges, dz=dz, cond_z=cond_z, phi_arr=phi_arr)

def assemble_and_eliminate(model, dt):
    Nr, Nz = model["Nr"], model["Nz"]
    r_edges, r_centers, dz = model["r_edges"], model["r_centers"], model["dz"]
    cond_z, phi_arr = model["cond_z"], model["phi_arr"]
    N = Nr * Nz
    idx = lambda i, j: i * Nz + j
    bc_idx = idx(0, 0)
    rows, cols, vals = [], [], []
    diagA = np.zeros(N); cap = np.zeros(N)
    bc_neighbor_T = {}
    for i in range(Nr):
        for j in range(Nz):
            p = idx(i, j)
            cell_vol = np.pi * (r_edges[i + 1] ** 2 - r_edges[i] ** 2) * dz[j]
            cap[p] = phi_arr[j] * cell_vol / dt
            diagA[p] += cap[p]
            area = np.pi * (r_edges[i + 1] ** 2 - r_edges[i] ** 2)

            def add_conn(p, q, T):
                # Each ordered (p,q) visited once; accumulate only from non-BC source.
                if p == bc_idx:
                    pass
                elif q == bc_idx:
                    bc_neighbor_T[p] = bc_neighbor_T.get(p, 0) + T
                    diagA[p] += T
                else:
                    rows.append(p); cols.append(q); vals.append(-T)
                    diagA[p] += T

            if i < Nr - 1:
                Tr = 2 * np.pi * dz[j] * cond_z[j] / np.log(r_centers[i + 1] / r_centers[i]); add_conn(p, idx(i + 1, j), Tr)
            if i > 0:
                Tr = 2 * np.pi * dz[j] * cond_z[j] / np.log(r_centers[i] / r_centers[i - 1]); add_conn(p, idx(i - 1, j), Tr)
            if j < Nz - 1:
                cfc = 2 * cond_z[j] * cond_z[j + 1] / (cond_z[j] + cond_z[j + 1]); Tz = cfc * area / (0.5 * (dz[j] + dz[j + 1])); add_conn(p, idx(i, j + 1), Tz)
            if j > 0:
                cfc = 2 * cond_z[j] * cond_z[j - 1] / (cond_z[j] + cond_z[j - 1]); Tz = cfc * area / (0.5 * (dz[j] + dz[j - 1])); add_conn(p, idx(i, j - 1), Tz)

    rows += [p for p in range(N) if p != bc_idx]
    cols += [p for p in range(N) if p != bc_idx]
    vals += [diagA[p] for p in range(N) if p != bc_idx]
    remap = -np.ones(N, dtype=int); cnt = 0
    for p in range(N):
        if p != bc_idx:
            remap[p] = cnt; cnt += 1
    A = sp.csr_matrix(([v for v in vals], ([remap[r] for r in rows], [remap[c] for c in cols])), shape=(N - 1, N - 1))
    cap_r = np.array([cap[p] for p in range(N) if p != bc_idx])
    bc_T_r = np.zeros(N - 1)
    for q, T in bc_neighbor_T.items():
        bc_T_r[remap[q]] = T
    return A, cap_r, bc_T_r, remap, idx(0, Nz - 1)

def solve_radial_2d(layers, mu_cp, cf_psi, BHP_target, tc_years, P_start, rw, Rmax, Nr=30, n_steps=80):
    model = build_radial_model(layers, mu_cp, cf_psi, rw, Rmax, Nr=Nr)
    dt = tc_years / n_steps
    A, cap, bc_T, remap, top_idx = assemble_and_eliminate(model, dt)
    lu = splu(A.tocsc())
    dp = np.zeros(A.shape[0])
    dP_target = BHP_target - P_start
    for n in range(1, n_steps + 1):
        t = n * dt
        dP_n = dP_target * min(t / tc_years, 1.0)
        rhs = cap * dp + bc_T * dP_n
        dp = lu.solve(rhs)
    return dp[remap[top_idx]]

def solve_bhp_for_2d(target, layers, duration, Pstart, mu_cp, cf_psi, gPp, rw, Rmax):
    total_thickness = sum(l[0] for l in layers)
    baseline = Pstart - gPp * total_thickness
    def f(BHP):
        dp = solve_radial_2d(layers, mu_cp, cf_psi, BHP, duration, Pstart, rw, Rmax)
        return baseline + dp - target
    lo, hi = Pstart + 0.01, 60.0
    tries = 0
    while f(hi) < 0 and tries < 50:
        hi *= 1.5; tries += 1
    if f(hi) < 0:
        return None
    return brentq(f, lo, hi, xtol=1e-2)

# ============================================================================
# PLOTTING
# ============================================================================
LITHOLOGY_STYLE = {
    "sandstone": dict(hatch="...", facecolor="#F4D58D", label="Sandstone"),
    "shale":     dict(hatch="----", facecolor="#9C8E7E", label="Shale"),
    "limestone": dict(hatch="++",  facecolor="#CDE3EE", label="Limestone"),
    "dolomite":  dict(hatch="xx",  facecolor="#D8E8DC", label="Dolomite"),
}

def plot_mohr_panel(ax, sigma1, sigma3, P_values, cohesion, mu, title, sigma_t=0.0, sigma_max=None):
    phi_rad = np.arctan(mu)
    if sigma_max is None:
        sigma_max = sigma1 * 1.25
    s = np.linspace(0, sigma_max, 200)
    tau_env = cohesion + s * np.tan(phi_rad)
    ax.plot(s, tau_env, 'k-', lw=2, label=f"MC envelope (C={cohesion:.1f}, $\\mu$={mu:.2f})")
    ax.axvline(-sigma_t, color='red', ls='--', lw=1.5, label=f"Tensile cutoff ($\\sigma_t$={sigma_t:.1f})")
    ax.axhline(0, color='gray', lw=0.6)
    colors = ['#2E86AB', '#E2574C']
    for i, (P, label) in enumerate(P_values):
        s1p, s3p = sigma1 - P, sigma3 - P
        center, radius = (s1p + s3p) / 2, (s1p - s3p) / 2
        th = np.linspace(0, np.pi, 200)
        ax.plot(center + radius * np.cos(th), radius * np.sin(th), color=colors[i % len(colors)],
                lw=2.2, label=f"{label} (P={P:.2f})")
        ax.plot([center - radius, center + radius], [0, 0], color=colors[i % len(colors)], lw=1, alpha=0.5)
    ax.set_xlabel("Effective normal stress (MPa)")
    ax.set_ylabel("Shear stress (MPa)")
    ax.set_title(title, fontsize=10.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.3)

def draw_layer(ax, top, bottom, name, k, phi, lithology, x0=0.15, x1=0.65):
    style = LITHOLOGY_STYLE.get(lithology, dict(hatch="", facecolor="#DDDDDD", label=lithology))
    rect = mpatches.Rectangle((x0, top), x1 - x0, bottom - top,
                              facecolor=style["facecolor"], hatch=style["hatch"],
                              edgecolor="black", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x1 + 0.04, (top + bottom) / 2,
            f"{name}\nk={k:.3g} mD, \u03c6={phi:.2f}, t={bottom - top:.0f} m",
            va="center", fontsize=8.5)

def plot_stratigraphic_log(column_df, zCaprock, zWell, caprock_thickness, caprock_lithology):
    cap_top = zCaprock - caprock_thickness
    column_bottom = (column_df["top_depth"] + column_df["thickness"]).max()
    fig, ax = plt.subplots(figsize=(7, 9))
    draw_layer(ax, cap_top, zCaprock, "CAPROCK", 0.001, 0.08, caprock_lithology)
    used = {caprock_lithology}
    for _, row in column_df.iterrows():
        draw_layer(ax, row["top_depth"], row["top_depth"] + row["thickness"],
                   row["name"], row["k_mD"], row["porosity"], row["lithology"])
        used.add(row["lithology"])
    well_x = (0.15 + 0.65) / 2
    ax.plot([well_x, well_x], [cap_top - (column_bottom - cap_top) * 0.04, zWell], color="black", lw=2.2, zorder=5)
    for dx in [-0.05, 0.05]:
        ax.annotate("", xy=(well_x + dx, zWell), xytext=(well_x + dx * 2.4, zWell),
                    arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.8), zorder=6)
    ax.text(well_x, zWell + (column_bottom - cap_top) * 0.012, "  Perforation / BHP applied here",
            fontsize=8, color="#C0392B", va="top")
    ax.axhline(zCaprock, color="#C0392B", ls="--", lw=1, alpha=0.7)
    ax.text(2.3, zCaprock, f" {zCaprock:.0f} m (interface)", fontsize=8, ha="left", va="center", color="#C0392B")
    ax.axhline(cap_top, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(0.0, cap_top, f"{cap_top:.0f} m ", fontsize=8, ha="right", va="center", color="gray")
    ax.set_ylim(column_bottom + (column_bottom - cap_top) * 0.03, cap_top - (column_bottom - cap_top) * 0.03)
    ax.set_xlim(-0.35, 2.3)
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)")
    ax.set_title("Stratigraphic Column & Wellbore", fontsize=12, fontweight="bold")
    handles = [mpatches.Patch(facecolor=LITHOLOGY_STYLE[l]["facecolor"], hatch=LITHOLOGY_STYLE[l]["hatch"],
                              edgecolor="black", label=LITHOLOGY_STYLE[l]["label"]) for l in used]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Lithology")
    plt.tight_layout()
    return fig

# ============================================================================
# DEFAULT DATA
# ============================================================================
DEFAULT_COLUMN = pd.DataFrame([
    {"name": "Wara",               "top_depth": 1950, "thickness": 150, "k_mD": 800.0, "porosity": 0.33, "lithology": "sandstone"},
    {"name": "Mauddud",            "top_depth": 2100, "thickness": 50,  "k_mD": 20.0,  "porosity": 0.25, "lithology": "limestone"},
    {"name": "Upper Burgan + gap", "top_depth": 2150, "thickness": 250, "k_mD": 450.0, "porosity": 0.18, "lithology": "sandstone"},
    {"name": "Middle Burgan",      "top_depth": 2400, "thickness": 150, "k_mD": 450.0, "porosity": 0.18, "lithology": "sandstone"},
    {"name": "Lower Burgan",       "top_depth": 2550, "thickness": 300, "k_mD": 450.0, "porosity": 0.18, "lithology": "sandstone"},
])
FLUID_PRESETS = {
    "Methane": (0.022, 3.45e-4),
    "CO2 (supercritical)": (0.055, 2.0e-4),
    "Brine / water": (0.6, 3.5e-6),
}

# ============================================================================
# UI
# ============================================================================
st.title("\U0001F6E2\uFE0F Caprock Integrity & Reservoir Fracture \u2014 BHP Max Predictor")
st.caption("2D axisymmetric (radial-vertical) pressure-diffusion + regime-agnostic Mohr-Coulomb screening. "
           "Single well, infinite-acting or bounded reservoir (set via effective radius Rmax).")

col_input, col_results = st.columns([1, 1.4])

with col_input:
    st.subheader("1\uFE0F\u20E3 Operational Parameters")
    Pstart = st.number_input("Current BHP \u2014 P_start (MPa)", value=10.0, step=0.5)
    zWell = st.number_input("Perforation depth \u2014 zWell (m)", value=2550.0, step=10.0)
    duration = st.number_input("Planned injection duration (years)", value=0.5, step=0.1, min_value=0.01)
    Rmax = st.number_input("Effective reservoir radius \u2014 Rmax (m)", value=1000.0, step=100.0, min_value=10.0,
                           help="Distance to the nearest sealing boundary/fault, or half the spacing to the "
                                "nearest neighboring well. Larger Rmax = more radial dissipation = caprock safer.")

    with st.expander("\u2699\uFE0F Advanced: Stress & Rock Mechanics"):
        c1, c2 = st.columns(2)
        with c1:
            SvGrad = st.number_input("Sv gradient (kPa/m)", value=21.4)
            ShminGrad = st.number_input("Shmin gradient (kPa/m)", value=11.5)
            zCaprock = st.number_input("Caprock interface depth (m)", value=1950.0)
            cohesionRes = st.number_input("Cohesion \u2014 reservoir (MPa)", value=8.0)
            mu_friction = st.number_input("Friction coefficient \u03bc", value=0.6, step=0.01)
        with c2:
            SHmaxGrad = st.number_input("SHmax gradient (kPa/m)", value=12.8)
            PpGrad = st.number_input("Pore pressure gradient (kPa/m)", value=9.9)
            sigma_t = st.number_input("Tensile strength (MPa)", value=0.0)
            cohesionCap = st.number_input("Cohesion \u2014 caprock (MPa)", value=8.0)
            rw = st.number_input("Wellbore radius (m)", value=0.1, step=0.01, format="%.3f")

    with st.expander("\U0001F4A7 Advanced: Injectant Properties"):
        fluid_choice = st.selectbox("Fluid preset", list(FLUID_PRESETS.keys()) + ["Custom"])
        if fluid_choice == "Custom":
            mu_cp = st.number_input("Viscosity (cp)", value=0.02, format="%.5f")
            cf_psi = st.number_input("Compressibility (1/psi)", value=3e-4, format="%.6f")
        else:
            mu_cp, cf_psi = FLUID_PRESETS[fluid_choice]
            st.caption(f"\u03bc = {mu_cp} cp, cf = {cf_psi:.2e} 1/psi")

    with st.expander("\U0001F5FA\uFE0F Advanced: Geological Column"):
        st.caption("Edit depths, thicknesses, permeabilities, porosities, lithologies. "
                   "Well\u2192caprock path is auto-sliced from this column.")
        column_df = st.data_editor(
            DEFAULT_COLUMN, num_rows="dynamic", use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Name", width="medium"),
                "top_depth": st.column_config.NumberColumn("Top Depth (m)", min_value=0.0, step=10.0, format="%.0f"),
                "thickness": st.column_config.NumberColumn("Thickness (m)", min_value=1.0, step=10.0, format="%.0f"),
                "k_mD": st.column_config.NumberColumn("Permeability (mD)", min_value=0.0001, step=1.0, format="%.3f"),
                "porosity": st.column_config.NumberColumn("Porosity", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "lithology": st.column_config.SelectboxColumn("Lithology", options=list(LITHOLOGY_STYLE.keys())),
            },
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            caprock_thickness = st.number_input("Caprock thickness (m)", value=150.0)
        with cc2:
            caprock_lithology = st.selectbox("Caprock lithology", list(LITHOLOGY_STYLE.keys()), index=1)

    compute = st.button("\U0001F680 Compute BHP Max", type="primary", use_container_width=True)

with col_results:
    if compute:
        gPp = PpGrad / 1000
        column_max_depth = (column_df["top_depth"] + column_df["thickness"]).max()

        error = None
        if zWell < zCaprock:
            error = f"zWell ({zWell:.0f} m) cannot be shallower than zCaprock ({zCaprock:.0f} m)."
        elif zWell > column_max_depth:
            error = (f"zWell ({zWell:.0f} m) exceeds the geological column depth ({column_max_depth:.0f} m). "
                     f"Add a deeper layer row.")

        if error:
            st.error(error)
        else:
            layers = get_connecting_layers(column_df, zCaprock, zWell)
            total_layers_thickness = sum(l[0] for l in layers)

            res_stress = sort_principal_stresses(SvGrad/1000*zWell, SHmaxGrad/1000*zWell, ShminGrad/1000*zWell)
            cap_stress = sort_principal_stresses(SvGrad/1000*zCaprock, SHmaxGrad/1000*zCaprock, ShminGrad/1000*zCaprock)

            P_res_tensile = tensile_failure(res_stress["sigma3"][1], sigma_t)
            P_res_shear = mohr_coulomb_shear(res_stress["sigma1"][1], res_stress["sigma3"][1], cohesionRes, mu_friction)
            target_cap_tensile = tensile_failure(cap_stress["sigma3"][1], sigma_t)
            target_cap_shear = mohr_coulomb_shear(cap_stress["sigma1"][1], cap_stress["sigma3"][1], cohesionCap, mu_friction)

            with st.spinner("Solving 2D axisymmetric diffusion\u2026"):
                BHP_cap_tensile = solve_bhp_for_2d(target_cap_tensile, layers, duration, Pstart, mu_cp, cf_psi, gPp, rw, Rmax)
                BHP_cap_shear = solve_bhp_for_2d(target_cap_shear, layers, duration, Pstart, mu_cp, cf_psi, gPp, rw, Rmax)

            BHP_max_reservoir = min(P_res_tensile, P_res_shear)
            res_mode = "tensile" if P_res_tensile <= P_res_shear else "shear (Mohr-Coulomb)"

            cap_candidates = [v for v in [BHP_cap_tensile, BHP_cap_shear] if v is not None]
            if cap_candidates:
                BHP_max_caprock = min(cap_candidates)
                cap_mode = "tensile reopening" if (BHP_cap_tensile is not None and BHP_cap_tensile == BHP_max_caprock) else "shear (Mohr-Coulomb)"
                overall = "CAPROCK" if BHP_max_caprock < BHP_max_reservoir else "RESERVOIR"
            else:
                BHP_max_caprock = None
                overall = "RESERVOIR"

            st.subheader("\U0001F4CA Results")
            st.info(f"Stress regime \u2014 Reservoir: **{res_stress['regime']}** | Caprock: **{cap_stress['regime']}**")

            diff_len = np.sqrt(k_to_cond_m2yr(450.0, mu_cp, cf_psi) * duration)
            st.caption(f"Diffusion length scale at this duration \u2248 {diff_len:,.0f} m. "
                       f"Rmax = {Rmax:,.0f} m \u2192 "
                       + ("reservoir behaves **bounded** (radial spreading limited)." if Rmax < diff_len
                          else "reservoir behaves **infinite-acting** (strong radial dissipation)."))

            m1, m2 = st.columns(2)
            with m1:
                st.metric("BHP Max \u2014 Reservoir Fracture", f"{BHP_max_reservoir:.2f} MPa", help=f"Governing mode: {res_mode}")
            with m2:
                if BHP_max_caprock is not None:
                    st.metric("BHP Max \u2014 Caprock Fail", f"{BHP_max_caprock:.2f} MPa", help=f"Governing mode: {cap_mode}")
                else:
                    st.metric("BHP Max \u2014 Caprock Fail", "Not reachable",
                              help="Radial dissipation too strong at this Rmax/duration for caprock to fail within a realistic BHP range.")

            if BHP_max_caprock is not None:
                ceiling = min(BHP_max_reservoir, BHP_max_caprock)
                st.success(f"**Overall operational ceiling: {ceiling:.2f} MPa** \u2014 limited by **{overall}**")
            else:
                st.success(f"**Overall operational ceiling: {BHP_max_reservoir:.2f} MPa** \u2014 limited by **RESERVOIR** "
                           f"(caprock not reachable at this Rmax/duration)")

            with st.expander("Full constraint breakdown"):
                rows_bd = [
                    {"Constraint": "Reservoir \u2014 tensile", "BHP (MPa)": round(P_res_tensile, 2), "Locus": "Well depth"},
                    {"Constraint": "Reservoir \u2014 shear", "BHP (MPa)": round(P_res_shear, 2), "Locus": "Well depth"},
                    {"Constraint": "Caprock \u2014 tensile reopening", "BHP (MPa)": round(BHP_cap_tensile, 2) if BHP_cap_tensile else None, "Locus": "Interface (z=0)"},
                    {"Constraint": "Caprock \u2014 shear", "BHP (MPa)": round(BHP_cap_shear, 2) if BHP_cap_shear else None, "Locus": "Interface (z=0)"},
                ]
                st.dataframe(pd.DataFrame(rows_bd), use_container_width=True)

            st.subheader("\U0001F4C9 Mohr Circle")
            fig1, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            plot_mohr_panel(axes[0], res_stress["sigma1"][1], res_stress["sigma3"][1],
                            [(Pstart, "Current BHP"), (BHP_max_reservoir, "BHP_max (res.)")],
                            cohesionRes, mu_friction, sigma_t=sigma_t,
                            title=f"RESERVOIR @ well depth (z={zWell:.0f} m)")
            local_now = Pstart - gPp * total_layers_thickness
            target_cap_binding = target_cap_tensile if (BHP_cap_tensile is not None and (BHP_cap_shear is None or BHP_cap_tensile <= BHP_cap_shear)) else target_cap_shear
            plot_mohr_panel(axes[1], cap_stress["sigma1"][1], cap_stress["sigma3"][1],
                            [(local_now, "Current (local)"), (target_cap_binding, "Failure threshold")],
                            cohesionCap, mu_friction, sigma_t=sigma_t,
                            title=f"CAPROCK @ interface (z={zCaprock:.0f} m) \u2014 local pressure")
            st.pyplot(fig1)

            st.subheader("\U0001F5FA\uFE0F Stratigraphic Log")
            fig2 = plot_stratigraphic_log(column_df, zCaprock, zWell, caprock_thickness, caprock_lithology)
            st.pyplot(fig2)

            st.caption(
                "**Assumptions:** single well; reservoir radial extent = Rmax (no-flow outer boundary); "
                "single-phase, weakly-to-moderately compressible fluid; constant permeability (no post-failure "
                "Barton-Bandis enhancement); caprock failure evaluated at the interface (z=0). The well\u2192caprock "
                "path uses a 2D axisymmetric diffusion solve that captures near-well radial spreading. Validate "
                "against full coupled simulation (e.g. CMG) before operational decisions."
            )
    else:
        st.info("Set parameters on the left, then click **Compute BHP Max**.")
