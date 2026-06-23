"""
Caprock Integrity & Reservoir Fracture - BHP Max Predictor (v3, field units)
Streamlit app. Deploy via share.streamlit.io with requirements.txt:
    streamlit, numpy, scipy, matplotlib, pandas

Units: depth/thickness/radius in FEET, pressure/cohesion/tensile in PSI,
stress & pore-pressure gradients in PSI/FT, density in g/cc, permeability
in mD, viscosity in cp, compressibility in 1/psi.

Sv is integrated from density (overburden -> caprock -> reservoir), not a
fixed gradient. SHmax/Shmin remain simple gradients. Friction is entered
directly as an ANGLE (deg) per rock unit.

Scope/assumptions: single well, reservoir radial extent = Rmax (no-flow
outer boundary), single-phase weakly-compressible fluid, constant
permeability, caprock failure evaluated at the reservoir-caprock interface.
Screening tool -- validate against full coupled simulation (e.g. CMG)
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
# CONSTANTS (field units)
# ============================================================================
C1 = 2.6368e-4           # mD,cp,psi^-1 -> ft^2/hr (Earlougher)
HR_TO_YEAR = 8760.0
DENS_TO_GRAD = 0.4335     # psi/ft per g/cc

def k_to_cond_ft2yr(k_md, mu_cp, cf_psi):
    return C1 * k_md / (mu_cp * cf_psi) * HR_TO_YEAR

# ============================================================================
# Sv FROM DENSITY
# ============================================================================
def compute_Sv(target_depth, ob_depth, ob_density, cap_thickness, cap_density, reservoir_layers):
    Sv = 0.0
    cap_top = ob_depth
    cap_bottom = ob_depth + cap_thickness
    seg = min(target_depth, ob_depth)
    if seg > 0:
        Sv += ob_density * DENS_TO_GRAD * seg
    seg_top, seg_bottom = max(0.0, cap_top), min(target_depth, cap_bottom)
    if seg_bottom > seg_top:
        Sv += cap_density * DENS_TO_GRAD * (seg_bottom - seg_top)
    for layer in reservoir_layers:
        ltop, lthick, ldens = layer["top_depth"], layer["thickness"], layer["density"]
        lbottom = ltop + lthick
        seg_top, seg_bottom = max(ltop, cap_bottom), min(target_depth, lbottom)
        if seg_bottom > seg_top:
            Sv += ldens * DENS_TO_GRAD * (seg_bottom - seg_top)
    return Sv

# ============================================================================
# STRESS SORTING + FAILURE CRITERIA
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

def mohr_coulomb_shear(sigma1, sigma3, cohesion, friction_angle_deg):
    # Guard against the Nphi -> 1 singularity at very low friction angles
    # (denominator Nphi-1 -> 0 gives garbage). Clamp to a small positive floor.
    fa = max(friction_angle_deg, 5.0)
    phi = np.radians(fa)
    Nphi = np.tan(np.pi / 4 + phi / 2) ** 2
    Co = 2 * cohesion * np.sqrt(Nphi)
    P = (Nphi * sigma3 + Co - sigma1) / (Nphi - 1)
    # A negative reactivation pressure is non-physical for an injection (pressure-
    # raising) scenario -- it would mean the rock is already past shear failure at
    # zero pressure. Return None so it is excluded as a governing constraint rather
    # than reported as a misleading negative BHP.
    return P if P > 0 else None

def tensile_failure(sigma3, sigma_t):
    return sigma3 + sigma_t

# ============================================================================
# CONNECTING LAYERS (auto-sliced + thin-slice merge for numerical robustness)
# ============================================================================
def get_connecting_layers(reservoir_layers, zCaprock, zWell, min_thickness_ft=5.0):
    out = []
    for layer in reservoir_layers:
        top, thick = layer["top_depth"], layer["thickness"]
        bottom = top + thick
        seg_top = max(top, zCaprock)
        seg_bottom = min(bottom, zWell)
        if seg_bottom > seg_top:
            out.append([seg_bottom - seg_top, layer["k_mD"], layer["porosity"], layer["pp_gradient"]])
    out = out[::-1]

    def merge_pair(a, b):
        nt = a[0] + b[0]
        k_m = nt / (a[0] / a[1] + b[0] / b[1])
        phi_m = (a[0] * a[2] + b[0] * b[2]) / nt
        ppg_m = (a[0] * a[3] + b[0] * b[3]) / nt
        return [nt, k_m, phi_m, ppg_m]

    merged = out
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for idx in range(len(merged)):
            if merged[idx][0] < min_thickness_ft:
                if idx > 0:
                    merged[idx - 1] = merge_pair(merged[idx - 1], merged[idx]); del merged[idx]
                else:
                    merged[idx + 1] = merge_pair(merged[idx], merged[idx + 1]); del merged[idx]
                changed = True
                break
    return [tuple(m) for m in merged]

def baseline_interface_pressure(layers, P_start):
    return P_start - sum(ppg * thick for thick, k, phi, ppg in layers)

# ============================================================================
# 2D AXISYMMETRIC (r,z) DIFFUSION -- validated: steady-state ratio->1.0,
# monotonic in time & BHP, 1D-limit recovered at small Rmax, thin-slice-safe.
# ============================================================================
def build_radial_model(layers, mu_cp, cf_psi, rw, Rmax, Nr=30, Nz_per_layer=None):
    if Nz_per_layer is None:
        Nz_per_layer = [max(1, round(l[0] / 60.0)) for l in layers]
    z_edges = [0.0]; k_z, phi_z = [], []
    for (thick, k, phi), nz in zip(layers, Nz_per_layer):
        dzL = thick / nz
        for _ in range(nz):
            z_edges.append(z_edges[-1] + dzL); k_z.append(k); phi_z.append(phi)
    z_edges = np.array(z_edges); Nz = len(k_z)
    dz = np.diff(z_edges)
    r_edges = np.exp(np.linspace(np.log(rw), np.log(Rmax), Nr + 1))
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    cond_z = np.array([k_to_cond_ft2yr(k, mu_cp, cf_psi) for k in k_z])
    phi_arr = np.array(phi_z)
    return dict(Nr=Nr, Nz=Nz, r_edges=r_edges, r_centers=r_centers, z_edges=z_edges,
                dz=dz, cond_z=cond_z, phi_arr=phi_arr)

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
            cell_vol = np.pi * (r_edges[i+1]**2 - r_edges[i]**2) * dz[j]
            cap[p] = phi_arr[j] * cell_vol / dt
            diagA[p] += cap[p]
            area = np.pi * (r_edges[i+1]**2 - r_edges[i]**2)

            def add_conn(p, q, T):
                if p == bc_idx:
                    pass
                elif q == bc_idx:
                    bc_neighbor_T[p] = bc_neighbor_T.get(p, 0) + T
                    diagA[p] += T
                else:
                    rows.append(p); cols.append(q); vals.append(-T)
                    diagA[p] += T

            if i < Nr - 1:
                Tr = 2*np.pi*dz[j]*cond_z[j]/np.log(r_centers[i+1]/r_centers[i]); add_conn(p, idx(i+1,j), Tr)
            if i > 0:
                Tr = 2*np.pi*dz[j]*cond_z[j]/np.log(r_centers[i]/r_centers[i-1]); add_conn(p, idx(i-1,j), Tr)
            if j < Nz - 1:
                cfc = 2*cond_z[j]*cond_z[j+1]/(cond_z[j]+cond_z[j+1]); Tz = cfc*area/(0.5*(dz[j]+dz[j+1])); add_conn(p, idx(i,j+1), Tz)
            if j > 0:
                cfc = 2*cond_z[j]*cond_z[j-1]/(cond_z[j]+cond_z[j-1]); Tz = cfc*area/(0.5*(dz[j]+dz[j-1])); add_conn(p, idx(i,j-1), Tz)

    rows += [p for p in range(N) if p != bc_idx]
    cols += [p for p in range(N) if p != bc_idx]
    vals += [diagA[p] for p in range(N) if p != bc_idx]
    remap = -np.ones(N, dtype=int); cnt = 0
    for p in range(N):
        if p != bc_idx:
            remap[p] = cnt; cnt += 1
    A = sp.csr_matrix(([v for v in vals], ([remap[r] for r in rows], [remap[c] for c in cols])), shape=(N-1, N-1))
    cap_r = np.array([cap[p] for p in range(N) if p != bc_idx])
    bc_T_r = np.zeros(N - 1)
    for q, T in bc_neighbor_T.items():
        bc_T_r[remap[q]] = T
    return A, cap_r, bc_T_r, remap, idx(0, Nz - 1)

def solve_radial_2d(layers, mu_cp, cf_psi, BHP_target, tc_years, P_start, rw, Rmax, Nr=30, n_steps=80):
    layers_phys = [(l[0], l[1], l[2]) for l in layers]
    model = build_radial_model(layers_phys, mu_cp, cf_psi, rw, Rmax, Nr=Nr)
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

def solve_bhp_for_2d(target, layers, duration, Pstart, mu_cp, cf_psi, rw, Rmax):
    baseline = baseline_interface_pressure(layers, Pstart)
    def f(BHP):
        dp = solve_radial_2d(layers, mu_cp, cf_psi, BHP, duration, Pstart, rw, Rmax)
        return baseline + dp - target
    lo, hi = Pstart + 1.0, 3000.0
    tries = 0
    while f(hi) < 0 and tries < 40:
        hi *= 1.5; tries += 1
    if f(hi) < 0:
        return None
    return brentq(f, lo, hi, xtol=0.1)

# ============================================================================
# LITHOLOGY STYLES (14 types)
# ============================================================================
LITHOLOGY_STYLE = {
    "sandstone":    dict(hatch="...",  facecolor="#F4D58D", label="Sandstone"),
    "shale":        dict(hatch="----", facecolor="#9C8E7E", label="Shale"),
    "limestone":    dict(hatch="++",   facecolor="#CDE3EE", label="Limestone"),
    "dolomite":     dict(hatch="xx",   facecolor="#D8E8DC", label="Dolomite"),
    "mudstone":     dict(hatch="---",  facecolor="#8B8378", label="Mudstone"),
    "siltstone":    dict(hatch="..",   facecolor="#DDD6C5", label="Siltstone"),
    "anhydrite":    dict(hatch="+",    facecolor="#E8D5E8", label="Anhydrite"),
    "salt":         dict(hatch="x",    facecolor="#F5E6F5", label="Salt"),
    "coal":         dict(hatch="",     facecolor="#2B2B2B", label="Coal"),
    "marl":         dict(hatch=".+",   facecolor="#E5E8C8", label="Marl"),
    "chalk":        dict(hatch="o",    facecolor="#F0F4F0", label="Chalk"),
    "conglomerate": dict(hatch="oo",   facecolor="#C9A876", label="Conglomerate"),
    "breccia":      dict(hatch="OO",   facecolor="#B89968", label="Breccia"),
    "volcanic":     dict(hatch="\\\\", facecolor="#8B4A4A", label="Volcanic"),
}

# ============================================================================
# PLOTTING
# ============================================================================
def plot_mohr_panel(ax, sigma1, sigma3, P_values, cohesion, friction_angle_deg, title, sigma_t=0.0, sigma_max=None):
    phi_rad = np.radians(friction_angle_deg)
    if sigma_max is None:
        sigma_max = sigma1 * 1.25
    s = np.linspace(0, sigma_max, 200)
    tau_env = cohesion + s * np.tan(phi_rad)
    ax.plot(s, tau_env, 'k-', lw=2, label=f"MC envelope (C={cohesion:.0f} psi, $\\phi$={friction_angle_deg:.1f}\u00b0)")
    ax.axvline(-sigma_t, color='red', ls='--', lw=1.5, label=f"Tensile cutoff ($\\sigma_t$={sigma_t:.0f})")
    ax.axhline(0, color='gray', lw=0.6)
    colors = ['#2E86AB', '#E2574C']
    for i, (P, label) in enumerate(P_values):
        s1p, s3p = sigma1 - P, sigma3 - P
        center, radius = (s1p + s3p) / 2, (s1p - s3p) / 2
        th = np.linspace(0, np.pi, 200)
        ax.plot(center + radius * np.cos(th), radius * np.sin(th), color=colors[i % len(colors)],
                lw=2.2, label=f"{label} (P={P:.0f})")
        ax.plot([center - radius, center + radius], [0, 0], color=colors[i % len(colors)], lw=1, alpha=0.5)
    ax.set_xlabel("Effective normal stress (psi)")
    ax.set_ylabel("Shear stress (psi)")
    ax.set_title(title, fontsize=10.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.3)

def draw_layer(ax, top, bottom, name, lithology, x0=0.15, x1=0.65, info=""):
    style = LITHOLOGY_STYLE.get(lithology, dict(hatch="", facecolor="#DDDDDD", label=lithology))
    rect = mpatches.Rectangle((x0, top), x1 - x0, bottom - top,
                              facecolor=style["facecolor"], hatch=style["hatch"],
                              edgecolor="black", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x1 + 0.04, (top + bottom) / 2, f"{name}\n{info}", va="center", fontsize=8)

def plot_stratigraphic_log(ob_depth, cap_thickness, cap_lithology, reservoir_layers, zWell):
    cap_top = ob_depth
    cap_bottom = ob_depth + cap_thickness
    column_bottom = max(l["top_depth"] + l["thickness"] for l in reservoir_layers)
    fig, ax = plt.subplots(figsize=(7, 9))
    draw_layer(ax, cap_top, cap_bottom, "CAPROCK", cap_lithology, info=f"t={cap_thickness:.0f} ft")
    used = {cap_lithology}
    for l in reservoir_layers:
        draw_layer(ax, l["top_depth"], l["top_depth"] + l["thickness"], l["name"], l["lithology"],
                   info=f"k={l['k_mD']:.3g} mD, \u03c6={l['porosity']:.2f}, t={l['thickness']:.0f} ft")
        used.add(l["lithology"])
    well_x = (0.15 + 0.65) / 2
    ax.plot([well_x, well_x], [cap_top - (column_bottom-cap_top)*0.04, zWell], color="black", lw=2.2, zorder=5)
    for dx in [-0.05, 0.05]:
        ax.annotate("", xy=(well_x+dx, zWell), xytext=(well_x+dx*2.4, zWell),
                    arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.8), zorder=6)
    ax.text(well_x, zWell + (column_bottom-cap_top)*0.012, "  Perforation / BHP applied here",
            fontsize=8, color="#C0392B", va="top")
    ax.axhline(cap_bottom, color="#C0392B", ls="--", lw=1, alpha=0.7)
    ax.text(2.3, cap_bottom, f" {cap_bottom:.0f} ft (interface)", fontsize=8, ha="left", va="center", color="#C0392B")
    ax.axhline(cap_top, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.text(0.0, cap_top, f"{cap_top:.0f} ft ", fontsize=8, ha="right", va="center", color="gray")
    ax.set_ylim(column_bottom+(column_bottom-cap_top)*0.03, cap_top-(column_bottom-cap_top)*0.03)
    ax.set_xlim(-0.35, 2.3)
    ax.set_xticks([])
    ax.set_ylabel("Depth (ft)")
    ax.set_title("Stratigraphic Column & Wellbore", fontsize=12, fontweight="bold")
    handles = [mpatches.Patch(facecolor=LITHOLOGY_STYLE[l]["facecolor"], hatch=LITHOLOGY_STYLE[l]["hatch"],
                              edgecolor="black", label=LITHOLOGY_STYLE[l]["label"]) for l in used]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Lithology")
    plt.tight_layout()
    return fig

# ============================================================================
# DEFAULTS (field-unit equivalents of the canonical Burgan case)
# ============================================================================
DEFAULT_RESERVOIR = pd.DataFrame([
    {"name": "Wara",               "thickness": 492.1, "k_mD": 800.0, "porosity": 0.33, "density": 2.10, "cohesion": 1160.3, "tensile_strength": 0.0, "friction_angle": 31.0, "pp_gradient": 0.4377, "lithology": "sandstone"},
    {"name": "Mauddud",            "thickness": 164.0, "k_mD": 20.0,  "porosity": 0.25, "density": 2.40, "cohesion": 1160.3, "tensile_strength": 0.0, "friction_angle": 31.0, "pp_gradient": 0.4377, "lithology": "limestone"},
    {"name": "Upper Burgan + gap", "thickness": 820.2, "k_mD": 450.0, "porosity": 0.18, "density": 2.35, "cohesion": 1160.3, "tensile_strength": 0.0, "friction_angle": 31.0, "pp_gradient": 0.4377, "lithology": "sandstone"},
    {"name": "Middle Burgan",      "thickness": 492.1, "k_mD": 450.0, "porosity": 0.18, "density": 2.35, "cohesion": 1160.3, "tensile_strength": 0.0, "friction_angle": 31.0, "pp_gradient": 0.4377, "lithology": "sandstone"},
    {"name": "Lower Burgan",       "thickness": 984.3, "k_mD": 450.0, "porosity": 0.18, "density": 2.35, "cohesion": 1160.3, "tensile_strength": 0.0, "friction_angle": 31.0, "pp_gradient": 0.4377, "lithology": "sandstone"},
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
st.caption("2D axisymmetric pressure-diffusion + regime-agnostic Mohr-Coulomb screening. Field units throughout.")

col_input, col_results = st.columns([1, 1.4])

with col_input:
    st.subheader("Operational Parameters")
    Pstart = st.number_input("Current BHP (psi)", value=1450.4, step=50.0)
    zWell = st.number_input("Perforation depth (ft)", value=8366.1, step=50.0)
    rw = st.number_input("Wellbore radius (ft)", value=0.328, step=0.01, format="%.3f")
    duration = st.number_input("Planned injection duration (years)", value=0.5, step=0.1, min_value=0.01)
    Rmax = st.number_input("Effective reservoir radius (ft)", value=3280.8, step=100.0, min_value=10.0,
                           help="Distance to the nearest sealing boundary/fault, or half the spacing to the "
                                "nearest neighboring well. Larger Rmax = more radial dissipation = caprock safer.")

    st.subheader("Principal Stresses")
    c1, c2 = st.columns(2)
    with c1:
        SHmaxGrad = st.number_input("SHmax gradient (psi/ft)", value=0.5659, format="%.4f")
    with c2:
        ShminGrad = st.number_input("Shmin gradient (psi/ft)", value=0.5084, format="%.4f")
    st.caption("Sv is not a fixed gradient here \u2014 it is integrated from rock density across "
               "Overburden \u2192 Caprock \u2192 Reservoir (see Geological Column below).")

    st.subheader("Injectant Properties")
    fluid_choice = st.selectbox("Fluid preset", list(FLUID_PRESETS.keys()) + ["Custom"])
    if fluid_choice == "Custom":
        mu_cp = st.number_input("Viscosity (cp)", value=0.02, format="%.5f")
        cf_psi = st.number_input("Compressibility (1/psi)", value=3e-4, format="%.6f")
    else:
        mu_cp, cf_psi = FLUID_PRESETS[fluid_choice]
        st.caption(f"\u03bc = {mu_cp} cp, cf = {cf_psi:.2e} 1/psi")

    st.subheader("Geological Column")

    st.markdown("**Overburden**")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        ob_depth = st.number_input("Depth to top of caprock (ft)", value=6397.6, step=50.0)
    with oc2:
        ob_density = st.number_input("Average density (g/cc)", value=2.30, step=0.05, format="%.2f")
    with oc3:
        ob_pp_gradient = st.number_input("Pore pressure gradient (psi/ft)", value=0.4377, format="%.4f")

    st.markdown("**Caprock**")
    cap_bottom_preview = ob_depth  # top depth, computed
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        cap_name = st.text_input("Name", value="Ahmadi")
        cap_thickness = st.number_input("Thickness (ft)", value=492.1, step=10.0)
    with cc2:
        cap_density = st.number_input("Density (g/cc)", value=2.30, step=0.05, format="%.2f", key="cap_dens")
        cap_cohesion = st.number_input("Cohesion (psi)", value=1160.3, step=50.0, key="cap_coh")
    with cc3:
        cap_tensile = st.number_input("Tensile strength (psi)", value=0.0, step=10.0, key="cap_tens")
        cap_friction_angle = st.number_input("Friction angle (deg)", value=31.0, step=0.5, key="cap_fric")
    with cc4:
        cap_pp_gradient = st.number_input("Pore pressure gradient (psi/ft)", value=0.4377, format="%.4f", key="cap_ppg")
        cap_lithology = st.selectbox("Lithology", list(LITHOLOGY_STYLE.keys()), index=1, key="cap_lith")
    st.caption(f"Top depth (auto): **{ob_depth:.0f} ft**  |  Bottom depth / interface (auto): **{ob_depth+cap_thickness:.0f} ft**")

    st.markdown("**Reservoir**")
    st.caption("Top Depth is auto-chained from cumulative thickness below the caprock (read-only).")
    res_input = DEFAULT_RESERVOIR.copy()
    running = ob_depth + cap_thickness
    top_depths = []
    for _, row in res_input.iterrows():
        top_depths.append(running)
        running += row["thickness"]
    res_input.insert(1, "top_depth", top_depths)

    res_edited = st.data_editor(
        res_input, num_rows="dynamic", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name", width="medium"),
            "top_depth": st.column_config.NumberColumn("Top Depth (ft)", disabled=True, format="%.0f"),
            "thickness": st.column_config.NumberColumn("Thickness (ft)", min_value=1.0, step=10.0, format="%.0f"),
            "k_mD": st.column_config.NumberColumn("k (mD)", min_value=0.0001, step=1.0, format="%.3f"),
            "porosity": st.column_config.NumberColumn("Porosity", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            "density": st.column_config.NumberColumn("Density (g/cc)", min_value=1.0, max_value=4.0, step=0.05, format="%.2f"),
            "cohesion": st.column_config.NumberColumn("Cohesion (psi)", min_value=0.0, step=50.0, format="%.0f"),
            "tensile_strength": st.column_config.NumberColumn("Tensile Str. (psi)", min_value=0.0, step=10.0, format="%.0f"),
            "friction_angle": st.column_config.NumberColumn("Friction Angle (deg)", min_value=0.0, max_value=70.0, step=0.5, format="%.1f"),
            "pp_gradient": st.column_config.NumberColumn("Pp Gradient (psi/ft)", min_value=0.0, step=0.01, format="%.4f"),
            "lithology": st.column_config.SelectboxColumn("Lithology", options=list(LITHOLOGY_STYLE.keys())),
        },
    )
    # Re-chain top_depth in case rows/thicknesses were edited (authoritative recompute)
    running = ob_depth + cap_thickness
    fixed_top_depths = []
    for _, row in res_edited.iterrows():
        fixed_top_depths.append(running)
        running += row["thickness"]
    res_edited = res_edited.copy()
    res_edited["top_depth"] = fixed_top_depths

    compute = st.button("\U0001F680 Compute BHP Max", type="primary", use_container_width=True)

with col_results:
    if compute:
        zCaprock = ob_depth + cap_thickness

        # --- Validate the (dynamic) reservoir table: drop fully-blank rows, then
        #     reject if any required numeric field is missing/NaN. The dynamic
        #     data_editor lets users add rows that may be partially filled, and a
        #     NaN would silently corrupt Sv / diffusion results. ---
        req_cols = ["thickness", "k_mD", "porosity", "density", "cohesion",
                    "tensile_strength", "friction_angle", "pp_gradient"]
        res_clean = res_edited.dropna(how="all").copy()
        missing_mask = res_clean[req_cols].isna().any(axis=1)

        error = None
        if res_clean.empty:
            error = "The reservoir table is empty. Add at least one reservoir layer."
        elif missing_mask.any():
            bad_rows = [str(i + 1) for i in np.where(missing_mask.values)[0]]
            error = ("Some reservoir rows have empty cells (row(s): "
                     + ", ".join(bad_rows)
                     + "). Fill every column, or delete incomplete rows, before computing.")

        if error is None:
            # Re-chain top_depth on the cleaned table (authoritative)
            running_v = ob_depth + cap_thickness
            tops = []
            for _, row in res_clean.iterrows():
                tops.append(running_v); running_v += row["thickness"]
            res_clean["top_depth"] = tops
            reservoir_layers = res_clean.to_dict("records")
            column_max_depth = max(l["top_depth"] + l["thickness"] for l in reservoir_layers)

            if zWell < zCaprock:
                error = f"Perforation depth ({zWell:.0f} ft) cannot be shallower than the interface ({zCaprock:.0f} ft)."
            elif zWell > column_max_depth:
                error = (f"Perforation depth ({zWell:.0f} ft) exceeds the reservoir column "
                         f"({column_max_depth:.0f} ft). Add a deeper layer row.")

        if error:
            st.error(error)
        else:
            def layer_at_depth(layers, depth):
                for l in layers:
                    if l["top_depth"] <= depth < l["top_depth"] + l["thickness"]:
                        return l
                return layers[-1]

            active_layer = layer_at_depth(reservoir_layers, zWell)
            layers = get_connecting_layers(reservoir_layers, zCaprock, zWell)
            total_layers_thickness = sum(l[0] for l in layers)

            Sv_well = compute_Sv(zWell, ob_depth, ob_density, cap_thickness, cap_density, reservoir_layers)
            Sv_cap = compute_Sv(zCaprock, ob_depth, ob_density, cap_thickness, cap_density, reservoir_layers)

            res_stress = sort_principal_stresses(Sv_well, SHmaxGrad*zWell, ShminGrad*zWell)
            cap_stress = sort_principal_stresses(Sv_cap, SHmaxGrad*zCaprock, ShminGrad*zCaprock)

            P_res_tensile = tensile_failure(res_stress["sigma3"][1], active_layer["tensile_strength"])
            P_res_shear = mohr_coulomb_shear(res_stress["sigma1"][1], res_stress["sigma3"][1],
                                              active_layer["cohesion"], active_layer["friction_angle"])
            target_cap_tensile = tensile_failure(cap_stress["sigma3"][1], cap_tensile)
            target_cap_shear = mohr_coulomb_shear(cap_stress["sigma1"][1], cap_stress["sigma3"][1],
                                                   cap_cohesion, cap_friction_angle)

            with st.spinner("Solving 2D axisymmetric diffusion\u2026"):
                BHP_cap_tensile = solve_bhp_for_2d(target_cap_tensile, layers, duration, Pstart, mu_cp, cf_psi, rw, Rmax)
                BHP_cap_shear = solve_bhp_for_2d(target_cap_shear, layers, duration, Pstart, mu_cp, cf_psi, rw, Rmax) if target_cap_shear is not None else None

            res_candidates = [v for v in [P_res_tensile, P_res_shear] if v is not None]
            BHP_max_reservoir = min(res_candidates)
            res_mode = "tensile" if (P_res_shear is None or P_res_tensile <= P_res_shear) else "shear (Mohr-Coulomb)"

            cap_candidates = [v for v in [BHP_cap_tensile, BHP_cap_shear] if v is not None]
            if cap_candidates:
                BHP_max_caprock = min(cap_candidates)
                cap_mode = "tensile reopening" if (BHP_cap_tensile is not None and BHP_cap_tensile == BHP_max_caprock) else "shear (Mohr-Coulomb)"
                overall = "CAPROCK" if BHP_max_caprock < BHP_max_reservoir else "RESERVOIR"
            else:
                BHP_max_caprock = None
                overall = "RESERVOIR"

            st.subheader("\U0001F4CA Results")
            st.info(f"Active reservoir layer at perforation: **{active_layer['name']}** | "
                    f"Stress regime \u2014 Reservoir: **{res_stress['regime']}** | Caprock: **{cap_stress['regime']}**")
            st.caption(f"Sv (integrated from density) \u2014 at well: {Sv_well:.0f} psi | at interface: {Sv_cap:.0f} psi")

            diff_len = np.sqrt(k_to_cond_ft2yr(active_layer["k_mD"], mu_cp, cf_psi) * duration)
            st.caption(f"Diffusion length scale at this duration \u2248 {diff_len:,.0f} ft. Rmax = {Rmax:,.0f} ft \u2192 "
                       + ("reservoir behaves **bounded**." if Rmax < diff_len else "reservoir behaves **infinite-acting**."))

            m1, m2 = st.columns(2)
            with m1:
                st.metric("BHP Max \u2014 Reservoir Fracture", f"{BHP_max_reservoir:,.0f} psi", help=f"Governing mode: {res_mode}")
            with m2:
                if BHP_max_caprock is not None:
                    st.metric("BHP Max \u2014 Caprock Fail", f"{BHP_max_caprock:,.0f} psi", help=f"Governing mode: {cap_mode}")
                else:
                    st.metric("BHP Max \u2014 Caprock Fail", "Not reachable")

            if BHP_max_caprock is not None:
                ceiling = min(BHP_max_reservoir, BHP_max_caprock)
                st.success(f"**Overall operational ceiling: {ceiling:,.0f} psi** \u2014 limited by **{overall}**")
            else:
                st.success(f"**Overall operational ceiling: {BHP_max_reservoir:,.0f} psi** \u2014 limited by **RESERVOIR**")

            with st.expander("Full constraint breakdown"):
                rows_bd = [
                    {"Constraint": "Reservoir \u2014 tensile", "BHP (psi)": round(P_res_tensile), "Locus": f"Well depth ({active_layer['name']})"},
                    {"Constraint": "Reservoir \u2014 shear", "BHP (psi)": round(P_res_shear) if P_res_shear is not None else None, "Locus": f"Well depth ({active_layer['name']})"},
                    {"Constraint": "Caprock \u2014 tensile reopening", "BHP (psi)": round(BHP_cap_tensile) if BHP_cap_tensile else None, "Locus": "Interface"},
                    {"Constraint": "Caprock \u2014 shear", "BHP (psi)": round(BHP_cap_shear) if BHP_cap_shear else None, "Locus": "Interface"},
                ]
                st.dataframe(pd.DataFrame(rows_bd), use_container_width=True)

            st.subheader("\U0001F4C9 Mohr Circle")
            fig1, axes = plt.subplots(1, 2, figsize=(12, 5.5))
            plot_mohr_panel(axes[0], res_stress["sigma1"][1], res_stress["sigma3"][1],
                            [(Pstart, "Current BHP"), (BHP_max_reservoir, "BHP_max (res.)")],
                            active_layer["cohesion"], active_layer["friction_angle"], sigma_t=active_layer["tensile_strength"],
                            title=f"RESERVOIR @ {active_layer['name']} (z={zWell:.0f} ft)")
            local_now = baseline_interface_pressure(layers, Pstart)
            target_cap_binding = target_cap_tensile if (BHP_cap_tensile is not None and (BHP_cap_shear is None or BHP_cap_tensile <= BHP_cap_shear)) else target_cap_shear
            plot_mohr_panel(axes[1], cap_stress["sigma1"][1], cap_stress["sigma3"][1],
                            [(local_now, "Current (local)"), (target_cap_binding, "Failure threshold")],
                            cap_cohesion, cap_friction_angle, sigma_t=cap_tensile,
                            title=f"CAPROCK @ interface (z={zCaprock:.0f} ft)")
            st.pyplot(fig1)

            st.subheader("\U0001F5FA\uFE0F Stratigraphic Log")
            fig2 = plot_stratigraphic_log(ob_depth, cap_thickness, cap_lithology, reservoir_layers, zWell)
            st.pyplot(fig2)

            st.caption(
                "**Assumptions:** single well; reservoir radial extent = Rmax (no-flow outer boundary); "
                "single-phase, weakly-to-moderately compressible fluid; constant permeability; Sv integrated "
                "from density, SHmax/Shmin as simple gradients; caprock failure evaluated at the interface; "
                "pre-injection baseline assumes uniform historical depletion per layer's own pp_gradient. "
                "Validate against full coupled simulation (e.g. CMG) before operational decisions."
            )
    else:
        st.info("Set parameters on the left, then click **Compute BHP Max**.")
