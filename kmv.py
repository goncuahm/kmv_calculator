import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# ----------------------------------------
# Streamlit App Configuration
# ----------------------------------------
st.title("📉 KMV Company Default Probability Calculator with Climate Adjustment")

st.markdown("""
This app calculates a company's **default probability** using the **KMV structural model** 
and provides three estimates:
1. **Standard KMV Normal Model**
2. **Climate-Risk Adjusted Model (Student-t or Mixture Shock)**
3. **Moody’s KMV Empirical EDF Estimate**
""")

# ----------------------------------------
# User Inputs with Defaults
# ----------------------------------------
st.sidebar.header("Input Parameters")

E = st.sidebar.number_input("Market Value of Equity (E)", value=1e9, step=1e8, format="%.2e")
D = st.sidebar.number_input("Book Value of Debt (D)", value=8e8, step=1e8, format="%.2e")
sigma_E = st.sidebar.number_input("Equity Volatility (σE)", value=0.44)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.03)
T = st.sidebar.number_input("Time Horizon (T, years)", value=1.0)

# ----------------------------------------
# Climate Risk Section
# ----------------------------------------
st.markdown("### 🌍 Climate Risk Parameters")

col3, col4 = st.columns(2)
with col3:
    use_t = st.checkbox("Use Student-t heavy-tail adjustment", value=True)
    nu = st.slider("Degrees of freedom (ν) for Student-t (lower ⇒ heavier tails)", 3, 50, 6)
with col4:
    use_mixture = st.checkbox("Use discrete climate-shock mixture", value=True)
    p_shock = st.slider("Probability of climate shock (p)", 0.0, 0.5, 0.05, 0.01)
    shock_frac = st.slider("Shock severity (fractional drop in assets) s", 0.0, 0.9, 0.25, 0.01)

# ----------------------------------------
# Core KMV Calculations
# ----------------------------------------
V = E + D  # Approximate asset value
sigma_A = sigma_E * E / (E + D)
DD = (np.log(V / D) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))

# 1️⃣ Standard KMV Probability
PD_normal = norm.cdf(-DD)

# 2️⃣ Climate-Adjusted Probability
PD_climate = PD_normal  # start with base

# Student-t adjustment
if use_t:
    PD_t = t.cdf(-DD, df=nu)
    PD_climate = PD_t

# Discrete climate-shock mixture
if use_mixture:
    PD_shock = norm.cdf(-(DD - np.log(1 - shock_frac)) / 1)
    PD_climate = (1 - p_shock) * PD_climate + p_shock * PD_shock

# 3️⃣ Empirical EDF Approximation
def empirical_edf(dd):
    return 1 / (1 + np.exp(2.0 * dd)) * 0.5

PD_empirical = empirical_edf(DD)

# ----------------------------------------
# Display Results
# ----------------------------------------
st.subheader("🧮 Default Probability Results")
col1, col2, col3 = st.columns(3)
col1.metric("KMV Normal PD", f"{PD_normal * 100:.4f}%")
col2.metric("Climate-Adjusted PD", f"{PD_climate * 100:.4f}%")
col3.metric("KMV Empirical EDF", f"{PD_empirical * 100:.4f}%")

st.write(f"**Estimated Asset Value (V):** {V:,.0f}")
st.write(f"**Estimated Asset Volatility (σA):** {sigma_A:.4f}")
st.write(f"**Distance to Default (DD):** {DD:.4f}")

# ----------------------------------------
# Plot Comparison
# ----------------------------------------
x = np.linspace(-4, 6, 500)
normal_cdf = norm.cdf(-x)
t_cdf = t.cdf(-x, df=nu)
empirical_curve = [empirical_edf(i) for i in x]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, normal_cdf, label="KMV Normal Model")
if use_t:
    ax.plot(x, t_cdf, label=f"Student-t (df={nu}) Climate Model")
if use_mixture:
    shock_cdf = (1 - p_shock) * normal_cdf + p_shock * norm.cdf(-(x - np.log(1 - shock_frac)))
    ax.plot(x, shock_cdf, "--", label=f"Mixture Shock Model (p={p_shock}, s={shock_frac})")
ax.plot(x, empirical_curve, "--", color="orange", label="KMV Empirical EDF")
ax.axvline(DD, color="red", linestyle=":", label=f"DD = {DD:.2f}")
ax.set_title("Comparison of Default Probability Models")
ax.set_xlabel("Distance to Default (DD)")
ax.set_ylabel("Probability of Default")
ax.legend()
st.pyplot(fig)

# ----------------------------------------
# Notes / Formulas
# ----------------------------------------
st.markdown("""
### 📘 KMV Model Formulas

1. **Asset Value Approximation**
   \\[
   V_A \\approx E + D
   \\]

2. **Asset Volatility**
   \\[
   \\sigma_A = \\sigma_E \\times \\frac{E}{E + D}
   \\]

3. **Distance to Default (DD)**
   \\[
   DD = \\frac{\\ln(V_A / D) + (r - 0.5 \\sigma_A^2)T}{\\sigma_A \\sqrt{T}}
   \\]

4. **Default Probability (Normal)**
   \\[
   PD_{normal} = N(-DD)
   \\]

5. **Climate-Adjusted**
   - **Student-t**: \\( PD_t = T_{\\nu}(-DD) \\)
   - **Mixture Model**: \\( PD_{mix} = (1 - p)PD + p N(-(DD - \\ln(1-s))) \\)

6. **KMV Empirical EDF**
   \\[
   PD_{empirical} = \\frac{1}{1 + e^{2 \\times DD}} \\times 0.5
   \\]
""")






# # app.py
# import streamlit as st
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="KMV + Climate Risk PD Calculator", layout="centered", page_icon="🌩️")

# st.title("KMV Default Probability Calculator — Climate Risk Adjusted")
# st.markdown("""
# Estimate a company's default probability using the KMV (Merton) framework, and compare:
# - **Baseline (Normal)** assumption, and
# - **Climate-adjusted** variants: Student-t (heavy tails) and a discrete *climate shock* mixture.
# """)

# # -------------------
# # Inputs
# # -------------------
# st.header("Firm / Market Inputs")
# col1, col2 = st.columns(2)
# with col1:
#     equity_value = st.number_input("Market Value of Equity (E)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
#     equity_vol = st.number_input("Equity Volatility (σ_E)", value=0.44, min_value=0.01, step=0.01, format="%.2f")
#     risk_free_rate = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, step=0.005, format="%.3f")
# with col2:
#     debt_value = st.number_input("Book Value of Debt (D)", value=85.0, min_value=0.01, step=1.0, format="%.2f")
#     time_horizon = st.number_input("Time Horizon (T, years)", value=1.0, min_value=0.01, step=0.1, format="%.2f")
#     tol = st.number_input("Solver Tolerance", value=1e-5, format="%.0e")

# st.markdown("### Climate risk parameters (adjust the sliders to represent stronger climate exposure)")
# col3, col4 = st.columns(2)
# with col3:
#     # Student-t params
#     use_t = st.checkbox("Use Student-t climate heavy-tail adjustment", value=True)
#     nu = st.slider("Degrees of freedom (ν) for Student-t (lower => heavier tails)", 3, 50, 6)
# with col4:
#     # mixture shock params
#     use_mixture = st.checkbox("Use discrete climate shock mixture model", value=True)
#     p_shock = st.slider("Probability of a climate shock (p)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
#     shock_frac = st.slider("Shock severity (fractional drop in assets) s", min_value=0.0, max_value=0.9, value=0.25, step=0.01)

# st.markdown("---")

# # -------------------
# # KMV iterative solver to estimate V and sigma_V
# # -------------------
# def kmv_solver(E, sigma_E, D, r, T, tol=1e-6, max_iter=200):
#     # Initial guesses
#     V = E + D
#     sigma_V = sigma_E * (E / V)  # naive start
#     for i in range(max_iter):
#         d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
#         d2 = d1 - sigma_V * np.sqrt(T)
#         # equity as call option on assets
#         E_est = V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2)
#         if E_est <= 0:
#             # fallback to avoid division by zero
#             E_est = 1e-8
#         sigma_E_est = (V * stats.norm.cdf(d1) * sigma_V) / E_est
#         # update using simple fixed point scaling
#         V_new = V * (E / E_est)
#         sigma_V_new = sigma_V * (sigma_E / sigma_E_est) if sigma_E_est > 0 else sigma_V
#         if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
#             V, sigma_V = V_new, sigma_V_new
#             break
#         V, sigma_V = V_new, sigma_V_new
#     # final d1,d2
#     d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
#     d2 = d1 - sigma_V * np.sqrt(T)
#     return V, sigma_V, d1, d2

# def baseline_DD_PD(V, sigma_V, D, r, T):
#     """Return Distance to Default and PD under normal assumption."""
#     d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
#     d2 = d1 - sigma_V * np.sqrt(T)
#     DD = d2
#     PD = stats.norm.cdf(-DD)
#     return DD, PD

# def t_pd_from_DD(DD, nu):
#     """
#     Compute PD under Student-t assumption.
#     If DD is distance to default (under normal derivation), PD_t = t_cdf(-DD * sqrt(nu/(nu-2))).
#     Derivation: scale t to unit variance before comparing.
#     Requires nu > 2 for finite variance.
#     """
#     if nu <= 2:
#         return 1.0  # degenerate: infinite variance => very high PD
#     scale_factor = np.sqrt(nu / (nu - 2.0))
#     # PD is probability of standardized t < -DD scaled appropriately:
#     pd_t = stats.t.cdf(-DD * scale_factor, df=nu)
#     return pd_t

# def PD_with_mixture(V, sigma_V, D, r, T, p_shock, shock_frac):
#     """
#     Mixture: with probability p_shock assets are hit by a one-time downward shock of fraction s.
#     After shock, new asset value becomes V*(1-s). We approximate by computing PD conditional on shock (using same sigma_V).
#     """
#     # baseline PD
#     DD, PD_baseline = baseline_DD_PD(V, sigma_V, D, r, T)
#     # PD if shock happens (we assume shock occurs immediately and then compute PD over horizon with reduced V)
#     V_shocked = V * (1.0 - shock_frac)
#     # avoid nonpositive V
#     if V_shocked <= 0:
#         PD_shock = 1.0
#     else:
#         DD_shock, PD_shock = baseline_DD_PD(V_shocked, sigma_V, D, r, T)
#     PD_mix = (1.0 - p_shock) * PD_baseline + p_shock * PD_shock
#     return PD_mix, PD_baseline, PD_shock, DD, DD_shock if V_shocked > 0 else None

# # -------------------
# # Compute
# # -------------------
# if st.button("Calculate"):
#     try:
#         V, sigma_V, d1, d2 = kmv_solver(equity_value, equity_vol, debt_value, risk_free_rate, time_horizon, tol=tol)
#         DD, PD_norm = baseline_DD_PD(V, sigma_V, debt_value, risk_free_rate, time_horizon)

#         # Student-t PD
#         PD_t = None
#         if use_t:
#             PD_t = t_pd_from_DD(DD, nu)

#         # Mixture PD
#         PD_mix = None
#         PD_shock = None
#         DD_shock = None
#         if use_mixture:
#             PD_mix, PD_norm_again, PD_shock, DD_base, DD_shock = PD_with_mixture(V, sigma_V, debt_value, risk_free_rate, time_horizon, p_shock, shock_frac)

#         # Show results
#         st.subheader("Estimated firm/asset quantities")
#         st.write(f"**Estimated Asset Value (V):** {V:,.2f}")
#         st.write(f"**Estimated Asset Volatility (σ_V):** {sigma_V:.4f}")
#         st.write(f"**Distance to Default (DD) [baseline]:** {DD:.4f}")

#         st.subheader("Default Probabilities")
#         st.write(f"- **Baseline (Normal) PD:** {PD_norm*100:.4f}%")
#         if use_t:
#             st.write(f"- **Student-t (ν={nu}) PD:** {PD_t*100:.4f}%")
#         if use_mixture:
#             st.write(f"- **Mixture PD (p={p_shock:.3f}, s={shock_frac:.2f}):** {PD_mix*100:.4f}%")
#             st.write(f"  - PD conditional on shock: {PD_shock*100:.4f}%")

#         # Plotting: standardized variable distributions and shaded default tail
#         st.header("Distribution visualization (standardized variable)")
#         fig, ax = plt.subplots(figsize=(8,4))

#         # Standardized domain x
#         x = np.linspace(-6, 6, 1000)

#         # Baseline normal PDF (standard normal)
#         pdf_norm = stats.norm.pdf(x)
#         ax.plot(x, pdf_norm, label="Standard Normal (baseline)", linewidth=1.8)

#         # Student-t PDF (scaled to unit variance)
#         if use_t:
#             # raw t has var = nu/(nu-2), scale to unit variance by multiplying t by sqrt((nu-2)/nu)
#             # That means PDF transformation: f_Z(z) = f_t(z * sqrt(nu/(nu-2))) * sqrt(nu/(nu-2))
#             scale_back = np.sqrt(nu/(nu-2.0))
#             t_x = x * scale_back
#             pdf_t = stats.t.pdf(t_x, df=nu) * scale_back
#             ax.plot(x, pdf_t, label=f"Student-t ν={nu}", linestyle="--")

#             # shade PD area: standardized threshold is -DD; for t we scaled threshold by sqrt(nu/(nu-2))
#             thresh_t = -DD * np.sqrt(nu/(nu-2.0))
#             # convert to x domain: threshold_t_scaled corresponds to x = -DD
#             # but when plotting in x (standardized unit-variance), PD_t = t.cdf(thresh_t)
#             ax.fill_between(x, 0, pdf_t, where=(x < -DD), color='red', alpha=0.2)

#         # For normal shade baseline PD region
#         ax.fill_between(x, 0, pdf_norm, where=(x < -DD), color='orange', alpha=0.15)

#         # Mixture: show as vertical line indicator(s)
#         if use_mixture:
#             # PD baseline and PD_shock correspond to thresholds: -DD and -DD_shock
#             ax.axvline(-DD, color='orange', linestyle=':', label=f'Baseline threshold (-DD={-DD:.2f})')
#             if DD_shock is not None:
#                 ax.axvline(-DD_shock, color='red', linestyle=':', label=f'After-shock threshold (-DD_shock={-DD_shock:.2f})')
#                 # Shade mixture expected tail approximated as weighted shading:
#                 # We will visually show both shaded regions
#                 ax.fill_between(x, 0, pdf_norm, where=(x < -DD_shock), color='red', alpha=0.2)

#         ax.set_xlim(-6, 6)
#         ax.set_ylim(bottom=0)
#         ax.set_xlabel("Standardized asset shock (unit variance)")
#         ax.set_ylabel("Density")
#         ax.legend()
#         ax.set_title("Standardized shock distributions and default tail (left side)")

#         st.pyplot(fig)

#         st.markdown("### Interpretation notes")
#         st.markdown("""
#         - Student-t with low ν increases left-tail mass: PD_t > PD_normal for the same DD.  
#         - The mixture model shows how a discrete climate shock (probability *p* and size *s*) increases the **effective** PD by blending PDs conditional on shock/no-shock:
#           \[
#             PD_{mix} = (1-p)\,PD_{base} + p\,PD_{shock}
#           \]
#         - You can combine both methods (e.g., heavy tails **and** shocks) for additional conservatism.
#         """)
#     except Exception as e:
#         st.error(f"Calculation error: {e}")

# st.markdown("---")
# st.markdown("**Notes:** This is an illustrative model. Climate risk is complex — consider scenario analysis, stress-testing, and empirical calibration (EDF mapping) for production use.")














# # -------------------
# # Mathematical notes
# # -------------------
# st.markdown("---")
# with st.expander("📘 Mathematical Formulas and Model Notes"):
#     st.markdown(r"""
# ### **1. Merton (KMV) Model Setup**

# The firm's **equity** is modeled as a **call option on its assets**:

# \[
# E = V \, N(d_1) - D \, e^{-rT} \, N(d_2)
# \]

# where:
# - \(E\): market value of equity (input)  
# - \(V\): market value of assets (estimated)  
# - \(D\): face value of debt (input)  
# - \(r\): risk-free rate (input)  
# - \(T\): time horizon (input)  
# - \(N(\cdot)\): standard normal CDF  

# ---

# ### **2. Asset Dynamics**

# Firm value follows a geometric Brownian motion:

# \[
# \frac{dV_t}{V_t} = \mu \, dt + \sigma_V \, dW_t
# \]

# where \( \mu \) is expected asset return and \( \sigma_V \) is asset volatility (estimated).

# ---

# ### **3. Option Parameters**

# \[
# d_1 = \frac{\ln(V/D) + (r + 0.5 \sigma_V^2)T}{\sigma_V \sqrt{T}}, 
# \quad
# d_2 = d_1 - \sigma_V \sqrt{T}
# \]

# ---

# ### **4. Relation Between Equity and Asset Volatility**

# \[
# \sigma_E = \frac{V N(d_1)}{E} \, \sigma_V
# \]

# This is used to solve for \(V\) and \(σ_V\) iteratively.

# ---

# ### **5. Distance to Default (DD)**

# \[
# DD = \frac{\ln(V/D) + (r - 0.5 \sigma_V^2)T}{\sigma_V \sqrt{T}}
# \]

# It represents how many standard deviations the firm’s asset value is away from the default point \(D\).

# ---

# ### **6. Probability of Default (PD)**

# **Baseline KMV:**
# \[
# PD_{\text{Normal}} = N(-DD)
# \]

# **Climate-adjusted (Student-t):**
# \[
# PD_t = F_t(-DD \sqrt{\tfrac{\nu}{\nu - 2}})
# \]
# where \(F_t(\cdot)\) is the CDF of the Student-t distribution with \(\nu\) degrees of freedom.

# **Climate-shock mixture:**
# \[
# PD_{\text{mix}} = (1 - p_{\text{shock}})PD_{\text{base}} + p_{\text{shock}}PD_{\text{shock}}
# \]
# where \(PD_{\text{shock}}\) is the PD after an adverse climate-induced asset shock.

# ---

# ### **7. Empirical EDF Mapping (KMV-style)**

# Empirically observed mapping from **Distance-to-Default** to **Expected Default Frequency**:

# \[
# EDF = f(DD)
# \]

# The function \(f(\cdot)\) is interpolated from empirical or vendor (e.g., Moody's KMV) calibration data.

# ---

# ### **8. Interpretation Summary**

# | Symbol | Meaning | Observed / Estimated |
# |:--|:--|:--|
# | \(E\) | Market value of equity | Observed |
# | \(σ_E\) | Equity volatility | Observed |
# | \(D\) | Debt value | Input |
# | \(V\) | **Estimated asset value** | Solved |
# | \(σ_V\) | **Estimated asset volatility** | Solved |
# | \(DD\) | Distance to default | Derived |
# | \(PD\) | Default probability | Derived |

# ---

# These formulas form the basis of the **KMV default risk model** and its **climate-adjusted extensions**.
#     """)
