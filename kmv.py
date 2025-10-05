import streamlit as st
import numpy as np
import scipy.stats as stats

# ==========================
# KMV Model Default Risk Calculator
# ==========================

st.set_page_config(page_title="KMV Default Probability Calculator", page_icon="📉", layout="centered")

st.title("📉 KMV Model Default Probability Calculator")
st.markdown("""
This calculator estimates a company's **probability of default (PD)** using the **KMV structural model**.
It is based on Merton’s model, where the firm's equity is treated as a call option on its assets.
""")

# --------------------------
# Input Section
# --------------------------
st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    equity_value = st.number_input("Market Value of Equity (E)", min_value=1.0, value=100.0, step=1.0)
    equity_vol = st.number_input("Equity Volatility (σ_E)", min_value=0.01, value=0.44, step=0.01, format="%.2f")
    risk_free_rate = st.number_input("Risk-free Rate (r)", min_value=0.0, value=0.05, step=0.01, format="%.2f")

with col2:
    debt_value = st.number_input("Book Value of Debt (D)", min_value=1.0, value=80.0, step=1.0)
    time_horizon = st.number_input("Time Horizon (T in years)", min_value=0.1, value=1.0, step=0.1, format="%.1f")
    tolerance = st.number_input("Solver Tolerance", min_value=1e-6, value=1e-5, step=1e-6, format="%.0e")

# --------------------------
# Helper Functions
# --------------------------
def kmv_solver(E, sigma_E, D, r, T, tol=1e-5, max_iter=100):
    """
    Estimate asset value (V) and asset volatility (sigma_V) using iterative KMV approach.
    """
    # Initial guess
    V = E + D
    sigma_V = sigma_E * (E / V)

    for _ in range(max_iter):
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)

        E_est = V * stats.norm.cdf(d1) - D * np.exp(-r * T) * stats.norm.cdf(d2)
        sigma_E_est = (V * stats.norm.cdf(d1) * sigma_V) / E_est

        # Update guesses
        V_new = V * (E / E_est)
        sigma_V_new = sigma_E * (E / V) / stats.norm.cdf(d1)

        if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
            break

        V, sigma_V = V_new, sigma_V_new

    return V, sigma_V, d1, d2


def kmv_default_probability(V, sigma_V, D, r, T):
    """
    Compute Distance to Default (DD) and Probability of Default (PD)
    """
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)

    # Distance to default
    DD = d2

    # Probability of default
    PD = stats.norm.cdf(-DD)
    return DD, PD

# --------------------------
# Compute Results
# --------------------------
if st.button("Calculate Default Probability"):
    try:
        V, sigma_V, d1, d2 = kmv_solver(equity_value, equity_vol, debt_value, risk_free_rate, time_horizon, tolerance)
        DD, PD = kmv_default_probability(V, sigma_V, debt_value, risk_free_rate, time_horizon)

        st.subheader("Results")
        st.write(f"**Estimated Firm Value (V):** {V:,.2f}")
        st.write(f"**Asset Volatility (σ_V):** {sigma_V:.4f}")
        st.write(f"**Distance to Default (DD):** {DD:.4f}")
        st.write(f"**Probability of Default (PD):** {PD*100:.2f}%")

        st.markdown("""
        **Interpretation:**
        - A higher DD means the firm is safer.
        - A lower DD (or higher PD) indicates greater credit risk.
        """)

        # Plot
        st.header("Visual Representation")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,3))
        x = np.linspace(-4, 4, 500)
        ax.plot(x, stats.norm.pdf(x), 'b-', label="Normal Distribution")
        ax.axvline(-DD, color='red', linestyle='--', label=f'Default Threshold (-DD={-DD:.2f})')
        ax.fill_between(x, 0, stats.norm.pdf(x), where=(x < -DD), color='red', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during calculation: {e}")

# Footer
st.markdown("""
---
**Notes:**
- Model is based on the KMV (Merton-type) structural credit risk framework.
- Inputs are illustrative; market data should be used for accurate analysis.
""")
