import numpy as np
from scipy.stats import norm
import streamlit as st

warning = False
def discount_factors(t, T, freq):
    step = 1 if freq == "Annual" else 0.5
    times = np.arange(t, T + step, step)

    dfs = {}
    # chunk times into groups of 3
    for i in range(0, len(times), 3):
        cols = st.columns(3)
        for j, k in enumerate(times[i:i+3]):
            with cols[j]:
                dfs[k] = st.number_input(
                    f"Year {k:.1f}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.01,
                    key=f"df_{k}"
                )
    return dfs

def black_price(N, t, T, K, sigma, type_, freq, dfs):
    step = 1 if freq == "Annual" else 0.5
    A0 = (sum(dfs.values()) - dfs[t]) * step
    if A0 <= 0:
        st.error("Annuity computed as zero or negative, check discount factors")
        return "Error"
    F = (dfs[t] - dfs[T]) / A0
    if F <= 0:
        st.error("Forward rate computed as zero or negative. The Black model assumes lognormal distribution of the forward rate.")
        return "Error"
    d1 = (np.log(F/K) + 0.5*sigma**2*t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    if type_ == "Payer":
        V = N * A0 * (F*Nd1 - K*Nd2)
    elif type_ == "Receiver":
        V = N * A0 * (K*(1-Nd2) - F*(1-Nd1))
    return V

st.set_page_config(page_title="Swaption Pricing", layout="wide")
st.title("Black Model Swaption Pricing Engine")

with st.container(border=True):
    st.subheader("Parameters")
    col1, col2 = st.columns(2)
    with col1:
        type_ = st.selectbox("Swaption Type", ("Payer", "Receiver"))
        freq = st.selectbox("Coupon Frequency", ("Annual", "Semi-Annual"))
        t = st.slider("Swaption Expiry", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        sigma = st.slider("Volatility", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    with col2:
        N = st.number_input("Notional", min_value=100, value=1_000_000, step=100_000)
        K = st.number_input("Strike Rate (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.01) / 100

        T = st.slider("Swap Maturity", min_value=0.5, max_value=10.0, value=2.0, step=0.5) 
    if T <= t:
        st.warning("Swap Maturity must be greater than Swaption Expiry")
        warning = True
    if (T-t) % 1 != 0 and freq == "Annual":
        st.warning("Swap Duration Must Match Coupon Frequency")
        warning = True
if not warning:
    with st.container(border=True):
        st.subheader("Discount Factors")
        dfs = discount_factors(t, T, freq)
    V = black_price(N, t, T, K, sigma, type_, freq, dfs)
    if V == "Error":
        st.markdown(f"""
            <div style="
                background-color:#890a1e;
                color:#fbfef4;
                padding:15px;
                border-radius:10px;
                font-size:38px;
                font-weight:bold;
                text-align:center;">
                Pricing Error
            </div>
            """, unsafe_allow_html=True)       
    else:
        st.markdown(f"""
            <div style="
                background-color:#38830e;
                color:#fbfef4;
                padding:15px;
                border-radius:10px;
                font-size:38px;
                font-weight:bold;
                text-align:center;">
                Price: £{V:,.2f}
            </div>
            """, unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; margin-top:20px;">
    <a href="https://www.linkedin.com/in/e-chamberlain-hall/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
             width="30" height="30" alt="LinkedIn Logo">
    </a>
    <a href="https://www.linkedin.com/in/e-chamberlain-hall/" target="_blank" 
       style="color:#B390D4; font-size:16px; text-decoration:none;">
        LinkedIn
    </a>
</div>
""", unsafe_allow_html=True)



