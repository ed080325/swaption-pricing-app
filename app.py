import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                    value=np.exp(-0.03*k),
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
        delta = N * A0 * Nd1 / 10000
    elif type_ == "Receiver":
        V = N * A0 * (K*(1-Nd2) - F*(1-Nd1))
        delta = N * A0 * (Nd1-1) / 10000
    gamma = (N * A0 * norm.pdf(d1) / (F * sigma * np.sqrt(t))) / 10000**2
    vega  = N * A0 * F * np.sqrt(t) * norm.pdf(d1) * 0.01
    return V, delta, gamma, vega, F

def pv01(N, t, T, K, sigma, type_, freq, dfs, V):
    bp = 0.0001
    dfs_up = {}
    dfs_down = {}
    for year, df in dfs.items():
        r = -np.log(df) / year
        r_up = r + bp
        r_down = r - bp
        dfs_up[year] = np.exp(-r_up * year)
        dfs_down[year] = np.exp(-r_down * year)
    p_up, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_up)
    p_down, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_down)
    pv01 = p_up - p_down
    return pv01

def pv01_per_date(N, t, T, K, sigma, type_, freq, dfs, V):
    bp = 0.0001
    table = []
    for year, df in dfs.items():
        dfs_up = dfs.copy()
        dfs_down = dfs.copy()
        r = -np.log(df) / year
        r_up = r + bp
        r_down = r - bp
        dfs_up[year] = np.exp(-r_up * year)
        dfs_down[year] = np.exp(-r_down * year)
        p_up, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_up)
        p_down, _, _, _, _ = black_price(N, t, T, K, sigma, type_, freq, dfs_down)
        pv01 = p_up - p_down
        table.append({"Year": year, "PV01": pv01})
    return pd.DataFrame(table)  


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
    #V, delta, gamma, vega = black_price(N, t, T, K, sigma, type_, freq, dfs)
    output = black_price(N, t, T, K, sigma, type_, freq, dfs)
    if output == "Error":
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
        V, delta, gamma, vega, F = output
        st.subheader(f"Forward Rate: {F*100:.2f}%")
        pv01 = pv01(N, t, T, K, sigma, type_, freq, dfs, V)
        pv01_table = pv01_per_date(N, t, T, K, sigma, type_, freq, dfs, V)
        rpv01_table = round(pv01_table, 2)
        pv01_table.set_index("Year", inplace=True) # Set index after creating rounded table so it stays as a column to be plotted
        fig = px.line(pv01_table, x=pv01_table.index, y="PV01", markers=True)
        fig.update_traces(line_color="#CFFF04")
        fig.update_layout(
            font=dict(size=30), xaxis_title="Maturity", yaxis_title="PV01",
        )
        
        fig1 = go.Figure(data=[go.Table(
            header=dict(values=list(rpv01_table.columns),
                        align='center',
                        font=dict(size=20, weight="bold"),
                        height=40),
            cells=dict(values=[rpv01_table[col] for col in rpv01_table.columns],
                       align='center',
                       font=dict(size=20),
                       height=40))
        ])
        fig1.update_layout(
            height=81*T,
            margin=dict(l=0, r=0, b=0, t=0)
           )
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
        st.markdown("***") # Horizontal line
        st.markdown(f"""
            <div style="display:flex; gap:10px;">
                <div style="
                    background-color:#44a4c4;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Delta: £{delta:,.2f}
                </div>
                <div style="
                    background-color:#f39c12;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Gamma: £{gamma:,.2f}
                </div>
                <div style="
                    background-color:#872657;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    Vega: £{vega:,.2f}
                </div>
                <div style="
                    background-color:#1252cf;
                    color:#fbfef4;
                    padding:15px;
                    border-radius:10px;
                    font-size:24px;
                    font-weight:bold;
                    text-align:center;
                    flex:1;">
                    PV01: £{pv01:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="text-align:center; font-size:30px; font-weight:bold;">
                PV01 at each Maturity
            </div>
        """, unsafe_allow_html=True)
        st.markdown("\n") # Line break
        st.plotly_chart(fig1)
        st.plotly_chart(fig)
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



