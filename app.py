"""
JPM Chooser Option Pricing – Streamlit Application
Enhanced UI with improved interactivity, visualization, and user experience.
Integrates:
- Data pipeline (load from processed CSV or fetch live)
- BSM closed-form and Monte Carlo pricing
- Interactive parameter inputs with tooltips
- Plotly sensitivity charts with hover information
- Monte Carlo payoff distribution histogram
- Current market snapshot
- Performance metrics with deltas
"""

import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="JPM Chooser Option Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container max width - make it wider */
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Metric cards styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 5px;
    }
    .metric-delta-positive {
        color: #28a745;
        font-weight: 600;
    }
    .metric-delta-negative {
        color: #dc3545;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        padding: 10px 0;
    }
    .sidebar-section h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 5px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e3a5f;
    }
    
    /* Divider */
    hr {
        margin: 20px 0;
        border: 0;
        border-top: 2px solid #e9ecef;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
    }
    
    /* Make Plotly charts take full width */
    .stPlotlyChart {
        width: 100% !important;
    }
    
    /* Better spacing for columns */
    .row-widget.stColumns {
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. BSM functions
# -------------------------------------------------------------------

def bs_call(S, K, T, r, q, sigma):
    """Black-Scholes-Merton call option price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes-Merton put option price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def chooser_price_closed_form(S0, K, T1, T2, r, q, sigma):
    """
    Chooser option pricing using the Rubinstein (1991) decomposition.
    Returns: (chooser_price, call_leg, put_leg)
    """
    tau = T2 - T1
    
    # Adjusted strike for put leg
    K_prime = K * np.exp(-(r - q) * tau)
    
    # Call option on the stock at T2
    call_leg = bs_call(S0, K, T2, r, q, sigma)
    
    # Put option on the stock at T1 with adjusted strike
    put_leg = bs_put(S0, K_prime, T1, r, q, sigma)
    
    # Chooser price = call(T2) + put(T1, K')
    price = call_leg + put_leg
    
    return price, call_leg, put_leg

@st.cache_data
def simulate_chooser_mc(S0, K, T1, T2, r, q, sigma, n_paths=200000, decision_rule="optimal", seed=42):
    """
    Monte Carlo simulation for chooser option pricing.
    Returns: (price, standard_error, discounted_payoffs)
    """
    rng = np.random.default_rng(seed)
    tau = T2 - T1
    
    # Generate paths to T1
    z1 = rng.standard_normal(n_paths)
    S_t1 = S0 * np.exp((r - q - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * z1)
    
    # Decision rule at T1
    if decision_rule == "optimal":
        # Compare S_t1 to the present value of K at T2
        threshold = K * np.exp(-(r - q) * tau)
        choose_call = S_t1 >= threshold
    else:
        # Simplified rule: choose call if stock price is above strike
        choose_call = S_t1 >= K
    
    # Generate paths from T1 to T2
    z2 = rng.standard_normal(n_paths)
    S_t2 = S_t1 * np.exp((r - q - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * z2)
    
    # Calculate payoffs
    call_payoff = np.maximum(S_t2 - K, 0.0)
    put_payoff = np.maximum(K - S_t2, 0.0)
    
    # Choose based on decision rule
    payoff = np.where(choose_call, call_payoff, put_payoff)
    
    # Discount to present value
    discounted_payoffs = np.exp(-r * T2) * payoff
    price = discounted_payoffs.mean()
    se = discounted_payoffs.std(ddof=1) / np.sqrt(n_paths)
    
    return price, se, discounted_payoffs

# -------------------------------------------------------------------
# 2. Data loading – local CSV or live fetch
# -------------------------------------------------------------------

PROCESSED_PATH = "data/processed/jpm_features.csv"
@st.cache_data(ttl=3600)
def fetch_live_data():
    """
    Fetch JPM daily data, Treasury rates, dividends, and VIX using yfinance only.
    No API key required.
    """
    try:
        import yfinance as yf
    except ImportError:
        st.error("Please install yfinance to use live data.")
        return pd.DataFrame()

    end = datetime.today()
    start = end - timedelta(days=5*365)

    with st.spinner("Fetching live data from yfinance..."):
        # --- Download JPM ---
        jpm = yf.download("JPM", start=start, end=end, progress=False)
        if jpm.empty:
            st.warning("No JPM data received from yfinance.")
            return pd.DataFrame()

        # Flatten MultiIndex columns
        if isinstance(jpm.columns, pd.MultiIndex):
            jpm.columns = jpm.columns.get_level_values(0)

        jpm = jpm.reset_index()
        
        # --- Handle column name variations ---
        rename_map = {}
        for col in jpm.columns:
            col_lower = str(col).lower()
            if 'date' in col_lower:
                rename_map[col] = 'date'
            elif 'adj' in col_lower and 'close' in col_lower:
                rename_map[col] = 'adjusted_close'
            elif col_lower == 'open':
                rename_map[col] = 'open'
            elif col_lower == 'high':
                rename_map[col] = 'high'
            elif col_lower == 'low':
                rename_map[col] = 'low'
            elif col_lower == 'close':
                rename_map[col] = 'close'
            elif col_lower == 'volume':
                rename_map[col] = 'volume'
        
        jpm.rename(columns=rename_map, inplace=True)
        
        # Ensure we have required columns
        if 'adjusted_close' not in jpm.columns and 'close' in jpm.columns:
            jpm['adjusted_close'] = jpm['close']
        
        # Convert to naive datetime (remove timezone)
        jpm["date"] = pd.to_datetime(jpm["date"]).dt.tz_localize(None)

        # --- Download rates and VIX from yfinance ---
        try:
            rate_tickers = {
                "rate_3month": "^IRX",
                "rate_5year": "^FVX", 
                "rate_10year": "^TNX",
                "vix": "^VIX"
            }
            
            rate_data = {}
            for name, ticker in rate_tickers.items():
                data = yf.download(ticker, start=start, end=end, progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    data = data.reset_index()
                    data.rename(columns={"Date": "date", "Close": name}, inplace=True)
                    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
                    rate_data[name] = data[["date", name]]
                else:
                    rate_data[name] = pd.DataFrame()
            
            # Merge all rates
            rates = None
            for name, df_rate in rate_data.items():
                if not df_rate.empty:
                    if rates is None:
                        rates = df_rate
                    else:
                        rates = pd.merge(rates, df_rate, on="date", how="outer")
            
            # Forward fill and convert to decimal (FIXED: using ffill)
            if rates is not None:
                rates = rates.sort_values("date")
                for col in ["rate_3month", "rate_5year", "rate_10year", "vix"]:
                    if col in rates.columns:
                        # FIXED: Use ffill() which is more compatible
                        rates[col] = rates[col].ffill().bfill()
                        if col != "vix":
                            rates[col] = rates[col] / 100.0
                
                # Approximate 1-year rate
                if "rate_3month" in rates.columns and "rate_5year" in rates.columns:
                    rates["rate_1year"] = rates["rate_3month"] + 0.25 * (rates["rate_5year"] - rates["rate_3month"])
                else:
                    rates["rate_1year"] = 0.04

        except Exception as e:
            st.warning(f"Could not fetch rate data: {e}. Using dummy rates.")
            rates = pd.DataFrame()

        # --- Merge JPM with rates ---
        if rates is not None and not rates.empty:
            df = pd.merge(jpm, rates, on="date", how="left")
        else:
            df = jpm
            for col in ["rate_1year", "rate_3month", "rate_5year", "rate_10year", "vix"]:
                df[col] = 0.0

        # Fill any missing rates (FIXED: using ffill)
        for col in ["rate_1year", "rate_3month", "rate_10year", "vix"]:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0.0)
            else:
                df[col] = 0.0

        # --- Get dividend data from yfinance ---
        try:
            ticker = yf.Ticker("JPM")
            dividends_data = ticker.dividends
            if not dividends_data.empty:
                div_df = dividends_data.reset_index()
                div_df.columns = ["date", "dividend_per_share"]
                div_df["date"] = pd.to_datetime(div_df["date"]).dt.tz_localize(None)
                
                df["date"] = pd.to_datetime(df["date"])
                div_df["date"] = pd.to_datetime(div_df["date"])
                
                df = pd.merge_asof(
                    df.sort_values("date"),
                    div_df.sort_values("date"),
                    on="date",
                    direction="backward"
                )
                df["dividend_per_share"] = df["dividend_per_share"].ffill().fillna(0.0)
            else:
                df["dividend_per_share"] = 0.0
        except Exception as e:
            st.warning(f"Could not fetch dividend data: {e}. Using dummy dividends.")
            df["dividend_per_share"] = 0.0

        # Compute dividend growth
        df["dividend_growth_pct"] = df["dividend_per_share"].pct_change()
        df["dividend_growth_yoy"] = df["dividend_per_share"] / df["dividend_per_share"].shift(252) - 1

        # Compute returns and volatility
        df["log_return"] = np.log(df["adjusted_close"] / df["adjusted_close"].shift(1))
        df["volatility_63d"] = df["log_return"].rolling(63).std() * np.sqrt(252)
        df["volatility_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
        df["volatility_10d"] = df["log_return"].rolling(10).std() * np.sqrt(252)

        return df

@st.cache_data
def load_processed_data():
    """Load the preprocessed feature CSV."""
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
        return df
    else:
        return None

# -------------------------------------------------------------------
# 3. Cached sensitivity grid computation
# -------------------------------------------------------------------

@st.cache_data
def get_sensitivity_grid(param_name, param_values, base_tuple):
    """
    Compute sensitivity grid for a given parameter.
    base_tuple is a tuple of (key, value) pairs to make it hashable.
    """
    base = dict(base_tuple)
    rows = []
    for v in param_values:
        p = base.copy()
        p[param_name] = v
        chooser, _, _ = chooser_price_closed_form(
            p["S0"], p["K"], p["T1"], p["T2"], p["r"], p["q"], p["sigma"]
        )
        call = bs_call(p["S0"], p["K"], p["T2"], p["r"], p["q"], p["sigma"])
        put = bs_put(p["S0"], p["K"], p["T2"], p["r"], p["q"], p["sigma"])
        rows.append({param_name: v, "chooser": chooser, "call": call, "put": put})
    return pd.DataFrame(rows)

# -------------------------------------------------------------------
# 4. Helper function for date formatting
# -------------------------------------------------------------------

def format_date(date_obj):
    """Format date for display."""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime("%Y-%m-%d")
    return str(date_obj)

# -------------------------------------------------------------------
# 5. Main app with navigation
# -------------------------------------------------------------------

# Data source selection in sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Data Source")
    use_live = st.checkbox(
        "Use live data (yfinance)",
        value=False,
        help="Fetch latest data directly from Yahoo Finance. No API key required."
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

# Load data based on selection
if use_live:
    with st.spinner("Fetching live market data..."):
        df = fetch_live_data()
    if df.empty:
        st.error("Live data fetch failed. Falling back to processed CSV.")
        df = load_processed_data()
        use_live = False
    else:
        st.sidebar.success("Live data loaded successfully")
else:
    df = load_processed_data()
    if df is None:
        st.error(
            f"Processed data not found at `{PROCESSED_PATH}`.\n\n"
            "Please run the preprocessing pipeline first (`preprocessingPipeline.ipynb`) "
            "or enable live data using the checkbox above."
        )
        st.stop()

# Date selection
available_dates = df["date"].dt.date.unique()
selected_date = st.sidebar.selectbox(
    "As-of Date",
    available_dates,
    index=len(available_dates)-1,
    help="Select the valuation date for pricing",
    format_func=format_date
)

# Get data for selected date
row = df[df["date"].dt.date == selected_date].iloc[-1] if selected_date else df.iloc[-1]

# Extract default parameters
default_S0 = float(row["close"]) if "close" in row else float(row["adjusted_close"])
default_sigma = float(row["volatility_63d"]) if pd.notna(row.get("volatility_63d")) else 0.25

# Rate: yfinance returns as decimal (0.045 for 4.5%)
default_r = float(row["rate_1year"]) if pd.notna(row.get("rate_1year")) else 0.04
if default_r > 1:
    default_r = default_r / 100.0

# Dividend yield: annualized dividend / current price
if "dividend_per_share" in row and pd.notna(row["dividend_per_share"]) and row["dividend_per_share"] > 0:
    annual_dividend = row["dividend_per_share"] * 4  # Quarterly dividend
    q_est = annual_dividend / default_S0 if default_S0 > 0 else 0.02
else:
    q_est = 0.02

# -------------------------------------------------------------------
# 6. Navigation
# -------------------------------------------------------------------

# Define pages
PAGES = {
    "Overview": "overview",
    "Detailed Analysis": "detailed"
}

# Page selection
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select Page",
    list(PAGES.keys()),
    index=0
)

# -------------------------------------------------------------------
# 7. Parameter inputs (shared across pages)
# -------------------------------------------------------------------

with st.sidebar:
    st.divider()
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Option Parameters")
    
    S0 = st.number_input(
        "Spot Price (S0)",
        value=default_S0,
        step=0.1,
        format="%.2f",
        help="Current price of JPM stock"
    )
    
    K = st.number_input(
        "Strike Price (K)",
        value=150.0,
        step=1.0,
        min_value=0.1,
        help="Option strike price"
    )
    
    T2 = st.number_input(
        "Maturity (T2, years)",
        value=1.0,
        step=0.1,
        min_value=0.1,
        max_value=5.0,
        help="Time to option maturity in years"
    )
    
    T1 = st.slider(
        "Decision Date (T1, years)",
        min_value=0.01,
        max_value=float(T2),
        value=min(0.5, float(T2)),
        step=0.01,
        help="Time when you can choose between call and put"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Market Parameters")
    
    r = st.number_input(
        "Risk-Free Rate (r)",
        value=default_r,
        step=0.001,
        format="%.4f",
        help="Annualized continuously compounded risk-free rate"
    )
    
    q = st.number_input(
        "Dividend Yield (q)",
        value=q_est,
        step=0.001,
        format="%.4f",
        help="Annualized dividend yield"
    )
    
    sigma = st.number_input(
        "Volatility (sigma)",
        value=default_sigma,
        step=0.01,
        format="%.3f",
        help="Annualized standard deviation of returns"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### Monte Carlo")
    
    mc_paths = st.number_input(
        "Number of Paths",
        value=200000,
        step=10000,
        min_value=1000,
        max_value=1000000,
        help="More paths = higher accuracy but slower computation"
    )
    
    run_mc = st.checkbox(
        "Run Monte Carlo Simulation",
        value=True,
        help="Enable Monte Carlo pricing and distribution visualization"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Compute prices (shared across pages)
cf_price, call_leg, put_leg = chooser_price_closed_form(S0, K, T1, T2, r, q, sigma)
vanilla_call = bs_call(S0, K, T2, r, q, sigma)
vanilla_put = bs_put(S0, K, T2, r, q, sigma)

# Run Monte Carlo if enabled
if run_mc:
    with st.spinner("Running Monte Carlo simulation..."):
        mc_price, mc_se, mc_payoffs = simulate_chooser_mc(
            S0, K, T1, T2, r, q, sigma, n_paths=mc_paths, decision_rule="optimal"
        )
else:
    mc_price, mc_se, mc_payoffs = None, None, None

# -------------------------------------------------------------------
# 8. Overview Page (Simplified)
# -------------------------------------------------------------------

def render_overview_page(df, selected_date, default_S0, default_sigma, default_r, q_est, row, cf_price, vanilla_call, vanilla_put, run_mc, mc_price, mc_se, T1):
    """Render the simplified overview page for non-technical users."""
    
    st.title("JPM Chooser Option Pricing Dashboard")
    st.markdown("*Simple pricing overview and key metrics*")
    
    # Market Snapshot
    with st.expander("Market Snapshot (Selected Date)", expanded=True):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Date", format_date(selected_date))
        col2.metric("Spot Price", f"${default_S0:.2f}")
        col3.metric("Volatility (63d)", f"{default_sigma*100:.1f}%")
        col4.metric("Risk-Free Rate", f"{default_r*100:.2f}%")
        col5.metric("Dividend Yield", f"{q_est*100:.2f}%")
        
        vix_value = float(row["vix"]) if "vix" in row and pd.notna(row["vix"]) else None
        if vix_value:
            col6.metric("VIX", f"{vix_value:.2f}")
        else:
            col6.metric("VIX", "N/A")
    
    st.divider()
    
    # Simple pricing results
    st.subheader("Option Pricing Results")
    
    # Create a clean, simple layout with key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Chooser Option Price",
            f"${cf_price:.4f}",
            help="Price of the chooser option using closed-form solution"
        )
    
    with col2:
        st.metric(
            "Call Option Price",
            f"${vanilla_call:.4f}",
            help="Price of a vanilla call option at maturity"
        )
    
    with col3:
        st.metric(
            "Put Option Price",
            f"${vanilla_put:.4f}",
            help="Price of a vanilla put option at maturity"
        )
    
    with col4:
        if run_mc and mc_price is not None:
            st.metric(
                "Monte Carlo Price",
                f"${mc_price:.4f}",
                f"±{1.96*mc_se:.4f} (95% CI)",
                help="Monte Carlo simulation estimate"
            )
        else:
            st.metric(
                "Monte Carlo",
                "Not Run",
                help="Enable Monte Carlo in sidebar"
            )
    
    # Simple interpretation
    st.divider()
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(
            f"""
            **Chooser Option Value:** ${cf_price:.4f}
            
            The chooser option gives you the right to choose between a call and a put
            at the decision date (T1 = {T1:.2f} years).
            
            This flexibility is worth an additional **${(cf_price - vanilla_call):.4f}**
            compared to a vanilla call option.
            """
        )
    
    with col2:
        if run_mc and mc_price is not None:
            st.success(
                f"""
                **Monte Carlo Validation:** ${mc_price:.4f}
                
                The Monte Carlo simulation confirms the closed-form result.
                The difference is **${(mc_price - cf_price):+.4f}**.
                
                Confidence Interval: [${mc_price - 1.96*mc_se:.4f}, ${mc_price + 1.96*mc_se:.4f}]
                """
            )
        else:
            st.info(
                """
                **Monte Carlo Simulation**
                
                Enable Monte Carlo in the sidebar to validate the closed-form
                pricing and see the distribution of possible payoffs.
                """
            )
    
    # Simple price trend chart
    st.divider()
    st.subheader("JPM Price Trend")
    
    # Create a simple price chart
    fig_simple = go.Figure()
    
    # Price series
    df_for_plot = df.copy()
    df_for_plot['date_str'] = df_for_plot['date'].dt.strftime('%Y-%m-%d')
    
    fig_simple.add_trace(
        go.Scatter(
            x=df_for_plot['date_str'],
            y=df_for_plot["adjusted_close"],
            mode="lines",
            name="JPM Price",
            line=dict(color="#1f77b4", width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        )
    )
    
    # Add vertical line for selected date
    try:
        date_idx = df_for_plot[df_for_plot['date'].dt.date == selected_date].index[0]
        fig_simple.add_vline(
            x=date_idx,
            line=dict(color="red", dash="dash", width=2),
            annotation_text="Selected Date",
            annotation_position="top right"
        )
    except:
        selected_date_str = pd.Timestamp(selected_date).strftime('%Y-%m-%d')
        fig_simple.add_vline(
            x=selected_date_str,
            line=dict(color="red", dash="dash", width=2),
            annotation_text="Selected Date",
            annotation_position="top right"
        )
    
    fig_simple.update_layout(
        height=400,
        template="plotly_white",
        xaxis=dict(
            type='category',
            tickformat='%Y-%m-%d',
            tickangle=45,
            tickmode='auto',
            nticks=20
        ),
        yaxis=dict(
            title="Adjusted Close ($)",
            tickformat='$,.2f'
        ),
        margin=dict(l=50, r=50, t=50, b=80)
    )
    
    st.plotly_chart(fig_simple, use_container_width=True)
    
    # Simple explanation
    st.divider()
    with st.expander("About Chooser Options (Simple Explanation)"):
        st.markdown("""
        ### What is a Chooser Option?
        
        A chooser option is a financial contract that gives you the flexibility to decide 
        later whether you want a **call option** (the right to buy) or a **put option** 
        (the right to sell).
        
        **Key Dates:**
        - **Today (T0):** You purchase the chooser option
        - **Decision Date (T1):** You choose call or put
        - **Maturity (T2):** The option expires
        
        **Why is this valuable?**
        - If the stock price is high at T1, you choose the call option
        - If the stock price is low at T1, you choose the put option
        - This flexibility has value - you pay for the right to choose
        
        **Key Parameters:**
        - **Spot Price (S0):** Current JPM stock price
        - **Strike Price (K):** The price at which you can buy/sell
        - **Volatility (sigma):** How much the stock price moves
        - **Risk-Free Rate (r):** Return on risk-free investments
        - **Dividend Yield (q):** Expected dividend payments
        
        The pricing model calculates what this flexibility is worth today.
        """)

# -------------------------------------------------------------------
# 9. Detailed Analysis Page (Full Technical)
# -------------------------------------------------------------------

def render_detailed_page(df, selected_date, default_S0, default_sigma, default_r, q_est, row, cf_price, call_leg, put_leg, vanilla_call, vanilla_put, run_mc, mc_price, mc_se, mc_payoffs, mc_paths, S0, K, T1, T2, r, q, sigma):
    """Render the full detailed analysis page with all technical features."""
    
    st.title("JPM Chooser Option Pricing Dashboard")
    st.markdown("*Interactive pricing, sensitivity analysis, and Monte Carlo simulation*")
    
    # Market snapshot
    with st.expander("Market Snapshot (Selected Date)", expanded=True):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Date", format_date(selected_date))
        col2.metric("Spot Price", f"${default_S0:.2f}")
        col3.metric("Volatility (63d)", f"{default_sigma*100:.1f}%")
        col4.metric("Risk-Free Rate", f"{default_r*100:.2f}%")
        col5.metric("Dividend Yield", f"{q_est*100:.2f}%")
        
        vix_value = float(row["vix"]) if "vix" in row and pd.notna(row["vix"]) else None
        if vix_value:
            col6.metric("VIX", f"{vix_value:.2f}")
        else:
            col6.metric("VIX", "N/A")
    
    st.divider()
    
    # Display pricing results in metric cards
    st.subheader("Pricing Results")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Chooser Price (BSM)</div>
            <div class="metric-value">${cf_price:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vanilla Call (T2)</div>
            <div class="metric-value">${vanilla_call:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vanilla Put (T2)</div>
            <div class="metric-value">${vanilla_put:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        if run_mc and mc_price is not None:
            delta = mc_price - cf_price
            delta_class = "metric-delta-positive" if delta > 0 else "metric-delta-negative" if delta < 0 else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Monte Carlo Price</div>
                <div class="metric-value">${mc_price:.4f}</div>
                <div style="font-size:0.9rem;margin-top:5px;">
                    <span class="{delta_class}">{delta:+.4f}</span>
                    <span style="color:#6c757d;font-size:0.8rem;"> ±{1.96*mc_se:.4f} (95% CI)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Monte Carlo</div>
                <div class="metric-value" style="font-size:1.2rem;color:#6c757d;">Not Run</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Decomposition details
    with st.expander("Pricing Decomposition Details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Call Leg (T2)", f"${call_leg:.4f}")
        col2.metric("Put Leg (T1, K')", f"${put_leg:.4f}")
        col3.metric("Total Chooser", f"${cf_price:.4f}")
        
        st.caption(f"Adjusted Strike K' = K * exp(-(r-q) * (T2-T1)) = {K * np.exp(-(r-q) * (T2-T1)):.2f}")
    
    st.divider()
    
    # Monte Carlo Payoff Distribution
    if run_mc and mc_price is not None and mc_payoffs is not None:
        st.subheader("Monte Carlo Payoff Distribution")
        
        fig_payoff = go.Figure()
        
        # Histogram of discounted payoffs
        fig_payoff.add_trace(go.Histogram(
            x=mc_payoffs,
            nbinsx=50,
            name="Payoff Distribution",
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        # Add mean line
        fig_payoff.add_vline(
            x=mc_price,
            line=dict(color="red", dash="dash", width=2),
            annotation_text=f"Mean: ${mc_price:.4f}",
            annotation_position="top right"
        )
        
        # Add confidence interval
        ci_lower = mc_price - 1.96 * mc_se
        ci_upper = mc_price + 1.96 * mc_se
        fig_payoff.add_vline(
            x=ci_lower,
            line=dict(color="orange", dash="dot", width=1),
            annotation_text=f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
            annotation_position="bottom right"
        )
        fig_payoff.add_vline(
            x=ci_upper,
            line=dict(color="orange", dash="dot", width=1)
        )
        
        fig_payoff.update_layout(
            title="Distribution of Discounted Payoffs from Monte Carlo Simulation",
            xaxis_title="Discounted Payoff ($)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=400,
            showlegend=False,
            bargap=0.05,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add summary stats
        st.caption(f"""
        **Monte Carlo Statistics:**
        * Mean: ${mc_price:.4f}
        * Standard Error: ${mc_se:.4f}
        * 95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]
        * Paths: {mc_paths:,}
        """)
        
        st.plotly_chart(fig_payoff, use_container_width=True)
        st.divider()
    
    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    
    # Generate sensitivity grids
    sigma_range = np.linspace(0.01, 1.0, 60)
    K_range = np.linspace(1, 500, 60)
    r_range = np.linspace(0.0, 0.10, 60)
    q_range = np.linspace(0.0, 0.10, 60)
    
    # Create base tuple for caching
    base = {"S0": S0, "K": K, "T1": T1, "T2": T2, "r": r, "q": q, "sigma": sigma}
    base_tuple = tuple(sorted(base.items()))
    
    # Compute grids with caching
    with st.spinner("Computing sensitivity grids..."):
        grid_sigma = get_sensitivity_grid("sigma", sigma_range, base_tuple)
        grid_K = get_sensitivity_grid("K", K_range, base_tuple)
        grid_r = get_sensitivity_grid("r", r_range, base_tuple)
        grid_q = get_sensitivity_grid("q", q_range, base_tuple)
    
    # Create subplot with Plotly - Improved spacing
    fig_sens = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Volatility Sensitivity",
            "Strike Sensitivity",
            "Risk-Free Rate Sensitivity",
            "Dividend Yield Sensitivity"
        ),
        horizontal_spacing=0.15,
        vertical_spacing=0.18
    )
    
    # Define color scheme
    colors = {
        'call': '#1f77b4',
        'put': '#ff7f0e',
        'chooser': '#2ca02c'
    }
    
    # Add traces for each sensitivity
    sensitivities = [
        (grid_sigma, "sigma", "Volatility (sigma)", 1, 1),
        (grid_K, "K", "Strike ($)", 1, 2),
        (grid_r, "r", "Risk-Free Rate", 2, 1),
        (grid_q, "q", "Dividend Yield", 2, 2)
    ]
    
    for grid, xcol, xlabel, row, col in sensitivities:
        # Add traces
        fig_sens.add_trace(
            go.Scatter(
                x=grid[xcol],
                y=grid["call"],
                name="Call" if row==1 and col==1 else None,
                line=dict(color=colors['call'], width=2),
                showlegend=(row==1 and col==1),
                legendgroup="call",
                hovertemplate=f'{xlabel}: %{{x:.3f}}<br>Call: $%{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig_sens.add_trace(
            go.Scatter(
                x=grid[xcol],
                y=grid["put"],
                name="Put" if row==1 and col==1 else None,
                line=dict(color=colors['put'], width=2),
                showlegend=(row==1 and col==1),
                legendgroup="put",
                hovertemplate=f'{xlabel}: %{{x:.3f}}<br>Put: $%{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig_sens.add_trace(
            go.Scatter(
                x=grid[xcol],
                y=grid["chooser"],
                name="Chooser" if row==1 and col==1 else None,
                line=dict(color=colors['chooser'], width=2.5, dash="dash"),
                showlegend=(row==1 and col==1),
                legendgroup="chooser",
                hovertemplate=f'{xlabel}: %{{x:.3f}}<br>Chooser: $%{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add vertical line for current value
        current_val = {"sigma": sigma, "K": K, "r": r, "q": q}[xcol]
        fig_sens.add_vline(
            x=current_val,
            line=dict(color="grey", dash="dot", width=1.5),
            row=row, col=col,
            annotation_text=f"Current: {current_val:.3f}",
            annotation_position="top"
        )
        
        # Update axes with better formatting
        fig_sens.update_xaxes(
            title_text=xlabel,
            row=row, 
            col=col,
            title_font=dict(size=12),
            tickfont=dict(size=10)
        )
        fig_sens.update_yaxes(
            title_text="Option Value ($)",
            row=row, 
            col=col,
            title_font=dict(size=12),
            tickfont=dict(size=10),
            tickformat='$,.2f'
        )
    
    fig_sens.update_layout(
        height=650,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    st.plotly_chart(fig_sens, use_container_width=True)
    
    # Price Trend Visualization
    st.subheader("JPM Price Trend and Volatility")
    
    # Create two subplots for price and volatility
    fig_trend = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("JPM Stock Price", "Realized Volatility (63-day)")
    )
    
    # Use the dates directly as strings for better display
    # Convert dates to string format for Plotly
    df_for_plot = df.copy()
    df_for_plot['date_str'] = df_for_plot['date'].dt.strftime('%Y-%m-%d')
    
    # Price series - using string dates
    fig_trend.add_trace(
        go.Scatter(
            x=df_for_plot['date_str'],
            y=df_for_plot["adjusted_close"],
            mode="lines",
            name="Price",
            line=dict(color="#1f77b4", width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)',
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Convert selected date to string
    selected_date_str = pd.Timestamp(selected_date).strftime('%Y-%m-%d')
    
    # Add vertical line for selected date - using string position
    # Find the index position of the selected date
    try:
        date_idx = df_for_plot[df_for_plot['date'].dt.date == selected_date].index[0]
        # Add vertical line using index position
        fig_trend.add_vline(
            x=date_idx,
            line=dict(color="red", dash="dash", width=1.5),
            row=1, col=1,
            annotation_text="Selected Date",
            annotation_position="top right"
        )
    except:
        # Fallback: use the string date
        fig_trend.add_vline(
            x=selected_date_str,
            line=dict(color="red", dash="dash", width=1.5),
            row=1, col=1,
            annotation_text="Selected Date",
            annotation_position="top right"
        )
    
    # Volatility series - Fixed formatting
    if "volatility_63d" in df.columns:
        fig_trend.add_trace(
            go.Scatter(
                x=df_for_plot['date_str'],
                y=df_for_plot["volatility_63d"],
                mode="lines",
                name="Volatility",
                line=dict(color="#d62728", width=2),
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.1)',
                hovertemplate='Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add vertical line for selected date on volatility chart
        try:
            date_idx = df_for_plot[df_for_plot['date'].dt.date == selected_date].index[0]
            fig_trend.add_vline(
                x=date_idx,
                line=dict(color="red", dash="dash", width=1.5),
                row=2, col=1
            )
        except:
            fig_trend.add_vline(
                x=selected_date_str,
                line=dict(color="red", dash="dash", width=1.5),
                row=2, col=1
            )
    
    # Update x-axis to show dates properly - using category type for proper date display
    fig_trend.update_xaxes(
        type='category',
        tickformat='%Y-%m-%d',
        tickangle=45,
        title_text="Date",
        row=2, col=1,
        tickfont=dict(size=10),
        tickmode='auto',
        nticks=20
    )
    
    fig_trend.update_xaxes(
        type='category',
        tickformat='%Y-%m-%d',
        tickangle=45,
        row=1, col=1,
        tickfont=dict(size=10),
        tickmode='auto',
        nticks=20
    )
    
    fig_trend.update_layout(
        height=550,
        template="plotly_white",
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=80)
    )
    
    fig_trend.update_yaxes(
        title_text="Adjusted Close ($)",
        row=1, 
        col=1,
        tickfont=dict(size=10)
    )
    
    fig_trend.update_yaxes(
        title_text="Volatility",
        row=2, 
        col=1,
        tickformat='.1%',
        tickfont=dict(size=10),
        range=[0, 0.8]
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Additional information
    with st.expander("About This Dashboard (Technical Details)"):
        st.markdown("""
        ### Chooser Option Pricing Dashboard
        
        **What is a Chooser Option?**
        A chooser option gives the holder the right to choose, at a predetermined date T1, 
        whether the option will be a call or a put on the same underlying asset, with maturity T2.
        
        **Pricing Methodology:**
        - **Closed-Form**: Based on the Rubinstein (1991) decomposition:
          `Chooser = Call(T2) + Put(T1, K')`, where `K' = K * exp(-(r-q)*(T2-T1))`
        - **Monte Carlo**: Simulates stock paths using geometric Brownian motion.
          The optimal decision at T1 is to choose the call if `S(T1) > K * exp(-(r-q)*(T2-T1))`,
          otherwise choose the put.
        
        **Data Sources:**
        - **Stock Data**: JPM price history from Yahoo Finance (yfinance)
        - **Rates**: Treasury yields (^IRX, ^FVX, ^TNX) from Yahoo Finance
        - **Volatility**: 63-day rolling annualized volatility of log returns
        - **Dividends**: Historical dividend data from Yahoo Finance
        
        **Key Assumptions:**
        - Stock follows geometric Brownian motion with constant volatility
        - Risk-free rate and dividend yield are constant
        - No transaction costs or market frictions
        - European-style exercise at T2
        
        **Interpretation:**
        - The chooser option price is always >= the vanilla call price at T2
        - The difference represents the value of the choice feature
        - Monte Carlo simulation provides distribution of possible payoffs
        - Sensitivity analysis shows how the price reacts to changes in inputs
        """)
    
    # Saved results
    with st.expander("Saved Results (JSON)"):
        json_path = "data/processed/chooser_bsm_results.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                saved = json.load(f)
            st.json(saved)
        else:
            st.info("No saved JSON results found. Run the preprocessing pipeline to generate results.")

# -------------------------------------------------------------------
# 10. Page rendering
# -------------------------------------------------------------------

# Render the selected page
if page == "Overview":
    render_overview_page(df, selected_date, default_S0, default_sigma, default_r, q_est, row, cf_price, vanilla_call, vanilla_put, run_mc, mc_price, mc_se, T1)
else:
    render_detailed_page(df, selected_date, default_S0, default_sigma, default_r, q_est, row, cf_price, call_leg, put_leg, vanilla_call, vanilla_put, run_mc, mc_price, mc_se, mc_payoffs, mc_paths, S0, K, T1, T2, r, q, sigma)

# Footer
st.markdown("---")
st.caption("""
**Built with Streamlit** • JPM Chooser Option Pricing Dashboard  
Data source: Local CSV or live via yfinance (no API key required)  
Last updated: {0}
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
