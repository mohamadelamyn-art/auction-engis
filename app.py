import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MOCK DATA GENERATOR (High-Fidelity Institutional Simulation)
# =============================================================================
def generate_institutional_mock_data(symbol="XAUUSD", years=10):
    """
    Generates vectorized intraday mock data with realistic auction characteristics.
    Ensures proper UTC timezone awareness and robust DatetimeIndex construction.
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=years * 365)
    
    # Generate business days only (No weekends)
    dates = pd.bdate_range(start=start_date, periods=years * 252, freq='B')
    
    # Intraday schedule: 08:00 to 20:00 UTC (Core Session for Demo Speed)
    times = pd.date_range(start="08:00", end="20:00", freq="5min").time
    
    # CORRECTED: Explicit conversion of dates to date objects for datetime.combine
    date_objects = [d.date() for d in dates]
    
    # Robust DatetimeIndex construction
    idx = pd.MultiIndex.from_product([date_objects, times], names=["Date", "Time"])
    flat_index = [datetime.combine(d, t) for d, t in idx]
    
    # CRITICAL FIX: Use tz_localize instead of passing tz to constructor for Pandas 2.0+ compatibility
    df = pd.DataFrame(index=pd.DatetimeIndex(flat_index, name="Timestamp").tz_localize("UTC"))
    
    n_bars = len(df)
    
    # Vectorized Price Generation (Geometric Brownian Motion + Regime Switching)
    returns = np.random.normal(0.00002, 0.0012, n_bars)
    
    # Inject Auction Regimes at Daily Level
    regime = np.random.choice([0, 1], size=len(dates), p=[0.4, 0.6]) # 0=Trend, 1=Rotate
    regime_map = np.repeat(regime, len(times))
    
    # Adjust volatility and drift based on regime
    vol_mult = np.where(regime_map == 0, 2.8, 0.5)
    drift = np.where(regime_map == 0, 0.00015, 0.0)
    adjusted_returns = (returns * vol_mult) + drift
        base_price = 2000.0 if symbol == "XAUUSD" else 15000.0
    prices = base_price * np.exp(np.cumsum(adjusted_returns))
    
    # Construct OHLC with realistic wicks
    noise = np.abs(np.random.normal(0, 0.4, n_bars))
    df["Open"] = prices + np.random.normal(0, 0.15, n_bars)
    df["High"] = np.maximum(prices, df["Open"]) + noise
    df["Low"] = np.minimum(prices, df["Open"]) - noise
    df["Close"] = prices
    
    # Inject random missing bars to test gap repair logic
    mask = np.random.random(n_bars) > 0.995
    df.loc[mask, ["Open", "High", "Low", "Close"]] = np.nan
    
    return df.reset_index()

# =============================================================================
# 2. CORE ENGINE: PREPROCESSING & AUCTION MATH
# =============================================================================
class AuctionEngine:
    def __init__(self, df: pd.DataFrame, max_sl_pips: float):
        self.raw_df = df.copy()
        self.max_sl_points = float(max_sl_pips)
        self.df = self._preprocess()
        
    def _preprocess(self):
        """Vectorized cleaning, gap repair, and UTC standardization."""
        df = self.raw_df.copy()
        
        # Ensure UTC timezone awareness
        if df["Timestamp"].dt.tz is None:
            df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
        else:
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")
            
        df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])
        df = df[df["Timestamp"].dt.dayofweek < 5]
        
        # CRITICAL: Forward-fill then Backward-fill for gap repair
        # Apply only to price columns to avoid touching the Timestamp column unnecessarily
        price_cols = ["Open", "High", "Low", "Close"]
        df[price_cols] = df[price_cols].ffill().bfill()
        
        # Derived Vectorized Columns
        df["Date"] = df["Timestamp"].dt.normalize()
        df["Hour"] = df["Timestamp"].dt.hour
        
        return df

    def compute_daily_auction_structures(self):        """Vectorized calculation of IB, POC, VA, and Day Type Classification."""
        df = self.df.copy()
        
        # --- Initial Balance (First 60 mins: 08:00-09:00 UTC) ---
        ib_mask = df["Hour"] == 8
        ib_stats = df[ib_mask].groupby("Date").agg(
            IB_High=("High", "max"),
            IB_Low=("Low", "min")
        )
        ib_stats["IB_Range"] = ib_stats["IB_High"] - ib_stats["IB_Low"]
        
        # --- Daily Stats ---
        daily_stats = df.groupby("Date").agg(
            Day_High=("High", "max"),
            Day_Low=("Low", "min")
        )
        daily_stats["Day_Range"] = daily_stats["Day_High"] - daily_stats["Day_Low"]
        
        structures = ib_stats.join(daily_stats, how="inner")
        structures["Expansion_Ratio"] = structures["Day_Range"] / structures["IB_Range"].replace(0, np.nan)
        structures["Regime"] = np.where(structures["Expansion_Ratio"] > 1.5, "Trending", "Mean-Reverting")
        
        # --- Optimized POC & Value Area Calculation ---
        poc_va_list = []
        bin_size = 0.5 
        
        for date, group in df.groupby("Date"):
            prices = group["Close"].values
            if len(prices) == 0: 
                continue
            
            min_p, max_p = float(prices.min()), float(prices.max())
            
            # Safety check for flat markets
            if max_p <= min_p:
                poc_va_list.append({"Date": date, "POC": min_p, "VAH": min_p, "VAL": min_p})
                continue
                
            bins = np.arange(min_p, max_p + bin_size, bin_size)
            counts, edges = np.histogram(prices, bins=bins)
            
            if counts.sum() == 0:
                continue
                
            poc_idx = int(np.argmax(counts))
            poc = (edges[poc_idx] + edges[poc_idx+1]) / 2.0
            
            total_vol = float(counts.sum())
            target_vol = total_vol * 0.70
            accumulated = float(counts[poc_idx])            upper_idx, lower_idx = poc_idx, poc_idx
            
            while accumulated < target_vol:
                can_go_up = upper_idx < len(counts) - 1
                can_go_down = lower_idx > 0
                
                if not can_go_up and not can_go_down:
                    break
                    
                up_vol = counts[upper_idx+1] if can_go_up else -1
                dn_vol = counts[lower_idx-1] if can_go_down else -1
                
                if up_vol >= dn_vol and can_go_up:
                    upper_idx += 1
                    accumulated += counts[upper_idx]
                elif can_go_down:
                    lower_idx -= 1
                    accumulated += counts[lower_idx]
                elif can_go_up:
                    upper_idx += 1
                    accumulated += counts[upper_idx]
                else:
                    break
                    
            vah = edges[min(upper_idx+1, len(edges)-1)]
            val = edges[max(lower_idx, 0)]
            
            poc_va_list.append({"Date": date, "POC": poc, "VAH": vah, "VAL": val})
            
        if poc_va_list:
            poc_va_df = pd.DataFrame(poc_va_list).set_index("Date")
            structures = structures.join(poc_va_df, how="left")
        else:
            structures["POC"] = np.nan
            structures["VAH"] = np.nan
            structures["VAL"] = np.nan
        
        self.structures = structures
        return structures

# =============================================================================
# 3. VECTORIZED SEARCH ENGINE & STRATEGY DISCOVERY
# =============================================================================
class StrategyDiscovery:
    def __init__(self, df: pd.DataFrame, structures: pd.DataFrame, max_sl: float):
        # Explicit copy to prevent unintended modification of source data
        if "Timestamp" in df.columns:
            self.df = df.set_index("Timestamp").copy()
        else:
            self.df = df.copy()        self.structures = structures.copy()
        self.max_sl = float(max_sl)
        
    def run_brutal_search(self, min_trades=100):
        """Meshgrid parameter scan with strict institutional filters."""
        entry_hours = np.arange(9, 20)
        sl_fractions = np.array([0.25, 0.5, 0.75, 1.0])
        rr_targets = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
        regimes = ["Trending", "Mean-Reverting"]
        
        results = []
        
        # Pre-compute forward momentum threshold safely
        self.df["Fwd_Mom"] = self.df["Close"].shift(-3).sub(self.df["Close"]).abs()
        global_std = self.df["Close"].rolling(100, min_periods=1).std().mean()
        mom_threshold = float(global_std) * 0.5 if not np.isnan(global_std) else 0.1
        
        for regime in regimes:
            valid_dates = self.structures[self.structures["Regime"] == regime].index
            if len(valid_dates) == 0:
                continue
                
            subset = self.df[self.df.index.normalize().isin(valid_dates)]
            if subset.empty:
                continue
            
            for hour in entry_hours:
                hour_subset = subset[subset.index.hour == hour]
                if len(hour_subset) < min_trades:
                    continue
                
                entry_prices = hour_subset["Close"].values.astype(float)
                fwd_mom = hour_subset["Fwd_Mom"].values.astype(float)
                trade_dates = hour_subset.index.normalize()
                
                # Map daily structures to trade entries vectorially
                try:
                    daily_ranges = self.structures.loc[trade_dates]
                except KeyError:
                    continue
                
                upside_room = daily_ranges["Day_High"].values.astype(float) - entry_prices
                downside_room = entry_prices - daily_ranges["Day_Low"].values.astype(float)
                valid_entries_mask = fwd_mom > mom_threshold
                
                for sl_frac in sl_fractions:
                    current_sl = self.max_sl * sl_frac
                    if current_sl <= 0.01:
                        continue
                                        for rr in rr_targets:
                        tp_dist = current_sl * rr
                        
                        active_mask = valid_entries_mask.copy()
                        if active_mask.sum() < min_trades:
                            continue
                        
                        wins = (upside_room[active_mask] >= tp_dist)
                        losses = (downside_room[active_mask] >= current_sl)
                        
                        # First-touch logic proxy based on confirmed forward momentum
                        outcomes = np.zeros(wins.shape[0])
                        outcomes[wins] = 1
                        outcomes[losses & ~wins] = -1
                        
                        decided = outcomes != 0
                        final_outcomes = outcomes[decided]
                        
                        n_trades = len(final_outcomes)
                        if n_trades < min_trades:
                            continue
                        
                        win_rate = float((final_outcomes == 1).sum()) / n_trades
                        expectancy = (win_rate * tp_dist) - ((1.0 - win_rate) * current_sl)
                        
                        # BRUTAL ACCEPTANCE FILTER
                        if not (0.45 <= win_rate <= 0.65):
                            continue
                        if not (1.5 <= rr <= 3.5):
                            continue
                        if expectancy <= 0:
                            continue
                        
                        # Quality Score (0-100)
                        stability_bonus = min(n_trades / 1000.0, 1.0) * 20.0
                        wr_score = max(0.0, (1.0 - abs(win_rate - 0.55) / 0.1)) * 40.0
                        exp_score = min(expectancy / (self.max_sl * 0.5 + 0.01), 1.0) * 40.0
                        quality = max(0.0, min(100.0, stability_bonus + wr_score + exp_score))
                        
                        results.append({
                            "Regime": regime,
                            "Entry_Hour_UTC": int(hour),
                            "SL_Points": round(current_sl, 2),
                            "RR_Target": float(rr),
                            "Trades": int(n_trades),
                            "Win_Rate": round(win_rate * 100, 2),
                            "Expectancy": round(expectancy, 2),
                            "Quality_Score": round(quality, 1)
                        })
                if not results:
            return pd.DataFrame()
            
        res_df = pd.DataFrame(results)
        return res_df.sort_values("Quality_Score", ascending=False).head(5).reset_index(drop=True)

# =============================================================================
# 4. INSTITUTIONAL RISK SIMULATOR
# =============================================================================
class RiskSimulator:
    @staticmethod
    def simulate_equity_curve(df, structures, strategy_params, starting_balance, risk_pct=0.01):
        """Simulates equity curve with 4% Daily Drawdown Prop Firm Guardrail."""
        balance = float(starting_balance)
        equity_log = []
        
        regime = strategy_params["Regime"]
        hour = int(strategy_params["Entry_Hour_UTC"])
        sl = float(strategy_params["SL_Points"])
        rr = float(strategy_params["RR_Target"])
        tp = sl * rr
        
        if sl <= 0:
            return pd.DataFrame()
        
        valid_dates = structures[structures["Regime"] == regime].index
        if len(valid_dates) == 0:
            return pd.DataFrame()
            
        # Ensure df is a copy to prevent modification
        df = df.copy()
        if df.index.name != "Timestamp" and "Timestamp" in df.columns:
            df = df.set_index("Timestamp")
            
        trade_days = df[(df.index.normalize().isin(valid_dates)) & (df.index.hour == hour)].copy()
        
        if trade_days.empty:
            return pd.DataFrame()
            
        daily_start_bal = balance
        current_day = None
        
        for ts, row in trade_days.iterrows():
            day = ts.normalize()
            
            if day != current_day:
                current_day = day
                daily_start_bal = balance
                
            # Prop Firm Guardrail: 4% Max Daily DD Lockout            if balance < (daily_start_bal * 0.96):
                equity_log.append({"Timestamp": ts, "Balance": balance})
                continue
                
            risk_amount = balance * risk_pct
            
            try:
                struct_row = structures.loc[day]
                upside = float(struct_row["Day_High"]) - float(row["Close"])
                downside = float(row["Close"]) - float(struct_row["Day_Low"])
            except (KeyError, TypeError):
                continue
            
            pnl = 0.0
            if upside >= tp:
                pnl = risk_amount * rr
            elif downside >= sl:
                pnl = -risk_amount
                
            balance += pnl
            equity_log.append({"Timestamp": ts, "Balance": balance})
            
        return pd.DataFrame(equity_log)

# =============================================================================
# 5. STREAMLIT UI & DASHBOARD
# =============================================================================
def main():
    st.set_page_config(page_title="Institutional Auction Engine", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem;}
        [data-testid="stMetricValue"] {font-size: 28px; color: #00d4ff;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏛️ Market Auction & Quantitative Research Engine")
    st.caption("Phase 1 Core | Pure Math Discovery | No Lagging Indicators | Prop Firm Compliant")
    
    with st.sidebar:
        st.header("⚙️ Engine Configuration")
        uploaded_file = st.file_uploader("Upload Intraday CSV (M1/M5)", type=["csv"])
        
        starting_balance = st.number_input("Starting Balance ($)", value=100000, step=10000, min_value=1000)
        max_sl = st.slider("Max Allowed SL (Points/Pips)", 10, 200, 50, help="Strict Cap: Cannot exceed 200")
        min_trades = st.slider("Minimum Trade Count Filter", 50, 500, 100)
        
        use_mock = st.toggle("Use High-Fidelity Mock Data", value=True)
            if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Standardize column names
            df.columns = [c.strip().lower() for c in df.columns]
            col_mapping = {
                'timestamp': 'Timestamp',
                'date': 'Date',
                'time': 'Time',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
            
            if "Timestamp" not in df.columns:
                # Try to construct from Date/Time if available
                if "Date" in df.columns and "Time" in df.columns:
                    df["Timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
                    df = df.drop(columns=["Date", "Time"])
                else:
                    st.error("CSV must contain 'Timestamp' column or both 'Date' and 'Time' columns.")
                    st.stop()
            else:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
                st.stop()
                
            use_mock = False
        except Exception as e:
            st.error(f"CSV Load Error: {e}")
            st.stop()
    elif use_mock:
        with st.spinner("Generating 10-Year Institutional Grade Mock Dataset..."):
            df = generate_institutional_mock_data()
    else:
        st.warning("Please upload a CSV file or enable Mock Data generation.")
        st.stop()
        
    try:
        engine = AuctionEngine(df, max_sl)
        processed_df = engine.df
        structures = engine.compute_daily_auction_structures()
        
        col1, col2, col3 = st.columns(3)        trending_pct = (structures["Regime"] == "Trending").mean() * 100 if len(structures) > 0 else 0
        col1.metric("Trending Days", f"{trending_pct:.1f}%")
        col2.metric("Mean-Reverting Days", f"{100-trending_pct:.1f}%")
        col3.metric("Total Bars Processed", f"{len(processed_df):,}")
        
        st.divider()
        
        st.subheader("🔬 Brutal Strategy Discovery Engine")
        with st.spinner("Scanning thousands of mathematical combinations via NumPy meshgrid..."):
            discovery = StrategyDiscovery(processed_df, structures, max_sl)
            top_strategies = discovery.run_brutal_search(min_trades=min_trades)
            
        if top_strategies.empty:
            st.error("No strategies met the BRUTAL acceptance criteria. Try relaxing Min Trades or adjusting Max SL.")
        else:
            st.dataframe(top_strategies, use_container_width=True, hide_index=True)
            
            # Encode to UTF-8 for bulletproof Streamlit download compatibility
            csv = top_strategies.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Top Strategies Metrics (.csv)",
                data=csv,
                file_name="auction_engine_top_strategies.csv",
                mime="text/csv"
            )
            
            st.divider()
            st.subheader("📈 Institutional Equity Curve (Top Ranked System)")
            
            best_strat = top_strategies.iloc[0]
            sim = RiskSimulator()
            equity_df = sim.simulate_equity_curve(
                processed_df, structures, best_strat, starting_balance
            )
            
            if not equity_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df["Timestamp"], 
                    y=equity_df["Balance"],
                    mode='lines',
                    line=dict(color='#00d4ff', width=2),
                    name='Account Equity'
                ))
                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="Time",
                    yaxis_title="Balance ($)",                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                final_bal = equity_df["Balance"].iloc[-1]
                roi = ((final_bal - starting_balance) / starting_balance) * 100
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Final Simulated Balance", f"${final_bal:,.2f}")
                m_col2.metric("Total ROI", f"{roi:.2f}%")
            else:
                st.info("Insufficient data to simulate equity curve for selected parameters.")
                
    except Exception as e:
        st.error(f"Engine Error: {str(e)}")
        with st.expander("View Technical Traceback"):
            st.exception(e)

if __name__ == "__main__":
    main()
