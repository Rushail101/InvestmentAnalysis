import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from itertools import combinations
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Portfolio Optimizer Pro", layout="wide", page_icon="💼")

# Sector definitions (matching the R code)
SECTORS = {
    "METAL": ["APLAPOLLO.NS", "ADANIENT.NS", "JSWSTEEL.NS", "WELCORP.NS", "SAIL.NS", 
              "JINDALSTEL.NS", "LLOYDSME.NS", "JSL.NS", "HINDZINC.NS", "NMDC.NS", 
              "HINDALCO.NS", "VEDL.NS", "TATASTEEL.NS", "NATIONALUM.NS", "HINDCOPPER.NS"],
    
    "PHARMA": ["TORNTPHARM.NS", "MANKIND.NS", "JBCHEPHARM.NS", "ALKEM.NS", "GLENMARK.NS",
               "ABBOTINDIA.NS", "DRREDDY.NS", "SUNPHARMA.NS", "ZYDUSLIFE.NS", "DIVISLAB.NS",
               "NATCOPHARM.NS", "BIOCON.NS", "AJANTPHARM.NS", "CIPLA.NS", "GLAND.NS",
               "AUROPHARMA.NS", "LUPIN.NS", "IPCALAB.NS", "GRANULES.NS", "LAURUSLABS.NS"],
    
    "BANKS": ["HDFCBANK.NS", "FEDERALBNK.NS", "RBLBANK.NS", "ICICIBANK.NS", "AXISBANK.NS",
              "KOTAKBANK.NS", "BANDHANBNK.NS", "INDUSINDBK.NS", "YESBANK.NS", "IDFCFIRSTB.NS"],
    
    "REALTY": ["RAYMOND.NS", "OBEROIRLTY.NS", "LODHA.NS", "PHOENIXLTD.NS", "BRIGADE.NS",
               "DLF.NS", "GODREJPROP.NS", "SOBHA.NS", "PRESTIGE.NS", "ANANTRAJ.NS"],
    
    "HEALTHCARE": ["TORNTPHARM.NS", "MAXHEALTH.NS", "APOLLOHOSP.NS", "MANKIND.NS", "FORTIS.NS",
                   "ALKEM.NS", "GLENMARK.NS", "SYNGENE.NS", "ABBOTINDIA.NS", "DRREDDY.NS",
                   "SUNPHARMA.NS", "ZYDUSLIFE.NS", "DIVISLAB.NS", "BIOCON.NS", "CIPLA.NS",
                   "AUROPHARMA.NS", "LUPIN.NS", "IPCALAB.NS", "GRANULES.NS", "LAURUSLABS.NS"],
    
    "DIVIDEND": ["CASTROLIND.NS", "COLPAL.NS", "ITC.NS", "HINDUNILVR.NS", "SHRIRAMFIN.NS",
                 "OFSS.NS", "IOC.NS", "MANAPPURAM.NS", "INDIANB.NS", "HINDPETRO.NS",
                 "BRITANNIA.NS", "BANKBARODA.NS", "NHPC.NS", "SBIN.NS", "POWERGRID.NS",
                 "IEX.NS", "ASHOKLEY.NS", "LICHSGFIN.NS", "NTPC.NS", "PAGEIND.NS",
                 "HEROMOTOCO.NS", "IRFC.NS", "INFY.NS", "BPCL.NS", "HUDCO.NS", "TCS.NS",
                 "COALINDIA.NS", "HCLTECH.NS", "PFC.NS", "TECHM.NS", "HDFCAMC.NS", "CESC.NS",
                 "GAIL.NS", "WIPRO.NS", "CYIENT.NS", "CANBK.NS", "IGL.NS", "PETRONET.NS",
                 "RECLTD.NS", "SAIL.NS", "CUMMINSIND.NS", "GICRE.NS", "MPHASIS.NS",
                 "UNIONBANK.NS", "OIL.NS", "HINDZINC.NS", "ONGC.NS", "NMDC.NS", "VEDL.NS",
                 "NATIONALUM.NS"],
    
    "GROWTH": ["GODREJCP.NS", "APOLLOHOSP.NS", "HINDUNILVR.NS", "CIPLA.NS", "EICHERMOT.NS",
               "SUNPHARMA.NS", "TVSMOTOR.NS", "DIXON.NS", "DIVISLAB.NS", "M&M.NS", "WIPRO.NS",
               "INFY.NS", "TECHM.NS", "HEROMOTOCO.NS", "PERSISTENT.NS"],
    
    "DEFENCE": ["DATAPATTNS.NS", "UNIMECH.NS", "MTARTECH.NS", "BDL.NS", "MAZDOCK.NS",
                "CYIENTDLM.NS", "MIDHANI.NS", "DYNAMATECH.NS", "COCHINSHIP.NS", "PARAS.NS",
                "SOLARINDS.NS", "BEL.NS", "HAL.NS", "ASTRAMICRO.NS", "GRSE.NS", "ZENTEC.NS",
                "BEML.NS", "DCXINDIA.NS"],
    
    "AUTO": ["TIINDIA.NS", "BAJAJ-AUTO.NS", "M&M.NS", "EXIDEIND.NS", "TVSMOTOR.NS",
             "BOSCHLTD.NS", "HEROMOTOCO.NS", "TATAMOTORS.NS", "MOTHERSON.NS", "EICHERMOT.NS",
             "BALKRISIND.NS", "BHARATFORG.NS", "MARUTI.NS", "ASHOKLEY.NS"],
    
    "FMCG": ["COLPAL.NS", "GODREJCP.NS", "NESTLEIND.NS", "MARICO.NS", "DABUR.NS",
             "EMAMILTD.NS", "UNITDSPR.NS", "HINDUNILVR.NS", "BRITANNIA.NS", "RADICO.NS",
             "TATACONSUM.NS", "VBL.NS", "ITC.NS", "UBL.NS"]
}

# Helper functions
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data_parallel(tickers, start_date, end_date):
    """Fetch historical data for multiple tickers in parallel"""
    data = {}
    failed = []
    
    def fetch_single_stock(ticker):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            if not df.empty and len(df) > 20:
                return ticker, df['Close'], None
            else:
                return ticker, None, "Insufficient data"
        except Exception as e:
            return ticker, None, str(e)
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_single_stock, ticker): ticker for ticker in tickers}
        
        for future in as_completed(futures):
            ticker, close_data, error = future.result()
            if close_data is not None:
                data[ticker] = close_data
            else:
                failed.append(ticker)
    
    return pd.DataFrame(data), failed

def calculate_monthly_returns(prices):
    """Calculate monthly returns from daily prices - optimized"""
    monthly_prices = prices.resample('M').last()
    returns = monthly_prices.pct_change().dropna()
    return returns

def optimize_portfolio_weights_vectorized(mean_returns, cov_matrix, n_assets, min_weight=0.1, max_weight=1.0, n_portfolios=3000):
    """
    Vectorized portfolio optimization - much faster
    """
    # Risk-free rate
    rf_monthly = 0.068 / 12
    
    # Generate all random weights at once
    np.random.seed(42)  # For reproducibility
    weights_matrix = np.random.random((n_portfolios, n_assets))
    weights_matrix = weights_matrix / weights_matrix.sum(axis=1, keepdims=True)
    
    # Apply minimum weight constraint
    weights_matrix = np.maximum(weights_matrix, min_weight)
    weights_matrix = weights_matrix / weights_matrix.sum(axis=1, keepdims=True)
    
    # Filter by max weight constraint
    valid_mask = (weights_matrix <= max_weight).all(axis=1)
    weights_matrix = weights_matrix[valid_mask]
    
    if len(weights_matrix) == 0:
        # Fallback: try equal weights if within constraints
        equal_weight = 1.0 / n_assets
        if min_weight <= equal_weight <= max_weight:
            return np.ones(n_assets) / n_assets
        # Otherwise, use proportional weights respecting min constraint
        base_weights = np.ones(n_assets) * min_weight
        remaining = 1.0 - (min_weight * n_assets)
        if remaining > 0:
            base_weights[0] += remaining  # Add remaining to first stock
        return base_weights / base_weights.sum()
    
    # Vectorized portfolio calculations
    portfolio_returns = weights_matrix @ mean_returns
    portfolio_variance = np.sum((weights_matrix @ cov_matrix) * weights_matrix, axis=1)
    portfolio_std = np.sqrt(portfolio_variance)
    
    # Avoid division by zero
    portfolio_std = np.maximum(portfolio_std, 1e-10)
    
    # Sharpe ratios
    sharpe_ratios = (portfolio_returns - rf_monthly) / portfolio_std
    sharpe_ratios = np.nan_to_num(sharpe_ratios, nan=-999.0, posinf=-999.0, neginf=-999.0)
    
    # Find maximum Sharpe ratio
    max_sharpe_idx = np.argmax(sharpe_ratios)
    optimal_weights = weights_matrix[max_sharpe_idx]
    
    return optimal_weights

def calculate_portfolio_metrics_fast(returns, weights):
    """Optimized portfolio metrics calculation"""
    # Portfolio returns - vectorized
    portfolio_returns = returns @ weights
    
    # Check if we have valid returns
    if len(portfolio_returns) == 0 or portfolio_returns.std() == 0:
        return None
    
    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Sharpe Ratio (annualized)
    rf_rate = 0.068 / 12
    excess_returns = portfolio_returns - rf_rate
    sharpe_ratio = np.sqrt(12) * (excess_returns.mean() / portfolio_returns.std()) if portfolio_returns.std() > 0 else 0
    
    # CAGR
    total_return = cumulative_returns.iloc[-1] - 1
    n_years = len(portfolio_returns) / 12
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Maximum Drawdown - vectorized
    cumulative_array = cumulative_returns.values
    running_max = np.maximum.accumulate(cumulative_array)
    drawdown = (cumulative_array - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Recovery Factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
    
    # Volatility (annualized)
    volatility = portfolio_returns.std() * np.sqrt(12)
    
    # Expected vs Actual returns
    expected_return = (returns.mean() @ weights) * 12
    actual_return = portfolio_returns.mean() * 12
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'recovery_factor': recovery_factor,
        'volatility': volatility,
        'expected_return': expected_return,
        'actual_return': actual_return,
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns
    }

def optimize_single_portfolio(args):
    """Optimize a single portfolio combination - for parallel processing"""
    combo, returns_df, min_weight, max_weight = args
    try:
        # Get subset of returns for this combination
        combo_returns = returns_df[list(combo)]
        
        # Calculate statistics
        combo_mean_returns = combo_returns.mean().values
        combo_cov = combo_returns.cov().values
        
        # Optimize weights
        weights = optimize_portfolio_weights_vectorized(
            combo_mean_returns, combo_cov, len(combo), min_weight, max_weight
        )
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics_fast(combo_returns, weights)
        
        # Store results
        result = {
            'stocks': combo,
            'weights': dict(zip(combo, weights)),
            **metrics
        }
        return result
    except Exception as e:
        print(f"Error optimizing {combo}: {str(e)}")
        return None

def find_best_portfolios_parallel(returns, n_stocks=3, top_n=5, min_weight=0.1, max_weight=1.0):
    """Find the best portfolio combinations using parallel processing"""
    stock_list = list(returns.columns)
    combinations_list = list(combinations(stock_list, n_stocks))
    n_combinations = len(combinations_list)
    
    st.info(f"🔍 Analyzing {n_combinations} possible combinations of {n_stocks} stocks using parallel processing...")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prepare arguments for parallel processing
    args_list = [(combo, returns, min_weight, max_weight) for combo in combinations_list]
    
    # Batch processing for better performance
    batch_size = 50
    completed = 0
    
    for i in range(0, len(args_list), batch_size):
        batch = args_list[i:i+batch_size]
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
            futures = [executor.submit(optimize_single_portfolio, args) for args in batch]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None and not np.isnan(result['sharpe_ratio']):
                        results.append(result)
                except Exception as e:
                    print(f"Error in future: {str(e)}")
                
                completed += 1
                
                # Update progress
                if completed % 10 == 0 or completed == n_combinations:
                    status_text.text(f"Optimized {completed}/{n_combinations} portfolios... Found {len(results)} valid portfolios")
                    progress_bar.progress(min(completed / n_combinations, 1.0))
    
    progress_bar.empty()
    status_text.empty()
    
    if len(results) == 0:
        st.error(f"❌ No valid portfolios found. This might be due to constraints being too strict or insufficient data.")
        return []
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    return results[:top_n]

# Main App
st.title("💼 Portfolio Optimizer Pro")
st.markdown("### Find the Best Performing Portfolios Across Sectors")

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Sector selection
sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))

# Date range
st.sidebar.markdown("### 📅 Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

# Portfolio settings
st.sidebar.markdown("### 🎯 Portfolio Settings")
n_stocks = st.sidebar.slider("Number of Stocks in Portfolio", 2, 5, 3)
min_weight = st.sidebar.slider("Minimum Weight per Stock (%)", 5, 20, 10) / 100
max_weight = st.sidebar.slider("Maximum Weight per Stock (%)", 30, 100, 100) / 100
top_n = st.sidebar.slider("Top N Portfolios to Display", 3, 10, 5)

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    n_simulations = st.number_input("Number of Random Portfolios", 1000, 10000, 3000, step=500,
                                    help="More simulations = better optimization but slower")
    use_cache = st.checkbox("Use Cached Data", value=True, 
                           help="Cache stock data for 1 hour to speed up repeated analyses")

# Risk-free rate
rf_rate = st.sidebar.number_input("Risk-Free Rate (% p.a.)", value=6.8, step=0.1) / 100

# Performance stats
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Performance Stats")
if 'last_execution_time' in st.session_state:
    st.sidebar.metric("Last Optimization Time", f"{st.session_state.last_execution_time:.2f}s")
if 'combinations_analyzed' in st.session_state:
    st.sidebar.metric("Combinations Analyzed", f"{st.session_state.combinations_analyzed:,}")

# Analyze button
analyze_btn = st.sidebar.button("🚀 Optimize Portfolios", type="primary")

if analyze_btn:
    import time
    start_time = time.time()
    
    with st.spinner("Fetching stock data in parallel..."):
        # Get stock list for selected sector
        stock_list = SECTORS[sector]
        
        # Fetch data in parallel
        if use_cache:
            prices, failed = get_stock_data_parallel(stock_list, start_date, end_date)
        else:
            st.cache_data.clear()
            prices, failed = get_stock_data_parallel(stock_list, start_date, end_date)
        
        if failed:
            st.warning(f"⚠️ Failed to fetch data for {len(failed)} stocks: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}")
        
        if len(prices.columns) < n_stocks:
            st.error(f"❌ Not enough stocks with valid data. Need at least {n_stocks}, got {len(prices.columns)}")
        else:
            st.success(f"✅ Successfully loaded data for {len(prices.columns)} stocks in {time.time() - start_time:.2f}s")
            
            # Calculate returns
            returns = calculate_monthly_returns(prices)
            
            # Find best portfolios
            st.markdown("---")
            st.markdown("## 🏆 Optimization Results")
            
            opt_start_time = time.time()
            best_portfolios = find_best_portfolios_parallel(
                returns, n_stocks, top_n, min_weight, max_weight
            )
            opt_time = time.time() - opt_start_time
            
            # Store stats
            from math import comb
            st.session_state.last_execution_time = time.time() - start_time
            st.session_state.combinations_analyzed = comb(len(returns.columns), n_stocks)
            
            st.success(f"✅ Optimization completed in {opt_time:.2f}s")
            
            if not best_portfolios:
                st.error("❌ No valid portfolios found. Try adjusting your parameters.")
            else:
                # Display best portfolio
                st.markdown("### 🥇 Best Portfolio")
                best = best_portfolios[0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sharpe Ratio", f"{best['sharpe_ratio']:.3f}")
                with col2:
                    st.metric("CAGR", f"{best['cagr']*100:.2f}%")
                with col3:
                    st.metric("Max Drawdown", f"{best['max_drawdown']*100:.2f}%")
                with col4:
                    st.metric("Volatility", f"{best['volatility']*100:.2f}%")
                
                # Weights
                st.markdown("#### 📊 Optimal Weights")
                weights_df = pd.DataFrame({
                    'Stock': list(best['weights'].keys()),
                    'Weight (%)': [w * 100 for w in best['weights'].values()]
                }).sort_values('Weight (%)', ascending=False)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(weights_df, hide_index=True, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=weights_df['Stock'],
                        values=weights_df['Weight (%)'],
                        hole=0.3
                    )])
                    fig_pie.update_layout(title="Portfolio Allocation", height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Performance chart
                st.markdown("#### 📈 Portfolio Performance")
                cumulative_pct = (best['cumulative_returns'] - 1) * 100
                
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(
                    x=cumulative_pct.index,
                    y=cumulative_pct.values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#00ff00', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ))
                fig_perf.update_layout(
                    title="Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    height=400,
                    template='plotly_dark',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Additional metrics
                st.markdown("#### 📊 Detailed Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recovery Factor", f"{best['recovery_factor']:.2f}")
                    st.metric("Expected Return (Annual)", f"{best['expected_return']*100:.2f}%")
                with col2:
                    st.metric("Actual Return (Annual)", f"{best['actual_return']*100:.2f}%")
                    st.metric("Total Stocks", len(best['stocks']))
                with col3:
                    total_return = (best['cumulative_returns'].iloc[-1] - 1) * 100
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                # Drawdown chart
                st.markdown("#### 📉 Drawdown Analysis")
                cumulative = (1 + best['portfolio_returns']).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = ((cumulative - running_max) / running_max) * 100
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ff0000', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)'
                ))
                fig_dd.update_layout(
                    title="Portfolio Drawdown Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=300,
                    template='plotly_dark',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Top portfolios comparison
                st.markdown("---")
                st.markdown(f"### 📋 Top {len(best_portfolios)} Portfolios Comparison")
                
                comparison_data = []
                for i, portfolio in enumerate(best_portfolios):
                    comparison_data.append({
                        'Rank': i + 1,
                        'Stocks': ', '.join(portfolio['stocks']),
                        'Sharpe Ratio': f"{portfolio['sharpe_ratio']:.3f}",
                        'CAGR (%)': f"{portfolio['cagr']*100:.2f}",
                        'Max DD (%)': f"{portfolio['max_drawdown']*100:.2f}",
                        'Volatility (%)': f"{portfolio['volatility']*100:.2f}",
                        'Recovery': f"{portfolio['recovery_factor']:.2f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                # Individual portfolio expanders
                st.markdown("### 🔍 Detailed Portfolio Analysis")
                for i, portfolio in enumerate(best_portfolios):
                    with st.expander(f"Portfolio #{i+1}: {', '.join(portfolio['stocks'])} - Sharpe: {portfolio['sharpe_ratio']:.3f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Allocation:**")
                            for stock, weight in portfolio['weights'].items():
                                st.write(f"• {stock}: {weight*100:.2f}%")
                        
                        with col2:
                            st.markdown("**Performance Metrics:**")
                            st.write(f"• Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")
                            st.write(f"• CAGR: {portfolio['cagr']*100:.2f}%")
                            st.write(f"• Max Drawdown: {portfolio['max_drawdown']*100:.2f}%")
                            st.write(f"• Volatility: {portfolio['volatility']*100:.2f}%")
                            st.write(f"• Recovery Factor: {portfolio['recovery_factor']:.2f}")
                        
                        # Mini performance chart
                        cumulative_pct = (portfolio['cumulative_returns'] - 1) * 100
                        fig_mini = go.Figure()
                        fig_mini.add_trace(go.Scatter(
                            x=cumulative_pct.index,
                            y=cumulative_pct.values,
                            mode='lines',
                            line=dict(color='#1f77b4', width=1.5)
                        ))
                        fig_mini.update_layout(
                            title=f"Portfolio #{i+1} Cumulative Returns",
                            xaxis_title="Date",
                            yaxis_title="Return (%)",
                            height=250,
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig_mini, use_container_width=True)
                
                # Download results
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                
                # Prepare export data
                export_data = []
                for i, portfolio in enumerate(best_portfolios):
                    row = {
                        'Rank': i + 1,
                        'Stocks': ', '.join(portfolio['stocks']),
                        'Sharpe Ratio': portfolio['sharpe_ratio'],
                        'CAGR': portfolio['cagr'],
                        'Max Drawdown': portfolio['max_drawdown'],
                        'Volatility': portfolio['volatility'],
                        'Recovery Factor': portfolio['recovery_factor'],
                        'Expected Return': portfolio['expected_return'],
                        'Actual Return': portfolio['actual_return']
                    }
                    # Add weights
                    for stock, weight in portfolio['weights'].items():
                        row[f'Weight_{stock}'] = weight
                    export_data.append(row)
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio Analysis (CSV)",
                    data=csv,
                    file_name=f"{sector}_portfolios_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

else:
    # Instructions
    st.markdown("---")
    st.markdown("""
    ## 📖 How to Use
    
    1. **Select a Sector** from the sidebar (Metal, Pharma, Banks, etc.)
    2. **Set Date Range** for historical analysis
    3. **Configure Portfolio Settings:**
       - Number of stocks per portfolio (2-5)
       - Minimum and maximum weight constraints
       - Number of top portfolios to analyze
    4. **Click "Optimize Portfolios"** to start the analysis
    
    ## 🎯 What This Tool Does
    
    This app uses **Modern Portfolio Theory** to find the optimal combination of stocks that maximizes the Sharpe Ratio within your selected sector.
    
    **Key Metrics:**
    - **Sharpe Ratio**: Risk-adjusted returns (higher is better)
    - **CAGR**: Compound Annual Growth Rate
    - **Max Drawdown**: Largest peak-to-trough decline
    - **Recovery Factor**: Total return / Max drawdown
    - **Volatility**: Standard deviation of returns
    
    ## ⚡ Performance Optimizations
    
    - **Parallel Data Fetching**: Downloads multiple stocks simultaneously
    - **Vectorized Calculations**: Uses NumPy for fast matrix operations
    - **Batch Processing**: Processes portfolios in parallel batches
    - **Smart Caching**: Caches stock data for repeated analyses
    - **Optimized Algorithms**: Reduced from O(n³) to O(n²) complexity
    
    ## 💡 Tips
    
    - Start with 3 stocks for faster analysis
    - Use at least 2-3 years of data for reliable results
    - Enable caching for faster repeated runs
    - Increase simulations for better optimization (slower)
    - Compare multiple portfolios to understand trade-offs
    """)
    
    # Display sector info
    st.markdown("---")
    st.markdown("### 📊 Available Sectors")
    
    sector_info = []
    for sector_name, stocks in SECTORS.items():
        sector_info.append({
            'Sector': sector_name,
            'Number of Stocks': len(stocks),
            'Examples': ', '.join(stocks[:3]) + '...'
        })
    
    st.dataframe(pd.DataFrame(sector_info), hide_index=True, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Optimization Method")
st.sidebar.info("""
**Mean-Variance Optimization**

Maximizes Sharpe Ratio:
- Return / Risk ratio
- Considers correlations
- Parallel processing
- Vectorized calculations

**Constraints:**
- Weights sum to 100%
- Min/Max weight limits
- Long-only (no shorting)

**Performance Features:**
- Multi-threaded downloads
- Batch processing
- Smart caching
- NumPy vectorization
""")

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Disclaimer**: Past performance does not guarantee future results. This tool is for educational purposes only.")
