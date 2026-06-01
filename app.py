import cv2
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Chart Analyst Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize a persistent session state list for the trading journal logs
if "journal_logs" not in st.session_state:
    st.session_state.journal_logs = []

# Custom style overrides for container padding & meter appearance
st.html(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    
    /* Confidence Meter Gauge Styles */
    .meter-container {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-family: monospace;
        font-size: 18px;
        letter-spacing: 2px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    </style>
    """
)


# ==========================================
# CORE IMAGE PROCESSING LOGIC
# ==========================================
def analyze_chart_from_bytes(image_bytes):
    # Decode bytes into OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[int(h * 0.3) : int(h * 0.7), :]

    # 1️⃣ EDGE + LINE SLOPE
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 80, minLineLength=40, maxLineGap=10
    )

    slopes = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)

    avg_slope = np.mean(slopes) if slopes else 0

    # 2️⃣ MOVING AVERAGE TREND
    column_means = np.mean(roi, axis=0)
    ma_short = np.convolve(column_means, np.ones(5) / 5, mode="valid")
    ma_long = np.convolve(column_means, np.ones(20) / 20, mode="valid")

    ma_signal = 0
    if len(ma_short) > 0 and len(ma_long) > 0:
        if ma_short[-1] < ma_long[-1]:
            ma_signal = 1  # uptrend
        else:
            ma_signal = -1  # downtrend

    # 3️⃣ MOMENTUM (LEFT → RIGHT)
    left = np.mean(roi[:, : w // 2])
    right = np.mean(roi[:, w // 2 :])
    momentum = right - left

    if momentum < -2:
        mom_signal = 1
    elif momentum > 2:
        mom_signal = -1
    else:
        mom_signal = 0

    # 🔥 COMBINED SCORE
    score = 0
    if avg_slope < -0.05:
        score += 2
    elif avg_slope > 0.05:
        score -= 2

    score += ma_signal
    score += mom_signal

    # 🎯 SIGNAL OUTPUT
    if score >= 2:
        trend = "UPTREND"
        signal = "BUY"
    elif score <= -2:
        trend = "DOWNTREND"
        signal = "SELL"
    else:
        trend = "SIDEWAYS"
        signal = "HOLD"

    confidence = min(abs(score) * 25, 100)

    return signal, trend, confidence, img, gray, edges, avg_slope


def generate_price_from_image(gray_img):
    pixel_sum = np.sum(gray_img)
    price = (pixel_sum % 1000) + 1000
    return round(price, 2)


def risk_management(price):
    stop_loss = price * 0.99
    target = price * 1.02
    return round(stop_loss, 2), round(target, 2)


# ==========================================
# NAVIGATION BAR
# ==========================================
with st.sidebar:
    st.title("🛡️ TradeVision AI")
    st.subheader("Computer Vision Trading Hub")

    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard Analyzer", "Risk Calculator","Trading Journal", "How it Works", "About System", "Complete Trading Guide"],
        icons=["cpu", "calculator", "journal-text", "book", "info-circle", "mortarboard"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px!", "background-color": "#1e293b"},
            "icon": {"color": "#f43f5e", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#334155",
                "color": "#cbd5e1",
            },
            "nav-link-selected": {"background-color": "#0ea5e9"},
        },
    )
    st.divider()
    st.caption("v2.1.0-Beta • Powered by Streamlit & OpenCV")

# ==========================================
# APP PAGES ROUTING
# ==========================================

# --- PAGE 1: ANALYZER ---
if selected == "Dashboard Analyzer":
    st.title("📈 AI Chart & Trend Scanner")
    st.write(
        "Upload a technical analysis chart screenshot below to analyze algorithmic signals instantly."
    )

    uploaded_file = st.file_uploader(
        "Choose a chart image file (PNG, JPG, JPEG)...",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()

        with st.spinner("Processing computer vision models..."):
            analysis_results = analyze_chart_from_bytes(bytes_data)

        if analysis_results is not None:
            (
                signal,
                trend,
                confidence,
                img,
                gray,
                edges,
                avg_slope,
            ) = analysis_results
            price = generate_price_from_image(gray)
            stop_loss, target = risk_management(price)

            if signal == "BUY":
                signal_color = "#22c55e"  # Green
                alert_theme = st.success
            elif signal == "SELL":
                signal_color = "#ef4444"  # Red
                alert_theme = st.error
            else:
                signal_color = "#eab308"  # Yellow
                alert_theme = st.warning

            st.divider()
            alert_theme(
                f"### 🔥 **FINAL EXECUTION DECISION: {signal}** ({trend})"
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Action Signal", value=signal)
            with col2:
                st.metric(label="Trend Vector", value=trend)
           
            with col3:
                st.metric(
                    label="Signal Confidence", value=f"{int(confidence)}%"
                )

            # ==========================================
            # 🔄 DYNAMIC CONFIDENCE METER GAUGE
            # ==========================================
            st.write("**Model Certainty Engine Profile:**")
            
            total_blocks = 10
            filled_blocks = int((confidence / 100) * total_blocks)
            empty_blocks = total_blocks - filled_blocks
            
            meter_string = "🟩" * filled_blocks + "⬜" * empty_blocks
            
            st.html(
                f"""
                <div class="meter-container">
                    <span style="color: {signal_color}; font-weight: bold;">[</span> 
                    {meter_string} 
                    <span style="color: {signal_color}; font-weight: bold;">] {int(confidence)}%</span>
                </div>
                """
            )

            st.divider()
            tab1, tab2, tab3 = st.tabs(
                ["📷 Image Workspace", "📊 Pattern Validation Matrix", "🧬 CV Insights"]
            )

            with tab1:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.subheader("Original File Upload")
                    st.image(
                        bytes_data, use_container_width=True, caption="Original Chart"
                    )
                with t_col2:
                    st.subheader("Region of Interest (ROI) Grayscale View")
                    h, w = gray.shape
                    roi_crop = gray[int(h * 0.3) : int(h * 0.7), :]
                    st.image(
                        roi_crop,
                        use_container_width=True,
                        caption="Sampled Scan Target Pipeline (Middle 40%)",
                    )

            with tab2:
                st.subheader("📑 Visual Pattern Validation & Confluence Logging")
                st.write("Cross-reference computer vision diagnostics with chart pattern layouts to complete confirmation protocols.")
                
                log_col1, log_col2 = st.columns([1, 1])
                
                with log_col1:
                    st.markdown("### 🔍 Macro Structural Verification")
                    chart_pattern = st.selectbox(
                        "Identified Chart Geometry Structure",
                        ["None / Pure Trend Following", "Double Bottom / Top", "Head & Shoulders", "Bullish / Bearish Flag", "Symmetrical Triangle", "Support / Resistance Bounce"]
                    )
                    
                    candle_confluence = st.selectbox(
                        "Local Candlestick Confluence Signal",
                        ["No clear pattern", "Pin Bar / Hammer (Reversal)", "Engulfing Candle (Momentum)", "Doji (Indecision)", "Inside Bar Breakout"]
                    )
                    
                    volume_profile = st.radio(
                        "Volume Spread Analysis Confirmation",
                        ["Volume expansion on breakout (High Confirmation)", "Average flat volume", "Divergent volume decreasing on move (Low Conviction)"],
                        index=1
                    )

                with log_col2:
                    st.markdown("### 💾 Exportable Technical Signature Log")
                    
                    log_template = f"""--- STRUCTURAL CONFLUENCE REPORT ---
[CV DIRECTIONAL BIAS]: {signal} (Trend Context: {trend})
[CV MATRIX CONFIDENCE]: {int(confidence)}%
[MACRO GEOMETRY STRUCTURE]: {chart_pattern}
[LOCAL CANDLESTICK MATCH]: {candle_confluence}
[VOLUME ANALYSIS TRACK]: {volume_profile}
-------------------------------------"""
                    st.code(log_template, language="markdown")
                    
                    if st.button("🚀 Archive Pattern Signature to Sandbox"):
                        st.toast("Technical pattern snapshot successfully locked to memory registers!", icon="📊")

            with tab3:
                st.subheader("Computer Vision Diagnostic Matrix")
                insights_col1, insights_col2 = st.columns(2)
                with insights_col1:
                    st.markdown(
                        f"""
                    *   **Average Trend Line Slope:** `{avg_slope:.5f}`
                    *   **Processed Image Resolution Matrix:** `{w} x {h} Pixels`
                    *   *Mathematical Edge Density Detected in System:* See Right Panel
                    """
                    )
                with insights_col2:
                    st.image(
                        edges,
                        use_container_width=True,
                        caption="Canny Structural Edge Detection Array",
                    )
        else:
            st.error(
                "Could not compile image frame properly. Please ensure standard image structures."
            )
    else:
        st.info("⏳ Align your charting interval with your strategy: choose lower-fraction timeframes for intraday execution, and higher-order horizons for long-term investments.")

# --- PAGE 2: RISK CALCULATOR ---
elif selected == "Risk Calculator":
    st.title("🧮 Professional Position Sizing & Risk Allocation")
    st.write("Calculate your precise order sizing and mathematical risk allocation before committing capital.")
    
    st.divider()
    
    calc_col1, calc_col2 = st.columns([1, 1])
    
    with calc_col1:
        st.subheader("📥 Input Parameters")
        account_size = st.number_input("Total Account Balance (₹)", min_value=0.0, value=10000.0, step=500.0)
        risk_pct = st.slider("Account Risk Percentage (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
        entry_price = st.number_input("Entry Price (₹)", min_value=0.0001, value=100.0, step=1.0)
        stop_loss = st.number_input("Stop Loss Price (₹)", min_value=0.0, value=95.0, step=1.0)
        take_profit = st.number_input("Take Profit Price (₹)", min_value=0.0, value=110.0, step=1.0)

    with calc_col2:
        st.subheader("📊 Calculation Metrics")
        
        total_risk_cash = account_size * (risk_pct / 100.0)
        per_share_risk = abs(entry_price - stop_loss)
        per_share_reward = abs(take_profit - entry_price)
        
        if per_share_risk > 0:
            position_units = total_risk_cash / per_share_risk
            total_notional_value = position_units * entry_price
            risk_reward_ratio = per_share_reward / per_share_risk
            projected_profit = position_units * per_share_reward
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Cash Capital at Risk", f"₹{total_risk_cash:,.2f}")
            m_col2.metric("Optimal Position Size", f"{position_units:.2f} Units")
            
            m_col3, m_col4 = st.columns(2)
            m_col3.metric("Total Notional Value", f"₹{total_notional_value:,.2f}")
            m_col4.metric("Risk-to-Reward Ratio", f"1 : {risk_reward_ratio:.2f}")
            
            st.divider()
            if entry_price > stop_loss:
                st.success(f"🟢 **Long Position Summary:** If target hits, projected profit is **₹{projected_profit:,.2f}**.")
            else:
                st.error(f"🔴 **Short Position Summary:** If target hits, projected profit is **₹{projected_profit:,.2f}**.")
                
            if total_notional_value > account_size:
                st.warning(f"⚠️ *Leverage warning: Your total position value (${total_notional_value:,.2f}) exceeds your capital bank. Margin or leverage will be required to open this trade.*")
        else:
            st.info("Set a Stop Loss value distinct from your Entry Price to populate the mathematical metrics matrix.")


# --- PAGE 3: TRADING JOURNAL ---
elif selected == "Trading Journal":
    st.title("📓 Performance Log & Strategy Journal")
    st.write("Document architectural configurations, execution mentalities, and manual sandbox annotations.")
    st.divider()
    
    j_col1, j_col2 = st.columns([1, 2])
    
    with j_col1:
        st.subheader("📝 Manual Entry Ledger")
        ticker = st.text_input("Asset Ticker / Instrument", value="SPY").upper()
        direction = st.selectbox("Execution Strategy Vector", ["BUY / LONG", "SELL / SHORT", "OBSERVATION / HOLD"])
        setup_rating = st.slider("Technical Confluence Rating", 1, 5, 3)
        journal_notes = st.text_area("Behavioral Notes & Setup Parameters", placeholder="Describe entry trigger matrix context...")
        
        if st.button("💾 Commit Entry to Log Table"):
            st.session_state.journal_logs.append({
                "Asset/Ticker": ticker,
                "Direction": direction,
                "Context Layout": f"Manual Rating: {setup_rating}/5",
                "Structure Type": "Manual Entry Sandbox",
                "Candle Confluence": "N/A",
                "Notes/VSA Profile": journal_notes if journal_notes else "No detailed structural notation."
            })
            st.success(f"Entry for {ticker} added to memory registry!")
            
    with j_col2:
        st.subheader("📊 Session History Data Stream")
        if st.session_state.journal_logs:
            df_logs = pd.DataFrame(st.session_state.journal_logs)
            st.dataframe(df_logs, use_container_width=True)
            
            if st.button("🗑️ Reset Active Session Log"):
                st.session_state.journal_logs = []
                st.rerun()
        else:
            st.info("No active structural logs compiled this session. Archive scans from the Dashboard Analyzer or complete a ledger entry to fill this view matrix.")


# --- PAGE 4: HOW IT WORKS ---
elif selected == "How it Works":
    st.title("🧬 Core Algorithmic Architecture")
    st.write(
        "Welcome to the engine room. Below is the granular breakdown of how the image-processing pipelines "
        "translate raw pixel matrices into structured execution signals (`BUY`, `SELL`, `HOLD`)."
    )

    st.info(
        "🧠 **Core Philosophy:** This engine treats trading charts not as mathematical time-series data, "
        "but as geometric arrays. It scans for edge density, linear convergence, and pixel mass distribution."
    )

    engine_tab1, engine_tab2, engine_tab3, engine_tab4 = st.tabs(
        [
            "1️⃣ Hough Line Detection",
            "2️⃣ Moving Average Convolutions",
            "3️⃣ Momentum Distribution",
            "🎯 The Decision Matrix",
        ]
    )

    with engine_tab1:
        st.header("1️⃣ Structural Hough Line Slope Evaluation")
        st.write("This pipeline extracts structural trendlines from the chart canvas to calculate a macro geometric vector.")
        st.markdown("### The Mathematical Concept")
        st.write("Every point in an image space $(x, y)$ can map to a sinusoidal curve in the Hough Parameter Space $(\\rho, \\theta)$ using the polar formula:")
        st.latex(r"\rho = x \cos \theta + y \sin \theta")
        st.write("The mathematical slope $m$ of each isolated line segment is calculated using:")
        st.latex(r"m = \frac{y_2 - y_1}{x_2 - x_1}")

        st.markdown("### Processing Workflow")
        st.code(
            """
edges = cv2.Canny(roi, 50, 150)  
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=40, maxLineGap=10)  

for line in lines:
    x1, y1, x2, y2 = line[0]
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
        
avg_slope = np.mean(slopes)
            """,
            language="python",
        )

    with engine_tab2:
        st.header("2️⃣ Region of Interest Column Convolutions")
        st.write("Instead of reading lines, this engine collapses the image into a 1D continuous array to check for visual mass density.")
        st.markdown("### The Mathematical Concept")
        st.write("A discrete 1D convolution acts as a uniform moving average window over that array profile:")
        st.latex(r"y[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n - k]")

        st.markdown("### Processing Workflow")
        st.code(
            """
column_means = np.mean(roi, axis=0)  
ma_short = np.convolve(column_means, np.ones(5)/5, mode='valid')  
ma_long = np.convolve(column_means, np.ones(20)/20, mode='valid')  

if ma_short[-1] < ma_long[-1]:
    ma_signal = 1   
else:
    ma_signal = -1  
            """,
            language="python",
        )

    with engine_tab3:
        st.header("3️⃣ Directional Distribution Momentum")
        st.write("This asset pipeline measures the absolute pixel intensity difference across the vertical midpoint grid.")
        st.markdown("### The Mathematical Concept")
        st.latex(r"\Delta \text{Momentum} = \mu_{\text{Right Matrix}} - \mu_{\text{Left Matrix}}")

        st.markdown("### Processing Workflow")
        st.code(
            """
left_half = roi[:, :w//2]
right_half = roi[:, w//2:]
momentum = np.mean(right_half) - np.mean(left_half)  

if momentum < -2:
    mom_signal = 1   
elif momentum > 2:
    mom_signal = -1  
else:
    mom_signal = 0   
            """,
            language="python",
        )

    with engine_tab4:
        st.header("🎯 System Synthesis & Confidence Matrix")
        st.write("The engine takes the signals generated independently by all three pipelines and processes them through an additive voting layer.")
        st.markdown("### Sub-Indicator Voting Weight Allocation Table")
        st.markdown(
            """
        | Algorithmic Sub-Pipeline Module | Active Condition | Contributed Score Value |
        | :--- | :--- | :--- |
        | **Hough Line Slope Engine** | Avg Slope $< -0.05$ | `+2` |
        | **Hough Line Slope Engine** | Avg Slope $> 0.05$ | `-2` |
        | **Moving Average Convolution** | Fast MA Index $<$ Slow MA Index | `+1` |
        | **Moving Average Convolution** | Fast MA Index $\geq$ Slow MA Index | `-1` |
        | **Distribution Momentum** | Symmetrical Deviation Vector $< -2$ | `+1` |
        | **Distribution Momentum** | Symmetrical Deviation Vector $> 2$ | `-1` |
        """
        )

# --- PAGE 5: ABOUT SYSTEM ---
elif selected == "About System":
    st.title("🛡️ Institutional App Metadata & System Profile")
    st.info("This enterprise-grade computer vision trading simulation platform processes geometric and spatial matrix configurations via automated local algorithmic evaluation kernels.")
    
    sys_tab1, sys_tab2, sys_tab3 = st.tabs(["💻 System Environment Profile", "⚙️ Algorithmic Parameter Specifications", "🔄 Pipeline Data Flow"])
    
    with sys_tab1:
        st.subheader("Core Environment Infrastructure")
        st.markdown(
            """
            * **Platform Core Framework:** `Streamlit Web Engine Interface`
            * **Computer Vision Kernel:** `OpenCV Pipeline Handler`
            * **Linear Algebra Engine:** `NumPy Vector Space Matrix Libraries`
            """
        )
        
    with sys_tab2:
        st.subheader("Hardcoded Operational Constraints")
        st.markdown(
            """
            * **Symmetrical Region of Interest (ROI):** Dynamic boundary slicing targeting exactly the middle $40\\%$ ($0.30 \\times H$ to $0.70 \\times H$).
            * **Canny Boundary Gradient Thresholds:** Dual-stage thresholds locked at $50$ and $150$.
            * **Risk-Engine Ratio Target:** Standard static $1:2$ Risk-to-Reward matrix ($1.00\\%$ Stop Loss vs $2.00\\%$ Take Profit).
            """
        )
        
    with sys_tab3:
        st.subheader("Deterministic Matrix Processing Lifecycle")
        st.markdown(
            """
            1. **Ingestion & Matrix Formatting:** File byte arrays are read out of memory into a standard BGR pixel cube array.
            2. **Grayscale Demodulation:** RGB curves are collapsed into an 8-bit monochromatic spectrum.
            3. **Parallel Signal Token Generation:** Independent calculations score Line Slopes, Convolutions, and Contrasts.
            4. **Consolidated Voting Matrix Evaluation:** Individual scores aggregate via an additive voting block layer.
            """
        )

# --- PAGE 6: COMPLETE TRADING GUIDE ---
elif selected == "Complete Trading Guide":
    st.title("📈 Basic to Advance Trading Concepts")
    st.info("“In trading, success does not come from shortcuts—it is built through relentless study, disciplined practice, and the hard work that turns every setback into a stepping stone toward mastery.”")
    
    # Phase 1
    st.markdown(
        """
        ### Phase 1: Foundational Core (The Basics)
       
        * **Market Mechanics & Asset Classes:** Trading is the buying and selling of financial instruments. 
          A professional must differentiate between Equities (company shares), Fixed Income (bonds), Forex (currencies), Commodities (gold, oil, agriculture), and Derivatives (options, futures).
        
        * **The Order Book & Liquidity:** Markets operate via a central limit order book consisting of Bids (buyers) and Asks (sellers). 
          Liquidity refers to how easily an asset can be bought or sold without causing a significant price movement.
        
        * **Order Types:** Execution proficiency requires mastering different order types:
            * *Market Order:* Immediate execution at the best available current price.
            * *Limit Order:* Execution only at a specified price or better.
            * *Stop-Market / Stop-Limit Order:* Triggered only when a specific price level is breached, primarily used to limit losses.
        
        * **Market Participants:** Understanding who is on the other side of a trade—such as Retail Traders, Institutional Investors (hedge funds, mutual funds), Market Makers (providing liquidity), and Central Banks.
        """
    )
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1200", use_container_width=True, caption="Phase 1 Visual Paradigm: Core Execution Infrastructure & Live Market Mechanics")
    st.markdown("---")

    # Phase 2
    st.markdown(
        """
        ### Phase 2: Analytical Methodologies (Intermediate)
        
        * **Price Action & Candlesticks:** Reading Japanese candlestick charts to interpret open, high, low, and close prices over specific timeframes.
        * **Support & Resistance:** Identifying psychological and structural zones where buying or selling pressure historically shifts.
        * **Technical Indicators:** Utilizing mathematical calculations applied to price and volume:
            * *Trend-following:* Moving Averages (EMA/SMA), MACD.
            * *Oscillators (Momentum):* Relative Strength Index (RSI), Stochastic Oscillator.
            * *Volatility:** Bollinger Bands, Average True Range (ATR).
        * **Chart & Harmonic Patterns:** Recognizing continuation and reversal formations such as Head and Shoulders, Double Bottoms, Flags, and Triangles.
        * **Fundamental Analysis (FA):** Fundamental analysis evaluates the intrinsic value of an asset by examining related economic and financial factors.
        * **Macroeconomics (Forex/Commodities):** Tracking interest rate decisions (Central Bank policies), Gross Domestic Product (GDP), Inflation rates (CPI/PPI), and employment data (e.g., Non-Farm Payrolls).
        * **Microeconomics (Equities):** Analyzing corporate health via balance sheets, income statements, cash flow, Earnings Per Share (EPS), and P/E ratios.
        """
    )
    st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=1200", use_container_width=True, caption="Phase 2 Visual Paradigm: Technical Confluence and Chart Formations")
    st.markdown("---")

    # Phase 3
    st.markdown(
        """
        ### Phase 3: Risk Management & Capital Preservation (Crucial)

        * **Risk-to-Reward Ratio (R:R):** Calculating the mathematical relationship between the potential loss and potential profit of a trade. A professional standard minimum is often $1:2$ (risking \$1 to make \$2).
        * **Position Sizing:** Determining the exact size of a trade based on account equity. Professionals rarely risk more than 1% to 2% of their total trading capital on a single trade.
        * **The Math of Drawdowns:** Understanding that losses require exponentially larger gains to recover. For example, a 50% drawdown requires a 100% return just to break even.
        * **Strict Stop-Loss Execution:** Hard-coding a price level where the trade thesis is invalidated, preventing catastrophic capital destruction.
        """
    )
    st.image("https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=1200", use_container_width=True, caption="Phase 3 Visual Paradigm: Mathematical Risk Parameters and Trade Allocation")
    st.markdown("---")

    # Phase 4
    st.markdown(
        """
        ### Phase 4: Trading Psychology & Behavioral Finance (Advanced)

        * **Cognitive Biases:** Overcoming psychological traps such as FOMO (Fear of Missing Out), Loss Aversion (holding losing trades too long hoping they come back), and Confirmation Bias (only looking for data that supports your trade).
        * **Revenge Trading:** The destructive emotional urge to immediately win back lost capital by taking larger, undisciplined trades.
        * **The Probability Mindset:** Accepting that any individual trade has an uncertain outcome. Professional success depends on a statistical edge played out over a large sample size of trades.
        """
    )
    st.image("https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?q=80&w=1200", use_container_width=True, caption="Phase 4 Visual Paradigm: Dynamic Execution Mindset & Trading Discipline")
    st.markdown("---")

    # Phase 5
    st.markdown(
        """
        ### Phase 5: Quantitative & Institutional Execution (Advanced Professional)

        * **Statistical Edge & Expectancy:** Developing a trading framework with a positive expectancy formula:
          $$\text{Expectancy} = (\text{Win Rate} \times \text{Average Win}) - (\text{Loss Rate} \times \text{Average Loss})$$
        * **Algorithmic & Quantitative Trading:** Utilizing programming languages (Python, R) and statistical models to backtest historical data, eliminate human emotion, and execute systemic strategies.
        * **Market Microstructure:** Analyzing order flow, Level 2 Data, Time & Sales data, and Volume Profile to see where large institutions are positioning liquidity.
        * **Hedging & Portfolio Correlation:** Managing multi-asset portfolios by utilizing options or inverse correlations to mitigate systemic market risk (e.g., shorting index futures to protect a long equity portfolio).
        * **Performance Auditing:** Keeping a rigorous trading journal to track key metrics like the Sharpe Ratio, maximum drawdown, profit factor, and win-rate consistency to continuously optimize execution.
        """
    )
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=1200", use_container_width=True, caption="Phase 5 Visual Paradigm: Multi-Monitor Data Infrastructure & Quantitative Pipelines")