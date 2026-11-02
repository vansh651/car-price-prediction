import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Car Price Oracle",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.sav"
DATA_PATH = BASE_DIR / "cars.csv"

# These encodings and the age reference year match the saved model.
FUEL_MAP = {"Petrol": 0, "Diesel": 1, "CNG": 2}
SELLER_MAP = {"Dealer": 0, "Individual": 1}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}
REFERENCE_YEAR = 2025
MODEL_FEATURES = [
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
    "Age",
]
FEATURE_LABELS = {
    "Present_Price": "Current Market Price",
    "Kms_Driven": "Kilometres Driven",
    "Fuel_Type": "Fuel Type",
    "Seller_Type": "Seller Type",
    "Transmission": "Transmission",
    "Owner": "Previous Owners",
    "Age": "Car Age",
}


# ============================================================
# LOAD MODEL + DATASET
# ============================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.sav was not found beside streamlit_app.py")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("cars.csv was not found beside streamlit_app.py")
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_dataset()
except Exception as exc:
    st.error("The model or dataset could not be loaded.")
    st.code(str(exc))
    st.stop()


# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #f1f5f9;
        color: #172b4d;
    }
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: rgba(0,0,0,0);}

    .hero {
        background: linear-gradient(120deg, #397df6, #245fd0);
        padding: 2.6rem 1.2rem;
        text-align: center;
        border-radius: 16px;
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        margin: 0;
    }
    .hero p {
        color: #ffffff !important;
        font-size: 1.1rem;
        margin: .65rem 0 0 0;
    }
    .section-heading {
        color: #397df6;
        font-size: 1.8rem;
        font-weight: 800;
        border-bottom: 2px solid #dbe5f1;
        padding-bottom: .7rem;
        margin: 1rem 0 1.2rem 0;
    }
    label, .stMarkdown, .stText, p, li {
        color: #172b4d;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #172b4d !important;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: #397df6;
        color: #ffffff;
        border: 0;
        border-radius: 9px;
        font-weight: 700;
        padding: .65rem 1.2rem;
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #245fd0;
        color: #ffffff;
        border: 0;
    }
    .prediction-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem 1rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(15, 23, 42, .06);
        margin: 1rem 0 1.5rem 0;
    }
    .prediction-label {
        color: #64748b !important;
        font-size: 1.05rem;
    }
    .prediction-price {
        color: #2563eb !important;
        font-size: clamp(2.2rem, 5vw, 3rem);
        font-weight: 850;
        margin: .5rem 0;
    }
    .prediction-description {
        color: #64748b !important;
        font-size: .98rem;
    }
    .detail-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        padding: 1.15rem .7rem;
        text-align: center;
        min-height: 112px;
        margin-bottom: .8rem;
        box-shadow: 0 3px 10px rgba(15, 23, 42, .045);
    }
    .detail-label {
        color: #8493a8 !important;
        font-size: .9rem;
        margin-bottom: .55rem;
    }
    .detail-value {
        color: #172b4d !important;
        font-size: 1.08rem;
        font-weight: 750;
    }
    .footer-box {
        background: #101a30;
        color: #ffffff !important;
        text-align: center;
        padding: 2rem 1rem;
        border-radius: 14px;
        margin-top: 2.5rem;
    }
    .footer-box p, .footer-box div {
        color: #ffffff !important;
        margin: .55rem 0;
    }
    .footer-sub {
        color: #cbd5e1 !important;
        font-size: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🚘 Car Price Oracle</h1>
        <p>Predict your car's resale value with machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PREDICTION FORM
# ============================================================
st.markdown('<div class="section-heading">Enter Your Car Details</div>', unsafe_allow_html=True)

with st.form("car_prediction_form"):
    left, right = st.columns(2)

    with left:
        present_price = st.number_input(
            "Current Market Price (in lakhs)",
            min_value=0.1,
            max_value=1000.0,
            value=5.0,
            step=0.1,
        )
        kms_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            max_value=1_000_000,
            value=30_000,
            step=500,
        )
        fuel_type = st.selectbox("Fuel Type", list(FUEL_MAP.keys()))
        seller_type = st.selectbox("Seller Type", list(SELLER_MAP.keys()))

    with right:
        transmission = st.selectbox("Transmission", list(TRANSMISSION_MAP.keys()))
        owner = st.selectbox(
            "Number of Previous Owners",
            [0, 1, 2, 3],
            format_func=lambda value: {
                0: "0 (First Owner)",
                1: "1 (Second Owner)",
                2: "2 (Third Owner)",
                3: "3 (Fourth Owner or more)",
            }[value],
        )
        year = st.number_input(
            "Year of Purchase",
            min_value=1990,
            max_value=REFERENCE_YEAR,
            value=2020,
            step=1,
        )

    submitted = st.form_submit_button("🚗 Predict Price", use_container_width=True)


# ============================================================
# PREDICT - IMPORTANT: USE THE EXACT TRAINING COLUMN NAMES
# ============================================================
if submitted:
    age = REFERENCE_YEAR - int(year)
    input_df = pd.DataFrame(
        [[
            float(present_price),
            int(kms_driven),
            FUEL_MAP[fuel_type],
            SELLER_MAP[seller_type],
            TRANSMISSION_MAP[transmission],
            int(owner),
            int(age),
        ]],
        columns=MODEL_FEATURES,
    )

    try:
        prediction = float(model.predict(input_df)[0])
        st.markdown('<div class="section-heading">Your Prediction</div>', unsafe_allow_html=True)
        st.success("Prediction generated successfully!")
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimated Resale Price</div>
                <div class="prediction-price">₹{prediction:.2f} lakhs</div>
                <div class="prediction-description">
                    Based on the provided car details and our trained machine-learning model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        details = [
            ("Current Price", f"₹{present_price:.2f} lakhs"),
            ("Kilometres", f"{int(kms_driven):,} km"),
            ("Fuel Type", fuel_type),
            ("Seller Type", seller_type),
            ("Transmission", transmission),
            ("Previous Owners", str(int(owner))),
            ("Year of Purchase", str(int(year))),
            ("Car Age Used by Model", f"{age} years"),
        ]
        for offset in range(0, len(details), 4):
            cols = st.columns(4)
            for col, (label, value) in zip(cols, details[offset : offset + 4]):
                with col:
                    st.markdown(
                        f"""
                        <div class="detail-card">
                            <div class="detail-label">{label}</div>
                            <div class="detail-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.caption("This is an estimate. Actual resale prices may vary depending on condition, location and market demand.")
    except Exception as exc:
        st.error("Unable to generate the prediction. Check the model and input data.")
        st.code(str(exc))


# ============================================================
# ANALYTICS DATA - KEEP ORIGINAL MODEL FEATURE NAMES
# ============================================================
st.divider()
st.markdown(
    """
    <h1 style="text-align:center;color:#397df6;">📊 Car Price Analytics Dashboard</h1>
    <p style="text-align:center;color:#64748b;">
        Understand the factors affecting used-car resale prices
    </p>
    """,
    unsafe_allow_html=True,
)

required_columns = {
    "Year", "Selling_Price", "Present_Price", "Kms_Driven",
    "Fuel_Type", "Seller_Type", "Transmission", "Owner",
}
missing_columns = required_columns.difference(df.columns)

if missing_columns:
    st.warning("Analytics cannot be displayed because cars.csv is missing: " + ", ".join(sorted(missing_columns)))
else:
    analytics = df.copy()
    analytics["Fuel_Type"] = analytics["Fuel_Type"].map(FUEL_MAP)
    analytics["Seller_Type"] = analytics["Seller_Type"].map(SELLER_MAP)
    analytics["Transmission"] = analytics["Transmission"].map(TRANSMISSION_MAP)
    analytics["Age"] = REFERENCE_YEAR - pd.to_numeric(analytics["Year"], errors="coerce")

    # This DataFrame uses the exact seven feature names stored in model.sav.
    analytics_features = analytics[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    valid_mask = analytics_features.notna().all(axis=1) & analytics["Selling_Price"].notna()
    analytics_features = analytics_features.loc[valid_mask]
    actual_prices = pd.to_numeric(analytics.loc[valid_mask, "Selling_Price"], errors="coerce")

    if analytics_features.empty:
        st.warning("No valid rows were found in cars.csv for the analytics charts.")
    else:
        try:
            analytics_predictions = model.predict(analytics_features)
            comparison = pd.DataFrame({
                "Actual Price": actual_prices.to_numpy(),
                "Predicted Price": analytics_predictions,
            })
        except Exception as exc:
            comparison = pd.DataFrame()
            st.warning("The model could not generate analytics predictions.")
            st.code(str(exc))

        tab1, tab2, tab3 = st.tabs([
            "📌 Feature Importance",
            "🎯 Actual vs Predicted",
            "📈 Price Distribution",
        ])

        with tab1:
            st.subheader("Factors Affecting Car Prices")
            if hasattr(model, "feature_importances_"):
                importance = pd.DataFrame({
                    "Feature": [FEATURE_LABELS[name] for name in MODEL_FEATURES],
                    "Importance (%)": model.feature_importances_ * 100,
                }).sort_values("Importance (%)", ascending=False)
                chart_data = importance.set_index("Feature")
                st.bar_chart(chart_data, height=430)
                st.caption("A higher percentage means the feature was used more heavily by the Random Forest when making splits. It does not prove causation.")
            else:
                st.info("Feature importance is not available for this model type.")

        with tab2:
            st.subheader("Actual Price vs Model Predicted Price")
            if not comparison.empty:
                st.scatter_chart(
                    comparison,
                    x="Actual Price",
                    y="Predicted Price",
                    height=480,
                    use_container_width=True,
                )
                st.caption("Each dot represents one car in cars.csv. Points closer to the diagonal relationship indicate more accurate predictions.")
                st.info("This chart uses the available project dataset. If that dataset was used for training, it is not an independent test-set evaluation.")
            else:
                st.info("The comparison chart is unavailable because predictions could not be generated.")

        with tab3:
            st.subheader("Distribution of Used-Car Selling Prices")
            price_values = pd.to_numeric(df["Selling_Price"], errors="coerce").dropna()
            if not price_values.empty:
                counts, edges = np.histogram(price_values, bins=20)
                bins = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
                distribution = pd.DataFrame({"Price Range (₹ lakhs)": bins, "Number of Cars": counts})
                st.bar_chart(distribution.set_index("Price Range (₹ lakhs)"), height=420)
                st.caption("The chart shows how many cars in the dataset fall within each selling-price range.")
            else:
                st.info("No valid selling-price values were found in cars.csv.")


# ============================================================
# MODEL INFORMATION
# ============================================================
st.divider()
st.markdown('<div class="section-heading">About Our Prediction Model</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    st.subheader("🌳 Random Forest Regression")
    st.write(
        "Random Forest Regression combines predictions from multiple decision trees. "
        "The model uses patterns learned from historical vehicle data to estimate a used car's resale price."
    )
with right:
    st.subheader("⚙️ Model Features")
    st.write(
        "The saved model expects these inputs: current market price, kilometres driven, "
        "fuel type, seller type, transmission, previous owners and car age."
    )


# ============================================================
# FOOTER - HTML IS RENDERED, NOT SHOWN AS CODE
# ============================================================
st.markdown(
    """
    <div class="footer-box">
        <p>© 2025 Car Price Oracle. All rights reserved.</p>
        <p>Powered by Machine Learning and Streamlit</p>
        <div class="footer-sub">
            Car Price Prediction&nbsp; | &nbsp;Random Forest Regression&nbsp; | &nbsp;Machine Learning Project
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
