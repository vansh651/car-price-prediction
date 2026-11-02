
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Car Price Oracle",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).parent

# =====================================================
# LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model.sav")


@st.cache_data
def load_dataset():
    return pd.read_csv(BASE_DIR / "cars.csv")


try:
    model = load_model()
    df = load_dataset()

except Exception as e:
    st.error("Unable to load the model or dataset.")
    st.exception(e)
    st.stop()


# =====================================================
# CUSTOM WEBSITE DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f1f5f9;
    color: #172b4d;
}

.block-container {
    max-width: 1500px;
    padding: 1rem 2rem 3rem 2rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #397df6, #245fd0);
    padding: 48px 20px;
    text-align: center;
    border-radius: 14px;
    margin-bottom: 30px;
}

.hero h1 {
    color: white !important;
    font-size: 43px;
    font-weight: 800;
}

.hero p {
    color: white !important;
    font-size: 20px;
}

.section-title {
    color: #397df6;
    font-size: 30px;
    font-weight: 750;
    border-bottom: 2px solid #dbe5f1;
    padding-bottom: 15px;
    margin: 20px 0;
}

.stApp label {
    color: #172b4d !important;
    font-weight: 600 !important;
}

.stNumberInput input {
    background: white !important;
    color: #172b4d !important;
    border: 1px solid #ccd6e4 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background: white !important;
    color: #172b4d !important;
    border-radius: 8px !important;
}

.stFormSubmitButton button {
    background: #397df6 !important;
    color: white !important;
    border: none !important;
    padding: 12px 30px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stFormSubmitButton button:hover {
    background: #245fd0 !important;
}

.prediction-box {
    background: white;
    border-radius: 14px;
    padding: 35px;
    text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07);
    margin: 20px 0;
}

.prediction-label {
    color: #64748b;
    font-size: 18px;
}

.prediction-price {
    color: #397df6;
    font-size: 48px;
    font-weight: 800;
    margin: 15px 0;
}

.prediction-description {
    color: #475569;
    font-size: 16px;
}

.detail-card {
    background: white;
    padding: 22px 10px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    min-height: 105px;
    margin-bottom: 12px;
}

.detail-label {
    color: #8493a8;
    font-size: 15px;
    margin-bottom: 10px;
}

.detail-value {
    color: #172b4d;
    font-size: 19px;
    font-weight: 700;
}

.analytics-card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.custom-footer {
    background: #101a30;
    color: white;
    text-align: center;
    padding: 38px 15px;
    border-radius: 12px;
    margin-top: 45px;
}

.custom-footer p {
    color: white !important;
    font-size: 16px;
    margin: 10px;
}

.footer-links {
    color: #cbd5e1;
    font-size: 14px;
    margin-top: 18px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🚘 Car Price Oracle</h1>
    <p>Predict your car's resale value with machine learning</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# CAR PREDICTION FORM
# =====================================================

st.markdown(
    '<div class="section-title">Enter Your Car Details</div>',
    unsafe_allow_html=True
)

with st.form("car_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        present_price = st.number_input(
            "Current Market Price (in lakhs)",
            min_value=0.1,
            max_value=1000.0,
            value=5.0,
            step=0.1
        )

        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=30000,
            step=500
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG"]
        )

        seller_type = st.selectbox(
            "Seller Type",
            ["Dealer", "Individual"]
        )

    with col2:

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

        owner = st.selectbox(
            "Number of Previous Owners",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 (First Owner)",
                1: "1 (Second Owner)",
                2: "2 (Third Owner)",
                3: "3 (Fourth Owner or more)"
            }[x]
        )

        year = st.number_input(
            "Year of Purchase",
            min_value=1990,
            max_value=2025,
            value=2020,
            step=1
        )

    submitted = st.form_submit_button(
        "🚗 Predict Price",
        use_container_width=True
    )


# =====================================================
# PREDICTION LOGIC
# =====================================================

if submitted:

    fuel_dict = {
        "Petrol": 0,
        "Diesel": 1,
        "CNG": 2
    }

    seller_dict = {
        "Dealer": 0,
        "Individual": 1
    }

    transmission_dict = {
        "Manual": 0,
        "Automatic": 1
    }

    # Preserve the original model's age calculation
    age = 2025 - int(year)

    features = np.array([[
        float(present_price),
        int(kms_driven),
        fuel_dict[fuel_type],
        seller_dict[seller_type],
        transmission_dict[transmission],
        int(owner),
        age
    ]])

    try:

        prediction = float(model.predict(features)[0])

        st.markdown(
            '<div class="section-title">Your Prediction</div>',
            unsafe_allow_html=True
        )

        st.success("Prediction generated successfully!")

        st.markdown(f"""
        <div class="prediction-box">

            <div class="prediction-label">
                Estimated Resale Price
            </div>

            <div class="prediction-price">
                ₹{prediction:.2f} Lakhs
            </div>

            <div class="prediction-description">
                Based on your car details and our trained
                machine learning model.
            </div>

        </div>
        """, unsafe_allow_html=True)

        details = [
            ("Current Price", f"₹{present_price:.2f} Lakhs"),
            ("Kilometers", f"{kms_driven:,} km"),
            ("Fuel Type", fuel_type),
            ("Seller Type", seller_type),
            ("Transmission", transmission),
            ("Previous Owners", str(owner)),
            ("Year of Purchase", str(year)),
            ("Car Age", f"{age} years")
        ]

        for start in range(0, len(details), 4):

            cols = st.columns(4)

            for col, (label, value) in zip(
                cols, details[start:start + 4]
            ):

                with col:

                    st.markdown(f"""
                    <div class="detail-card">

                        <div class="detail-label">
                            {label}
                        </div>

                        <div class="detail-value">
                            {value}
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

        st.info(
            "This is an estimated resale value. "
            "Actual market prices may vary."
        )

    except Exception as e:

        st.error("Unable to generate prediction.")
        st.exception(e)


# =====================================================
# ANALYTICS DASHBOARD
# =====================================================

st.markdown("---")

st.markdown("""
<h1 style="text-align:center; color:#397df6;">
    📊 Car Price Analytics Dashboard
</h1>

<p style="text-align:center; color:#64748b;">
    Understand the factors affecting used car resale prices
</p>
""", unsafe_allow_html=True)


# =====================================================
# PREPARE DATA FOR ANALYTICS
# =====================================================

fuel_encoding = {
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
}

seller_encoding = {
    "Dealer": 0,
    "Individual": 1
}

transmission_encoding = {
    "Manual": 0,
    "Automatic": 1
}

feature_names = [
    "Present Price",
    "Kilometers Driven",
    "Fuel Type",
    "Seller Type",
    "Transmission",
    "Previous Owners",
    "Car Age"
]

# Prepare dataset using the same encoding as prediction
analytics_df = df.copy()

analytics_df["Age"] = 2025 - analytics_df["Year"]

analytics_df["Fuel_Type_Encoded"] = (
    analytics_df["Fuel_Type"].map(fuel_encoding)
)

analytics_df["Seller_Type_Encoded"] = (
    analytics_df["Seller_Type"].map(seller_encoding)
)

analytics_df["Transmission_Encoded"] = (
    analytics_df["Transmission"].map(transmission_encoding)
)

model_features = pd.DataFrame({
    "Present Price": analytics_df["Present_Price"],
    "Kilometers Driven": analytics_df["Kms_Driven"],
    "Fuel Type": analytics_df["Fuel_Type_Encoded"],
    "Seller Type": analytics_df["Seller_Type_Encoded"],
    "Transmission": analytics_df["Transmission_Encoded"],
    "Previous Owners": analytics_df["Owner"],
    "Car Age": analytics_df["Age"]
})

valid_rows = model_features.notna().all(axis=1)

model_features = model_features.loc[valid_rows]
actual_prices = analytics_df.loc[valid_rows, "Selling_Price"]

# Generate predictions using the existing saved model
try:
    dataset_predictions = model.predict(model_features)

    comparison_df = pd.DataFrame({
        "Actual Price": actual_prices.to_numpy(),
        "Predicted Price": dataset_predictions
    })

except Exception as e:
    comparison_df = pd.DataFrame()
    st.warning(f"Analytics predictions unavailable: {e}")


# =====================================================
# ANALYTICS TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📌 Feature Importance",
    "🎯 Actual vs Predicted",
    "📈 Price Distribution"
])


# =====================================================
# GRAPH 1: FEATURE IMPORTANCE
# =====================================================

with tab1:

    st.markdown("### Factors Affecting Car Prices")

    estimator = model

    if hasattr(model, "named_steps"):
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": estimator.feature_importances_
        })

        importance_df["Percentage"] = (
            importance_df["Importance"] /
            importance_df["Importance"].sum()
        ) * 100

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=True
        )

        fig1 = px.bar(
            importance_df,
            x="Percentage",
            y="Feature",
            orientation="h",
            text=importance_df["Percentage"].map(
                lambda x: f"{x:.1f}%"
            ),
            color="Percentage",
            color_continuous_scale=[
                "#bfdbfe",
                "#397df6",
                "#1749a5"
            ]
        )

        fig1.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0
        )

        fig1.update_layout(
            template="plotly_white",
            height=480,
            xaxis_title="Feature Importance (%)",
            yaxis_title="",
            coloraxis_showscale=False,
            showlegend=False,
            margin=dict(l=25, r=65, t=30, b=35),
            font=dict(
                family="Arial",
                size=13,
                color="#172b4d"
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        st.caption(
            "Higher percentages indicate features that have "
            "a greater influence on the Random Forest model."
        )

    else:

        st.warning(
            "Feature importance is not available for this model."
        )


# =====================================================
# GRAPH 2: ACTUAL VS PREDICTED
# =====================================================

with tab2:

    st.markdown("### Actual Price vs Model Predicted Price")

    if not comparison_df.empty:

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=comparison_df["Actual Price"],
            y=comparison_df["Predicted Price"],
            mode="markers",
            name="Car Predictions",
            marker=dict(
                color="#397df6",
                size=9,
                opacity=0.65,
                line=dict(
                    color="white",
                    width=1
                )
            ),
            hovertemplate=(
                "Actual: ₹%{x:.2f} Lakhs<br>"
                "Predicted: ₹%{y:.2f} Lakhs"
                "<extra></extra>"
            )
        ))

        minimum = min(
            comparison_df["Actual Price"].min(),
            comparison_df["Predicted Price"].min()
        )

        maximum = max(
            comparison_df["Actual Price"].max(),
            comparison_df["Predicted Price"].max()
        )

        fig2.add_trace(go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="Perfect Prediction",
            line=dict(
                color="#ef4444",
                width=2,
                dash="dash"
            )
        ))

        fig2.update_layout(
            template="plotly_white",
            height=480,
            xaxis_title="Actual Selling Price (Lakhs)",
            yaxis_title="Predicted Selling Price (Lakhs)",
            hovermode="closest",
            legend=dict(
                orientation="h",
                y=1.12,
                x=0
            ),
            margin=dict(l=30, r=30, t=55, b=40),
            font=dict(
                family="Arial",
                size=13,
                color="#172b4d"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.caption(
            "Each dot represents a dataset record. "
            "Dots closer to the red dashed line indicate "
            "closer predictions."
        )

        st.info(
            "Note: This graph compares predictions on the "
            "available dataset used for training. It is not "
            "a separate held-out test-set evaluation."
        )

    else:

        st.warning("Unable to generate the comparison graph.")


# =====================================================
# GRAPH 3: PRICE DISTRIBUTION
# =====================================================

with tab3:

    st.markdown("### Distribution of Used Car Prices")

    price_df = df.dropna(subset=["Selling_Price"])

    fig3 = px.histogram(
        price_df,
        x="Selling_Price",
        nbins=30,
        color_discrete_sequence=["#397df6"],
        labels={
            "Selling_Price": "Selling Price (Lakhs)",
            "count": "Number of Vehicles"
        }
    )

    fig3.update_traces(
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate=(
            "Price: ₹%{x:.2f} Lakhs<br>"
            "Number of Vehicles: %{y}"
            "<extra></extra>"
        )
    )

    fig3.update_layout(
        template="plotly_white",
        height=480,
        xaxis_title="Selling Price (Lakhs)",
        yaxis_title="Number of Vehicles",
        showlegend=False,
        bargap=0.08,
        margin=dict(l=30, r=30, t=30, b=40),
        font=dict(
            family="Arial",
            size=13,
            color="#172b4d"
        )
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.caption(
        "This histogram shows how frequently different "
        "resale price ranges occur in the dataset."
    )


# =====================================================
# MODEL INFORMATION
# =====================================================

st.markdown("---")

st.markdown("""
<div class="section-title">
    About Our Prediction Model
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🌳 Random Forest Regression")

    st.write("""
    Random Forest Regression is an ensemble machine learning
    algorithm that combines multiple decision trees to
    generate a final numerical prediction.

    Our model uses historical vehicle data to estimate
    the resale price of used vehicles.
    """)

with col2:

    st.markdown("### ⚙️ Model Features")

    st.write("""
    The prediction considers:

    - Current market price
    - Kilometers driven
    - Fuel type
    - Seller type
    - Transmission
    - Previous owners
    - Vehicle age
    """)


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="custom-footer">

    <p>© 2025 Car Price Oracle. All rights reserved.</p>

    <p>Powered by Machine Learning and Streamlit</p>

    <div class="footer-links">
        Car Price Prediction |
        Random Forest Regression |
        Machine Learning Project
    </div>

</div>
""", unsafe_allow_html=True)
