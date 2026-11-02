
import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.sav"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load model.sav")
    st.exception(e)
    st.stop()

# Website design

import streamlit as st
import numpy as np
import joblib
import pandas as pd
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

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main website */
.stApp {
    background-color: #f1f5f9;
    color: #172b4d;
}

.block-container {
    padding: 0rem 1.5rem 2rem 1.5rem;
    max-width: 1500px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #367ff0, #2864d9);
    padding: 48px 20px;
    text-align: center;
    border-radius: 0 0 15px 15px;
    margin: 0 -1.5rem 25px -1.5rem;
    color: white;
}

.hero h1 {
    color: white !important;
    font-size: 43px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: white !important;
    font-size: 21px;
    margin: 0;
}

/* Section cards */
.section-card {
    background: white;
    padding: 30px;
    border-radius: 13px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.09);
    margin-bottom: 25px;
}

.section-title {
    color: #397cf0;
    font-size: 30px;
    font-weight: 750;
    border-bottom: 2px solid #e1e8f0;
    padding-bottom: 15px;
    margin-bottom: 25px;
}

/* Form labels */
.stApp label {
    color: #172b4d !important;
    font-weight: 600 !important;
}

/* Input fields */
.stNumberInput input {
    background: white !important;
    color: #172b4d !important;
    border: 1px solid #ccd6e4 !important;
    border-radius: 7px !important;
    min-height: 48px;
}

div[data-baseweb="select"] > div {
    background: white !important;
    color: #172b4d !important;
    border: 1px solid #ccd6e4 !important;
    border-radius: 7px !important;
    min-height: 48px;
}

/* Predict button */
.stFormSubmitButton button {
    background: #397cf0 !important;
    color: white !important;
    border: none !important;
    padding: 13px 30px !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    min-height: 48px;
    transition: 0.2s;
}

.stFormSubmitButton button:hover {
    background: #205fce !important;
}

/* Prediction section */
.prediction-box {
    background: #f0f5fc;
    border-radius: 12px;
    padding: 35px 20px;
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    color: #52647c;
    font-size: 18px;
}

.prediction-price {
    color: #397cf0;
    font-size: 49px;
    font-weight: 800;
    margin: 12px 0;
}

.prediction-description {
    color: #334155;
    font-size: 17px;
}

/* Result detail cards */
.detail-card {
    background: white;
    padding: 20px 10px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 7px rgba(0,0,0,0.07);
    min-height: 105px;
    margin-top: 12px;
}

.detail-label {
    color: #8292a8;
    font-size: 15px;
    margin-bottom: 9px;
}

.detail-value {
    color: #14243b;
    font-size: 19px;
    font-weight: 700;
}

/* Footer */
.custom-footer {
    background: #101a30;
    color: white;
    text-align: center;
    padding: 38px 15px;
    margin: 35px -1.5rem -2rem -1.5rem;
}

.custom-footer p {
    color: white !important;
    font-size: 16px;
    margin: 9px;
}

.custom-footer .footer-links {
    color: #cbd5e1 !important;
    font-size: 15px;
    margin-top: 20px;
}

/* Mobile */
@media(max-width: 768px) {
    .hero h1 {
        font-size: 30px;
    }

    .hero p {
        font-size: 16px;
    }

    .prediction-price {
        font-size: 35px;
    }

    .section-card {
        padding: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.sav"
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


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
# CAR INPUT FORM
# =====================================================

st.markdown("""
<div class="section-title">Car Details</div>
""", unsafe_allow_html=True)

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

        st.write("")

    submitted = st.form_submit_button(
        "🚗 Predict Price",
        use_container_width=False
    )


# =====================================================
# PREDICTION LOGIC
# =====================================================

if submitted:

    # Encode categorical variables
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

    # Keep the same age convention used by the
    # existing trained model and original application.
    age = 2025 - int(year)

    # Model input
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

        st.markdown("""
        <div class="section-title">
            Predicted Resale Value
        </div>
        """, unsafe_allow_html=True)

        # Main prediction
        st.markdown(f"""
        <div class="prediction-box">

            <div class="prediction-label">
                Estimated Resale Price
            </div>

            <div class="prediction-price">
                ₹{prediction:.2f} lakhs
            </div>

            <div class="prediction-description">
                Based on the provided car details and our
                machine learning model
            </div>

        </div>
        """, unsafe_allow_html=True)

        # Car information cards
        details = [
            ("Current Price", f"₹{present_price:.2f} lakhs"),
            ("Kilometers", f"{kms_driven:,} km"),
            ("Fuel Type", fuel_type),
            ("Seller Type", seller_type),
            ("Transmission", transmission),
            ("Previous Owners", str(owner)),
            ("Year of Purchase", str(year)),
            ("Car Age", f"{age} years")
        ]

        st.write("")

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

        st.write("")

        st.info(
            "This prediction is an estimate generated by your "
            "trained machine learning model. Actual selling "
            "prices may vary."
        )

    except Exception as e:

        st.error("Unable to generate the prediction.")
        st.exception(e)


# =====================================================
# MACHINE LEARNING MODEL INFORMATION
# =====================================================

with st.expander("📊 Learn About Our Prediction Model"):

    st.subheader("How does Car Price Oracle work?")

    st.write("""
    Car Price Oracle uses a trained Random Forest Regression
    model to estimate the resale value of used cars.

    The model considers multiple factors, including:

    - Present market price
    - Kilometers driven
    - Fuel type
    - Seller type
    - Transmission
    - Previous owners
    - Vehicle age

    The model analyzes these inputs and generates an estimated
    resale price.
    """)

    if hasattr(model, "feature_importances_"):

        feature_names = [
            "Present Price",
            "Kilometers Driven",
            "Fuel Type",
            "Seller Type",
            "Transmission",
            "Previous Owners",
            "Car Age"
        ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        st.subheader("Feature Importance")

        st.bar_chart(
            importance_df.set_index("Feature")
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="custom-footer">

    <p>© 2025 Car Price Oracle. All rights reserved.</p>

    <p>Powered by Machine Learning and Streamlit</p>

    <div class="footer-links">
        Car Price Prediction | Random Forest Regression |
        Machine Learning Project
    </div>

</div>
""", unsafe_allow_html=True)


# Prediction
if submit:

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

    age = 2025 - int(year)

    features = np.array([[
        present_price,
        kms_driven,
        fuel_dict[fuel_type],
        seller_dict[seller_type],
        transmission_dict[transmission],
        owner,
        age
    ]])

    try:
        prediction = model.predict(features)[0]

        st.divider()
        st.header("Your Prediction")

        st.success("Prediction generated successfully!")

        st.metric(
            "Estimated Resale Price",
            f"₹ {prediction:.2f} Lakhs"
        )

        st.caption(
            "This is an estimated price generated by your trained "
            "machine learning model."
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.divider()

st.subheader("About the Project")
st.write("""
This project uses a Random Forest Regression model to predict
the resale price of used cars based on their features.

The model considers present price, kilometres driven, fuel type,
seller type, transmission, previous owners and vehicle age.
""")

if hasattr(model, "feature_importances_"):

    st.subheader("Machine Learning Feature Importance")

    import pandas as pd

    names = [
        "Present Price",
        "Kms Driven",
        "Fuel Type",
        "Seller Type",
        "Transmission",
        "Owner",
        "Age"
    ]

    importance = pd.DataFrame({
        "Feature": names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )
