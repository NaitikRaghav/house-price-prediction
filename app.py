
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/house_price_model.pkl"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run 'python train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 House Price Prediction")
st.write(
    "Estimate a California district's median house value "
    "using a trained Machine Learning regression model."
)

st.info(
    "Educational demonstration only. This model predicts median "
    "district-level house values from the California Housing dataset. "
    "It does not estimate the exact market price of an individual home."
)

st.divider()

st.subheader("Enter Housing Details")

left, right = st.columns(2)

with left:
    med_inc = st.number_input(
        "Median income (dataset units)",
        min_value=0.5,
        max_value=15.0,
        value=3.87,
        step=0.1,
        help="Median income for the district, measured in tens of thousands of dollars.",
    )

    house_age = st.number_input(
        "Median house age (years)",
        min_value=1.0,
        max_value=52.0,
        value=28.0,
        step=1.0,
    )

    avg_rooms = st.number_input(
        "Average rooms per household",
        min_value=0.5,
        max_value=20.0,
        value=5.4,
        step=0.1,
    )

    avg_bedrooms = st.number_input(
        "Average bedrooms per household",
        min_value=0.2,
        max_value=10.0,
        value=1.1,
        step=0.1,
    )

with right:
    population = st.number_input(
        "District population",
        min_value=1.0,
        max_value=40000.0,
        value=1400.0,
        step=100.0,
    )

    avg_occupancy = st.number_input(
        "Average household occupancy",
        min_value=0.5,
        max_value=100.0,
        value=3.0,
        step=0.1,
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.54,
        max_value=41.95,
        value=36.0,
        step=0.01,
        format="%.2f",
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-124.35,
        max_value=-114.31,
        value=-119.57,
        step=0.01,
        format="%.2f",
    )

if st.button("Predict House Value", type="primary", use_container_width=True):
    try:
        model = load_model()

        input_data = pd.DataFrame(
            [[
                med_inc,
                house_age,
                avg_rooms,
                avg_bedrooms,
                population,
                avg_occupancy,
                latitude,
                longitude,
            ]],
            columns=FEATURES,
        )

        prediction = float(model.predict(input_data)[0])

        # Dataset prices are expressed in units of $100,000.
        predicted_rupees = prediction * 100000 * 83

        st.divider()
        st.subheader("Prediction Result")

        if prediction <= 0:
            st.warning(
                "The model produced a non-positive value. "
                "This input may be outside the model's useful range."
            )
        else:
            st.success("Prediction generated successfully!")

        st.metric(
    "Estimated Median House Value (INR)",
    f"₹{predicted_rupees:,.0f}",
)

        st.caption(
    f"Raw model output: {prediction:.4f} units of $100,000. "
    "The displayed rupee amount uses an illustrative exchange rate of ₹83 per US dollar. "
    "This is an estimate, not a guaranteed sale price."
)

        st.dataframe(input_data, use_container_width=True)

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception as error:
        st.error(f"Could not generate prediction: {error}")

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", "Linear Regression")
col2.metric("Features", "8")
col3.metric("Training Data", "20,640 records")

with st.expander("About the dataset and evaluation"):
    st.markdown(
        """
        - **Dataset:** California Housing dataset from scikit-learn
        - **Target:** Median district house value
        - **Preprocessing:** Median imputation and StandardScaler
        - **Train/test split:** 80% / 20%
        - **Evaluation metrics:** MAE, MSE, RMSE and R²
        - **Model:** Linear Regression inside a scikit-learn Pipeline
        """
    )

st.caption("House Price Prediction | Academic Machine Learning Project")
