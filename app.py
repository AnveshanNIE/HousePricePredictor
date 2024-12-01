import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
st.title("House Price Prediction App")
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with `houseSize` and `price` columns", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Title
    st.header("Dataset Preview")
    st.dataframe(df.head())

    # Initialize session state variables if they don't exist
    if 'lr_model' not in st.session_state:
        st.session_state.lr_model = LinearRegression()

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'target' not in st.session_state:
        st.session_state.target = None

    # Display dataset scatter plot
    st.subheader("Original Data: House Size vs Price")
    fig, ax = plt.subplots()
    ax.scatter(df["houseSize"], df["price"], color="red", label="Original Data")
    ax.set_title("House Size vs Price")
    ax.set_xlabel("House Size (sqft)")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

    # Buttons for model training and showing results
    if st.button("Train Model"):
        # Train the model
        st.session_state.data = df[["houseSize"]].values
        st.session_state.target = df["price"].values
        st.session_state.lr_model.fit(st.session_state.data, st.session_state.target)
        st.success("Model trained successfully!")

    if st.button("Show Training Results"):
        if st.session_state.data is not None and st.session_state.target is not None:
            # Generate predictions and calculate R²
            y_pred = st.session_state.lr_model.predict(st.session_state.data)
            r2 = r2_score(st.session_state.target, y_pred)

            # Display R² score
            st.subheader(f"R² Score: {r2:.2f}")

            # Plot original vs predictions
            st.subheader("Predictions vs Actual")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Left: Actual Data
            ax1.scatter(df["houseSize"], df["price"], color="red", label="Actual Prices")
            ax1.set_title("Actual Data")
            ax1.set_xlabel("House Size (sqft)")
            ax1.set_ylabel("Price ($)")
            ax1.legend()

            # Right: Predictions
            ax2.scatter(df["houseSize"], y_pred, color="blue", label="Predicted Prices")
            ax2.set_title("Predicted Data")
            ax2.set_xlabel("House Size (sqft)")
            ax2.set_ylabel("Price ($)")
            ax2.legend()
            st.pyplot(fig)
        else:
            st.warning("Please train the model first.")

    # Custom prediction form
    st.subheader("Predict House Price")
    custom_size = st.number_input("Enter the house size (in sqft):", min_value=0, value=None, step=100)
    if st.button("Predict Price"):
        if not hasattr(st.session_state.lr_model, "coef_"):
            st.warning("Please train the model first!")
        else:
            predicted_price = st.session_state.lr_model.predict([[custom_size]])[0]
            st.write(f"The predicted price for a house of size {custom_size} sqft is Rs{predicted_price:,.2f}")

else:
    st.warning("Please upload a CSV file to proceed!")
