import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('customer_churn_model.pkl')


# Dummy data (same as before)
example_data = pd.DataFrame({
    'Tenure': [1, 10, 20],
    'WarehouseToHome': [5, 10, 15],
    'DaySinceLastOrder': [2, 4, 6],
    'SatisfactionScore': [3, 4, 5],
    'NumberOfDeviceRegistered': [1, 2, 3],
    'NumberOfAddress': [1, 2, 3],
    'Complain': [0, 1, 0],
    'CashbackAmount': [100, 200, 300],
    'PreferedOrderCat': [0, 1, 2],
    'MaritalStatus': [0, 1, 2],
    'DummyFeature1': [0, 0, 0],  # Placeholder
    'DummyFeature2': [0, 0, 0],  # Placeholder
    'DummyFeature3': [0, 0, 0],  # Placeholder
    'DummyFeature4': [0, 0, 0],  # Placeholder
    'DummyFeature5': [0, 0, 0]   # Placeholder
})

# Fit the scaler
scaler = StandardScaler()
scaler.fit(example_data)

def main():
    st.set_page_config(page_title="ChurnX: Next-Gen Customer Retention Analytics", layout="wide")
    st.title("ChurnX: Next-Gen Customer Retention Analytics")
    
    # Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        background-color: #000000; /* Dark background color */
        color: #b300ff; /* Light text color */
        font-family: 'Roboto', sans-serif;
    }

    .stButton button {
        background-color: #ff5722;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }

    .stButton button:hover {
        background-color: #e64a19;
    }

    .stTextInput input, .stNumberInput input, .stSlider input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 10px;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stSlider input:focus {
        border-color: #ff5722;
    }

    .stSelectbox select {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #444444;
        border-radius: 5px;
        padding: 10px;
    }

    .stSelectbox select:focus {
        border-color: #ff5722;
    }
    
    .centered {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    height: 100vh; /* Full viewport height */
    text-align: center;
    </style>
    """, unsafe_allow_html=True)

    
    
    

    # Input fields (same as before)
    Tenure = st.selectbox("Tenure (Months with the company)", 
                          [1, 10, 0, 8, 9, 4, 7, 5, 11, 3, 6, 14, 12, 13, 2, 19, 15, 16, 20, 18, 
                           17, 23, 21, 24, 22, 30, 28, 27, 26, 25, 29, 31, 61, 51, 60, 50])
    WarehouseToHome = st.number_input("Distance from Warehouse to Home (km)", min_value=0)
    DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0)
    SatisfactionScore = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    NumberOfDeviceRegistered = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=1)
    NumberOfAddress = st.number_input("Number of Addresses", min_value=1, max_value=20, value=1)
    Complain = st.selectbox("Complain", [0, 1])
    CashbackAmount = st.number_input("Cashback Amount", min_value=0, max_value=1000, value=0)

    # Dropdown for categorical features
    PreferedOrderCat = st.selectbox("Preferred Order Category", 
                                    ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    # Map categorical values to numerical
    pref_order_cat_mapping = {"Laptop & Accessory": 0, "Mobile Phone": 1, "Fashion": 2, "Grocery": 3, "Others": 4}
    marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}

    # Create a DataFrame from the input data
    input_df = pd.DataFrame({
        'Tenure': [Tenure],
        'WarehouseToHome': [WarehouseToHome],
        'DaySinceLastOrder': [DaySinceLastOrder],
        'SatisfactionScore': [SatisfactionScore],
        'NumberOfDeviceRegistered': [NumberOfDeviceRegistered],
        'NumberOfAddress': [NumberOfAddress],
        'Complain': [Complain],
        'CashbackAmount': [CashbackAmount],
        'PreferedOrderCat': [pref_order_cat_mapping[PreferedOrderCat]],
        'MaritalStatus': [marital_status_mapping[MaritalStatus]],
        'DummyFeature1': [0],  # Placeholder
        'DummyFeature2': [0],  # Placeholder
        'DummyFeature3': [0],  # Placeholder
        'DummyFeature4': [0],  # Placeholder
        'DummyFeature5': [0]   # Placeholder
    })

    # Standardize the input data
    input_df_scaled = scaler.transform(input_df)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_df_scaled)
        st.subheader("Prediction")
        if prediction[0] == 0:
            st.success("The customer is predicted to **not churn**.", icon="✅")
        else:
            st.warning("The customer is predicted to **churn**.", icon="⚠️")

if __name__ == '__main__':
    main()
