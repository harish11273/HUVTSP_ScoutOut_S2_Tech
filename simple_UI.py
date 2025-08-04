import streamlit as st
import numpy as np
import joblib as jl

model = jl.load('xgb_model.pkl')

st.title("HUVTSP ScoutOut Real Estate Price Prediction App")

st.subheader("by HUVTSP s2-int-scoutout-y2t")

st.write("Enter property details below:")

st.divider()

beds = st.number_input("Enter the Number of Bedrooms", min_value=1, max_value=10, value=3)
baths = st.number_input("Enter the Number of Bathrooms", min_value=1, max_value=10, value=3)
sqft = st.number_input("Enter the Square Footage", min_value=500, max_value=10000, value=1500)
lot = st.number_input("Enter the Lot Size in acres", min_value=0.001, max_value=10.0, value=0.5)
zipcode = st.number_input("Enter the Zip Code", min_value=00000, max_value=99999, value=12345)
date = st.date_input("Enter the Date of Sale (default 01/2000)", value=None)
if date is not None:
    year = date.year
    month = date.month
else:
    year = 2000  # or any default year you prefer
    month = 1    # or any default month you prefer

X = [beds, baths, sqft, lot, zipcode, year, month]

st.divider()

predictbutton = st.button("Predict Price!")

st.divider()

if predictbutton:
    st.balloons()
    X1 = np.array(X).reshape(1, -1)
    price = model.predict(X1)
    st.write(f"The predicted price of the property is: {price[0]:.2f}")
    
else:
    "Please use the button to predict the price"