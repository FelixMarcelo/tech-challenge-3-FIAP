import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

from service.Data_service import Data_service

# Define the Streamlit layout
st.title("Stock Prediction App")

st.sidebar.title("Select Stock Ticker")
ticker = st.sidebar.selectbox("Stock Ticker", ["BBAS3", "PETR4", "VALE3", "ITUB4"])
data_service = Data_service()

if st.sidebar.button("GET DATA!"):
    initial_date = '2014-01-01'
    today = datetime.now().date()
    n_days_target=15
    one_year_ago = today - timedelta(days=365)
    # Get data
    df_balanced = data_service.get_data(ticker, initial_date=initial_date, final_date=one_year_ago, n_days_target=n_days_target)
    df_validation = data_service.get_data(ticker=ticker, initial_date=one_year_ago, final_date=today, n_days_target=n_days_target)
    # Store data in session state
    st.session_state.df_balanced = df_balanced
    st.session_state.df_validation = df_validation    
    
    st.subheader("WOW! you've built your powerful dataset")
    st.write(df_balanced)

if (st.sidebar.button("Train a new model")):
    rf_model, report = data_service.train_new_model(st.session_state.df_balanced)
    st.subheader("Model Report")
    st.subheader(f"Accuracy: {report}")
    
    # Store the model in session state
    st.session_state.rf_model = rf_model
    
if st.sidebar.button("Predict"):
    validation_data = st.session_state.df_validation
    prediction = data_service.predict(st.session_state.rf_model, validation_data.iloc[:, :-1].tail(1))
    if prediction == 1:        
        st.subheader("OH MY GOD, OUR SUPER INTELLIGENCE IS TELLING YOU TO BUY IT")
    else:
        st.subheader("It is not the moment to buy. You should do nothing!")
        
           

    