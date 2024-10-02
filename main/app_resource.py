import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
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
    wanted_pct_change = 0.00
    df_balanced = data_service.get_data(ticker, initial_date=initial_date, final_date=one_year_ago, n_days_target=n_days_target, wanted_pct_change=wanted_pct_change)
    df_validation = data_service.get_data(ticker=ticker, initial_date=one_year_ago, final_date=today, n_days_target=n_days_target, param_return=True, wanted_pct_change=wanted_pct_change)
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
    validation_data = st.session_state.df_validation.sort_index()
    prediction = data_service.predict(st.session_state.rf_model, validation_data.iloc[:, :-2].tail(1))
    if prediction == 1:        
        st.subheader("OH MY GOD, OUR SUPER INTELLIGENCE IS TELLING YOU TO BUY IT.")
        st.subheader(f"there is a high chance that the share price of {ticker} will rise in the next 15 days")
    else:
        st.subheader("It is not the moment to buy. You should do nothing!")
        st.subheader(f"there is a high chance that the share price of {ticker} will fall in the next 15 days")
        
    st.subheader("Take a look at what your return would be like if, for each 15 days period, you consulted and followed the advice of our model during the last year, compared to a buy and hold strategy")
    
    validation_data['model_future_return'] = 0
    n = 15
    days_to_skip = n
    for i in range(len(validation_data)):
        if (days_to_skip < n) & (days_to_skip > 0):
            validation_data['model_future_return'].iloc[i] = 0
            days_to_skip -= 1
            continue
        else:
            if validation_data['target_cat'].iloc[i] == 1:
                if days_to_skip == n:
                    validation_data['model_future_return'].iloc[i] = validation_data['return_target'].iloc[i]
                    days_to_skip -= 1
                    continue
                if days_to_skip == 0:
                    days_to_skip = n
                    validation_data['model_future_return'].iloc[i] = validation_data['return_target'].iloc[i]
                    days_to_skip -= 1
                    continue
            if validation_data['target_cat'].iloc[i] == 0:
                if days_to_skip == n:
                    validation_data['model_future_return'].iloc[i] = -1*validation_data['return_target'].iloc[i]
                    days_to_skip -= 1
                    continue
                if days_to_skip == 0:
                    days_to_skip = n
                    validation_data['model_future_return'].iloc[i] = -1*validation_data['return_target'].iloc[i]
                    days_to_skip -= 1
                    continue        
    
    validation_data['buy_and_hold_future_return'] = np.where(validation_data['model_future_return'] == 0, 0, validation_data['return_target'])
    
    # chart comparing the model strategy with buy and hold strategy
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=validation_data.index, y=validation_data['model_future_return'].cumsum()*100, line=dict(color='green', width=2, dash='dash'), name='Model Strategy'))
    fig.add_trace(go.Scatter(x=validation_data.index, y=validation_data['buy_and_hold_future_return'].cumsum()*100, line=dict(color='firebrick', width=2), name='Buy and Hold Strategy'))
    
    fig.update_layout(title='Comparinson Between Both Strategies',
                   yaxis_title='Return Percentage')
    
    st.plotly_chart(fig)  
    
    st.write(validation_data)
        