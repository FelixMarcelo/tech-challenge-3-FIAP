# data
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np 
import plotly.graph_objects as go

# AI
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Google API
from pytrends.request import TrendReq

class Data_service():
    def get_data(self, ticker: str, initial_date: str, final_date: str, n_days_target: int):
        # Make connection
        pytrends = TrendReq(hl='en-US', tz=360)

        # Prepare Payload
        kw_list = [ticker]
        today = datetime.now().date()
        # one_year_ago = today - timedelta(days=365)
        pytrends.build_payload(kw_list, cat=0, timeframe=f'{initial_date} {final_date}', geo='BR', gprop='')
        
        # Get data
        df_trends = pytrends.interest_over_time()
        
        # get google trend from the last month
        df_trends['qtd_shifted'] = df_trends[ticker].shift(1)
        
        # Get data from BOVESPA
        historical_data = yf.download(f'{ticker}.SA', initial_date, final_date)
        
        # Merging BOVESPA data with Google Trends data
        historical_with_trends = historical_data.copy()
        historical_with_trends['trend_count'] = -1

        for i in df_trends.index:
            historical_with_trends['trend_count'] = np.where(((historical_with_trends.index.year == i.year) & (historical_with_trends.index.month == i.month)), 
                                                            df_trends[df_trends.index == i]['qtd_shifted'], historical_with_trends['trend_count'])
            
        # Creating Predictive Variables
        wanted_pct_change = 0.05
        historical_with_trends_return = historical_with_trends.copy()

        # target variable
        historical_with_trends_return['target'] = historical_with_trends_return['Close'].pct_change(n_days_target).shift(-n_days_target)
        historical_with_trends_return['target_cat'] = np.where(historical_with_trends_return['target'] >= wanted_pct_change, 1, 0)
        # direction 
        historical_with_trends_return['dir_D'] = np.where(historical_with_trends_return['Close'] > historical_with_trends_return['Open'], 1, 0)
        historical_with_trends_return['dir_D1'] = historical_with_trends_return['dir_D'].shift(1)
        historical_with_trends_return['dir_D2'] = historical_with_trends_return['dir_D'].shift(2)
        historical_with_trends_return['dir_D3'] = historical_with_trends_return['dir_D'].shift(3)
        historical_with_trends_return['dir_D4'] = historical_with_trends_return['dir_D'].shift(4)
        historical_with_trends_return['dir_D5'] = historical_with_trends_return['dir_D'].shift(5)
        # High mean for the last 10 days
        historical_with_trends_return["dir_D_mean"] = historical_with_trends_return["dir_D"].rolling(10).mean()
        # Normalize the 'Value' column
        historical_with_trends_return['trend_count_norm'] = (historical_with_trends_return['trend_count'] - historical_with_trends_return['trend_count'].min()) / (historical_with_trends_return['trend_count'].max() - historical_with_trends_return['trend_count'].min())
        # Standardize the 'Value' column
        mean = historical_with_trends_return['trend_count'].mean()
        std = historical_with_trends_return['trend_count'].std()
        historical_with_trends_return['trend_count_std'] = (historical_with_trends_return['trend_count'] - mean) / std
        
        # choose variables
        df = historical_with_trends_return[['dir_D1', 'dir_D2', 'dir_D3', 'dir_D4', 'dir_D5', 'dir_D_mean', 'trend_count_std', 'target_cat']]
        df_filtered = df.dropna(axis=0)
        df2 = df_filtered.copy()
        
        # Balance dataset
        df_minority = df2[df2['target_cat'] == 1]
        df_majority = df2[df2['target_cat'] == 0]

        df_majority_undersampled = resample(df_majority, 
                                            replace=False,
                                            n_samples=len(df_minority),
                                            random_state=42)

        # Combine the undersampled majority and minority data
        df_balanced = pd.concat([df_majority_undersampled, df_minority])

        # Shuffle the dataset
        # df_balanced = df_balanced.sample(frac=1, random_state=42)
        
        return df_balanced
    
    def train_new_model(self, df_balanced: pd.DataFrame, use_grid_search: bool = False):
        # Separate data
        X = df_balanced.iloc[:, :-1]  # Feature columns
        y = df_balanced.iloc[:, -1]  # Target column

        # Split the dataset into 80% training and 20% testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if (use_grid_search == False):       
            # Create a RandomForestClassifier instance
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train the model on the training set
            rf_classifier.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = rf_classifier.predict(X_test)
            
            rf_classifier           
            
            return rf_classifier, accuracy_score(y_test, y_pred)
        
        else:
            # Trying with grid search
            param_grid = {
                'n_estimators': [100, 200, 300],  # Number of trees in the forest
                'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
                'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
                'bootstrap': [True, False]        # Whether bootstrap samples are used when building trees
            }

            rf_classifier = RandomForestClassifier(random_state=42)

            grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring='accuracy')

            # Perform the grid search on the training set
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            best_rf_model = grid_search.best_estimator_
            
            # Evaluate the Best Model
            # Make predictions on the test set
            y_pred = best_rf_model.predict(X_test) 
            
            return best_rf_model, accuracy_score(y_test, y_pred)
        
    def predict(self, model, x_variables):        
        return model.predict(x_variables)
        