import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.regularizers import l2

# Prepare data for LSTM
def prepare_data(df, time_step=30):
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Coerce invalid dates to NaT

    # If there are NaT values after conversion, handle them (e.g., remove or fill)
    if df['Date'].isna().any():
        st.warning("There are invalid dates in your data. These rows will be removed.")
        df = df.dropna(subset=['Date'])  # Remove rows with invalid dates

    df = df.sort_values('Date')  # Ensure data is sorted by date
    value_data = df[['Value']].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(value_data)

    # Create sequences
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, y, scaler

# LSTM model for time series forecasting
def create_improved_lstm_model(input_shape, units=15, dropout=0.4, l2_penalty=0.002):
    model = Sequential()

    # Adding LSTM layer with L2 regularization
    model.add(LSTM(units=units, input_shape=input_shape, kernel_regularizer=l2(l2_penalty), return_sequences=False))

    # Dropout layer for regularization
    model.add(Dropout(dropout))

    # Adding Dense layer with L2 regularization for output prediction
    model.add(Dense(1, kernel_regularizer=l2(l2_penalty)))

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train and forecast with LSTM
def train_and_forecast_lstm(df, time_step=30, forecast_end_date='2023-01-02'):
    X, y, scaler = prepare_data(df, time_step)

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Create and train the model
    model = create_improved_lstm_model((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=14, batch_size=64, verbose=1)

    # Evaluate the model
    y_pred_scaled = model.predict(X_test)
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

    # Metrics
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)

    # Forecast
    last_data = df[['Value']].tail(time_step).values
    last_data_scaled = scaler.transform(last_data).reshape(1, time_step, 1)
    forecast_start_date = df['Date'].max() + pd.Timedelta(days=1)
    forecast_end_date = pd.to_datetime(forecast_end_date)
    forecast_steps = (forecast_end_date - forecast_start_date).days + 1

    predictions = []
    forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_steps, freq='D')

    for _ in range(forecast_steps):
        pred = model.predict(last_data_scaled)
        predictions.append(pred[0][0])
        last_data_scaled = np.roll(last_data_scaled, -1, axis=1)
        last_data_scaled[0, -1, 0] = pred

    # Inverse scaling
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    forecast_df = pd.DataFrame(predictions, columns=['Forecast'], index=forecast_dates)

    return forecast_df, {"MAE": mae, "RMSE": rmse, "R-squared": r2}

# Streamlit UI function
def price_prediction(df):
    # Get unique crops
    crops = df['Item'].unique()
    
    # Multi-select for crops
    selected_crops = st.multiselect("Select crops for forecasting:", options=crops, default=crops)

    # Input forecast end date
    forecast_end_date = st.date_input(
        "Select forecast end date:", value=df['Date'].max() + pd.Timedelta(days=30)
    )

    # Input ranking date
    ranking_date = st.date_input(
        "Select a date to rank prices:", value=df['Date'].max() + pd.Timedelta(days=30)
    )

    if st.button("Run Forecasting"):
        st.write("### Forecasting Results")
        
        forecast_results = {}
        combined_forecast = pd.DataFrame()  # To store all crop forecasts for combined graph

        for crop in selected_crops:
            crop_df = df[df['Item'] == crop]
            forecast, performance = train_and_forecast_lstm(
                crop_df, forecast_end_date=str(forecast_end_date)
            )
            forecast_results[crop] = forecast

            # Add crop forecast to combined DataFrame, scaling by 1000
            combined_forecast[crop] = (forecast['Forecast'] / 1000).round(2)

        # Ensure the index (dates) is set for combined forecast
        combined_forecast.index = forecast.index

        # Plot combined graph with labels
        st.write("### Combined Forecast for Selected Crops")
        plt.figure(figsize=(10, 6))
        for crop in combined_forecast.columns:
            plt.plot(combined_forecast.index, combined_forecast[crop], label=crop)
        plt.xlabel("Date")
        plt.ylabel("Price/Kg (RM)")
        plt.title("Forecasted Prices for Selected Crops")
        plt.legend(title="Crops")
        plt.grid()
        st.pyplot(plt)

        # Display ranking for the selected ranking date
        st.write("### Ranking Predicted Prices")
        ranking_date = pd.to_datetime(ranking_date)
        ranked_prices = []

        for crop, forecast in forecast_results.items():
            if ranking_date in forecast.index:
                price = forecast.loc[ranking_date, 'Forecast'] / 1000  # Scale by 1000
                ranked_prices.append((crop, f"{price:.2f}"))  # Round to 2 decimal places
            else:
                st.warning(f"Ranking date {ranking_date.date()} is outside the forecast range for {crop}.")

        # Sort crops by price
        ranked_prices.sort(key=lambda x: x[1], reverse=True)
        
        # Display ranking
        ranking_df = pd.DataFrame(ranked_prices, columns=["Crop", "Predicted Price/Kg (RM)"])
        st.write(ranking_df)
