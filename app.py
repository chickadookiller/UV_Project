import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import DeterministicProcess

# --- Page Config ---
st.set_page_config(page_title="UV Index Forecast", layout="wide")
st.title("☀️ UV Index Forecast: Specific Date Mode")

# --- Load Data ---
@st.cache_data
def load_data():
    # Make sure this filename matches exactly what is in your folder
    df = pd.read_csv("Untitled spreadsheet - Sheet1.csv", parse_dates=['time'])
    df = df.sort_values("time").drop_duplicates('time')
    df = df[['time', 'uv_index_max ()']].rename(columns={'uv_index_max ()': 'uv'})
    df = df.set_index('time').asfreq('D')
    df['uv'] = df['uv'].ffill()
    return df

try:
    df = load_data()
    last_date_in_data = df.index[-1].date()

    # --- Sidebar Configuration ---
    st.sidebar.header("Settings / ตั้งค่า")
    st.sidebar.write(f"📅 Last data available: **{last_date_in_data}**")
    
    # User selects a specific target date
    target_date = st.sidebar.date_input(
        "Select date to forecast (เลือกวันที่ต้องการพยากรณ์)", 
        value=last_date_in_data + pd.Timedelta(days=7),
        min_value=last_date_in_data + pd.Timedelta(days=1)
    )

    if st.sidebar.button("Calculate Forecast (เริ่มคำนวณ)"):
        
        # Calculate how many steps (days) from last data point to target date
        days_to_forecast = (target_date - last_date_in_data).days
        
        if days_to_forecast <= 0:
            st.error("Please select a date after the last available data point.")
        else:
            with st.spinner(f'Forecasting for {target_date} ({days_to_forecast} days ahead)...'):
                
                # --- Model Setup ---
                dp = DeterministicProcess(
                    index=df.index,
                    period=365,
                    fourier=2,
                    drop=True
                )
                exog_train = dp.in_sample()
                exog_future = dp.out_of_sample(steps=days_to_forecast)

                # Fit SARIMAX
                model = SARIMAX(df['uv'],
                                order=(1, 1, 1),
                                exog=exog_train,
                                seasonal_order=(0, 0, 0, 0))
                model_fit = model.fit(disp=False)

                # Forecast
                forecast = model_fit.forecast(steps=days_to_forecast, exog=exog_future)
                
                # Create Result DataFrame
                forecast_df = pd.DataFrame({
                    'Date': exog_future.index,
                    'Predicted UV': forecast.values
                })
                
                # Get the specific value for the target date
                target_value = forecast_df.iloc[-1]['Predicted UV']

                # --- Display Results ---
                
                # Metric Highlight
                st.metric(label=f"Predicted UV Index on {target_date}", value=f"{target_value:.2f}")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Forecast Trend")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Plot last 60 days of actual data
                    recent_data = df.iloc[-60:]
                    ax.plot(recent_data.index, recent_data['uv'], label='Actual History', color='gray', alpha=0.5)
                    
                    # Plot the forecast path
                    ax.plot(forecast_df['Date'], forecast_df['Predicted UV'], label='Forecast Path', color='blue', linestyle='--')
                    
                    # Highlight the specific target date
                    ax.scatter([forecast_df.iloc[-1]['Date']], [target_value], color='red', s=100, zorder=5, label='Target Date')

                    ax.set_title(f"Trajectory to {target_date}")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with col2:
                    st.subheader("Forecast Data")
                    st.dataframe(forecast_df.tail(10)) # Show last 10 days leading up to target
                    
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, f"uv_forecast_{target_date}.csv", "text/csv")

    else:
        st.info("Select a date in the sidebar and press 'Calculate Forecast'.")

except Exception as e:
    st.error(f"Error: {e}")
