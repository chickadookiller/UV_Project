import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import DeterministicProcess

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="UV Index Forecast App", layout="wide")
st.title("☀️ ระบบพยากรณ์ค่า UV Index ล่วงหน้า")

# --- โหลดข้อมูล ---
@st.cache_data # ใช้ Cache เพื่อให้แอปโหลดเร็วขึ้น
def load_data():
    df = pd.read_csv("Untitled spreadsheet - Sheet1.csv", parse_dates=['time'])
    df = df.sort_values("time").drop_duplicates('time')
    df = df[['time', 'uv_index_max ()']].rename(columns={'uv_index_max ()': 'uv'})
    df = df.set_index('time').asfreq('D')
    df['uv'] = df['uv'].ffill()
    return df

try:
    df = load_data()
    
    # --- ส่วนของการพยากรณ์ ---
    st.sidebar.header("ตั้งค่าการพยากรณ์")
    days_to_forecast = st.sidebar.slider("จำนวนวันที่ต้องการพยากรณ์", 1, 30, 14)

    if st.sidebar.button("เริ่มการคำนวณ"):
        with st.spinner('กำลังคำนวณ Model...'):
            # สร้าง Fourier Terms
            dp = DeterministicProcess(
                index=df.index,
                period=365,
                fourier=2,
                drop=True
            )
            exog_train = dp.in_sample()
            exog_future = dp.out_of_sample(steps=days_to_forecast)

            # Fit Model
            model = SARIMAX(df['uv'],
                          order=(1, 1, 1),
                          exog=exog_train,
                          seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)

            # Forecast
            forecast = model_fit.forecast(steps=days_to_forecast, exog=exog_future)
            
            # เตรียมข้อมูลสำหรับการแสดงผล
            forecast_df = pd.DataFrame({
                'Date': exog_future.index,
                'Predicted UV': forecast.values
            })

            # --- แสดงผลหน้าเว็บ ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"กราฟพยากรณ์ {days_to_forecast} วันข้างหน้า")
                fig, ax = plt.subplots(figsize=(10, 5))
                recent_data = df.iloc[-60:] # ดูย้อนหลัง 60 วัน
                ax.plot(recent_data.index, recent_data['uv'], label='Actual UV', color='black', alpha=0.5)
                ax.plot(forecast_df['Date'], forecast_df['Predicted UV'], label='Forecast', color='red', marker='o')
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.subheader("ข้อมูลตาราง")
                st.write(forecast_df)
                
                # ปุ่มดาวน์โหลด CSV
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "uv_forecast.csv", "text/csv")
    else:
        st.info("กดปุ่ม 'เริ่มการคำนวณ' ที่แถบด้านซ้ายเพื่อดูผลพยากรณ์")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")
    st.warning("กรุณาตรวจสอบว่ามีไฟล์ 'Untitled spreadsheet - Sheet1.csv' อยู่ในเครื่องมือส่งออก")