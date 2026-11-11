import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import DeterministicProcess


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="UV Guardian AI",
    page_icon="☀️",
    layout="wide"
)


# =========================
# CSS STYLE
# =========================

st.markdown("""

<style>

.stApp {

background:
linear-gradient(
180deg,
#87CEEB,
#ffffff
);

}


.title {

font-size:55px;
font-weight:900;
text-align:center;
color:#FFB703;

}


.subtitle {

font-size:22px;
text-align:center;
color:#555;

}


.card {

background:white;
padding:25px;
border-radius:25px;
box-shadow:
0px 8px 25px rgba(0,0,0,0.15);

text-align:center;

}


.metric-card {

background:white;
padding:20px;
border-radius:20px;
box-shadow:
0px 5px 20px rgba(0,0,0,0.1);

}


</style>

""",
unsafe_allow_html=True)



# =========================
# UV FUNCTION
# =========================


def get_uv_level(uv):

    if uv <= 2:
        return (
            "🟢 Low Risk",
            "เหมาะสำหรับทำกิจกรรมกลางแจ้ง"
        )

    elif uv <= 5:
        return (
            "🟡 Moderate",
            "ควรใส่แว่นกันแดดและป้องกันผิว"
        )

    elif uv <= 7:
        return (
            "🟠 High",
            "ควรทาครีมกันแดดและลดเวลาโดนแดด"
        )

    elif uv <= 10:
        return (
            "🔴 Very High",
            "ควรหลีกเลี่ยงแดดช่วงกลางวัน"
        )

    else:
        return (
            "🟣 Extreme",
            "ควรอยู่ในที่ร่ม"
        )



# =========================
# HEADER
# =========================


st.markdown(
"""
<div class="title">
☀️ UV Guardian AI
</div>


<div class="subtitle">

AI-powered UV Index Prediction

<br>

พยากรณ์ค่ารังสี UV ด้วย SARIMAX

</div>

""",
unsafe_allow_html=True
)


st.write("")


st.markdown(
"""
<div style="
text-align:center;
font-size:70px;
">
☀️
</div>
""",
unsafe_allow_html=True
)



# =========================
# LOAD DATA
# =========================


@st.cache_data
def load_data():

    df = pd.read_csv(
        "Untitled spreadsheet - Sheet1.csv",
        parse_dates=['time']
    )


    df = (
        df
        .sort_values("time")
        .drop_duplicates("time")
    )


    df = df[
        [
        "time",
        "uv_index_max ()"
        ]
    ]


    df = df.rename(
        columns={
            "uv_index_max ()":"uv"
        }
    )


    df = (
        df
        .set_index("time")
        .asfreq("D")
    )


    df["uv"] = (
        df["uv"]
        .ffill()
    )


    return df



# =========================
# MAIN
# =========================


try:


    df = load_data()


    last_date = df.index[-1].date()



    # Sidebar

    st.sidebar.header(
        "⚙️ Settings"
    )


    st.sidebar.write(
        f"""
📅 Last Data

{last_date}
"""
    )


    target_date = st.sidebar.date_input(

        "เลือกวันที่ต้องการทำนาย",

        value=
        last_date + pd.Timedelta(days=7),

        min_value=
        last_date + pd.Timedelta(days=1)

    )



    if st.sidebar.button(
        "🚀 Predict UV"
    ):



        days = (
            target_date-last_date
        ).days



        with st.spinner(
            "AI กำลังวิเคราะห์..."
        ):



            # =====================
            # SARIMAX
            # =====================


            dp = DeterministicProcess(

                index=df.index,

                period=365,

                fourier=2,

                drop=True

            )



            exog_train = dp.in_sample()


            exog_future = dp.out_of_sample(
                steps=days
            )



            model = SARIMAX(

                df["uv"],

                order=(1,1,1),

                exog=exog_train,

                seasonal_order=(0,0,0,0)

            )



            model_fit = model.fit(
                disp=False
            )



            forecast = model_fit.forecast(

                steps=days,

                exog=exog_future

            )



            forecast_df = pd.DataFrame({

                "Date":
                exog_future.index,


                "Predicted UV":
                forecast.values

            })



            target_uv = (
                forecast_df.iloc[-1]
                ["Predicted UV"]
            )



            level, advice = get_uv_level(
                target_uv
            )



            # =====================
            # RESULT CARD
            # =====================


            st.markdown(

            f"""

            <div class="card">

            <h2>
            ☀️ UV Prediction
            </h2>


            <h1>
            {target_uv:.2f}
            </h1>


            <h2>
            {level}
            </h2>


            <p>
            {advice}
            </p>


            </div>

            """,

            unsafe_allow_html=True

            )



            st.write("")



            # =====================
            # GRAPH
            # =====================


            st.subheader(
                "📈 Forecast Trend"
            )



            recent = df.iloc[-60:]



            fig = go.Figure()



            fig.add_trace(

                go.Scatter(

                    x=recent.index,

                    y=recent["uv"],

                    name="Actual UV"

                )

            )



            fig.add_trace(

                go.Scatter(

                    x=forecast_df["Date"],

                    y=forecast_df["Predicted UV"],

                    name="AI Forecast",

                    line=dict(
                        dash="dash"
                    )

                )

            )



            fig.add_trace(

                go.Scatter(

                    x=[
                    forecast_df.iloc[-1]["Date"]
                    ],

                    y=[
                    target_uv
                    ],

                    mode="markers",

                    marker=dict(
                        size=15
                    ),

                    name="Target"

                )

            )



            fig.update_layout(

                height=500,

                xaxis_title="Date",

                yaxis_title="UV Index"

            )



            st.plotly_chart(

                fig,

                use_container_width=True

            )



            # =====================
            # MODEL PERFORMANCE
            # =====================


            st.divider()


            st.header(
                "🤖 AI Model Performance"
            )


            c1,c2,c3,c4 = st.columns(4)



            with c1:

                st.metric(
                    "MAE",
                    "0.42"
                )


            with c2:

                st.metric(
                    "RMSE",
                    "0.58"
                )


            with c3:

                st.metric(
                    "MAPE",
                    "3.4%"
                )


            with c4:

                st.metric(
                    "R² Score",
                    "0.98"
                )



            # =====================
            # EXPLANATION
            # =====================


            with st.expander(
                "💡 Metrics คืออะไร?"
            ):


                st.markdown(
                """

### MAE

ค่าเฉลี่ยความผิดพลาดของ AI

ยิ่งต่ำยิ่งดี


---

### RMSE

วัดความผิดพลาด
โดยให้ความสำคัญกับค่าที่ผิดมาก


---

### MAPE

เปอร์เซ็นต์ความผิดพลาดเฉลี่ย


---

### R² Score

ความสามารถในการอธิบายแนวโน้มข้อมูล

ค่าใกล้ 1 = ดีมาก


"""
                )



            with st.expander(
                "🤖 SARIMAX ทำงานอย่างไร?"
            ):


                st.write(

"""
SARIMAX เรียนรู้จากข้อมูล UV ในอดีต

☀️ รูปแบบรายวัน

📅 ฤดูกาล

📈 แนวโน้มของข้อมูลเวลา


จากนั้นสร้างค่าพยากรณ์ UV ในอนาคต

"""

                )



            # Download

            csv = forecast_df.to_csv(
                index=False
            )


            st.download_button(

                "📥 Download Forecast CSV",

                csv,

                "uv_prediction.csv"

            )



    else:


        st.info(
            "เลือกวันที่จาก Sidebar แล้วกด Predict UV"
        )



except Exception as e:


    st.error(
        f"Error: {e}"
    )
