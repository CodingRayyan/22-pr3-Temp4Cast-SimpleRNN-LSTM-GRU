import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Title
st.title("🌡️ Temperature Forecast Dashboard")

# Load sample historical data
df = pd.read_csv("utils/your_temperature_data.csv")  # Replace with your actual CSV path
df['Date'] = pd.to_datetime(df['Date'])

# --------------------------
# 📊 Show visualizations above model selection
# --------------------------
st.subheader("📈 Historical Temperature Visualizations")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['Temperature'], df['Date'], color='blue', alpha=0.6, edgecolors='k')
    ax1.set_title('Scatter: Temp vs Date')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Date')
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Temperature'], df['Date'], color='blue', alpha=0.6)
    ax2.set_title('Line: Temp vs Date')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Date')
    ax2.grid(True)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = plt.subplots(figsize=(6, 4))

    # Create a boxplot with styling
    box = ax3.boxplot(df['Temperature'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', color='navy'),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='gray', linestyle='--'),
                      capprops=dict(color='gray'),
                      flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, linestyle='none'))

    ax3.set_title('🌡️ Box Plot: Temperature', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Temperature (°C)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_facecolor('#f9f9f9')  # Light background

    st.pyplot(fig3)


with col4:
    fig4, ax4 = plt.subplots()
    ax4.hist(df['Temperature'], bins=15, color='skyblue', edgecolor='black')
    ax4.set_title('Histogram: Temperature')
    ax4.grid(True)
    st.pyplot(fig4)

# --------------------------
# 🔍 Model Selection
# --------------------------
st.subheader("🧠 Forecasting Model Selection")

model_choice = st.selectbox("Select Model", ["SimpleRNN", "LSTM", "GRU"])

model_paths = {
    "SimpleRNN": "model/rnn_temperature_model.keras",
    "LSTM": "model/lstm_temperature_model.keras",
    "GRU": "model/gru_temperature_model.keras"
}

try:
    model = load_model(model_paths[model_choice])
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Model loading error: {e}")

# --------------------------
# 🔢 Input and Forecast
# --------------------------
st.subheader("📥 Enter the last 30 days of temperature data:")


temps = st.text_area("Enter 30 comma-separated temperatures")


if st.button("🔮 Forecast Next 7 Days"):
    try:
        values = np.array([float(i.strip()) for i in temps.split(",")]).reshape(-1, 1)

        if len(values) != 30:
            st.error("⚠️ You must enter exactly 30 values.")
        else:
            scaled_input = scaler.transform(values).reshape(1, 30, 1)

            forecast = []
            for _ in range(7):
                pred = model.predict(scaled_input, verbose=0)[0][0]
                forecast.append(pred)
                scaled_input = np.append(scaled_input[:, 1:, :], [[[pred]]], axis=1)

            forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            # 📊 Plot forecast
            # 📊 Styled 7-Day Forecast Plot
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot forecast values with improved style
            ax.plot(range(1, 8), forecast_actual, marker='v', linestyle='--',color='#1f77b4',linewidth=2.5,label='Forecast',markersize=12,markerfacecolor='yellow',markeredgecolor='red',markeredgewidth = 2,alpha=0.9)

            # Enhancing visuals
            ax.set_title(f"📉 7-Day Temperature Forecast ({model_choice})", fontsize=14, fontweight='bold')
            ax.set_xlabel("Day", fontsize=12)
            ax.set_ylabel("Predicted Temperature (°C)", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6, color = '#006400')
            ax.set_facecolor('#f9f9f9')  # light background
            ax.tick_params(axis='both', labelsize=10)
            ax.legend()

            # Show plot in Streamlit
            st.pyplot(fig)

            # Show forecast values as text
            st.subheader("📋 Forecasted Temperature Values (Next 7 Days)")
            forecast_df = pd.DataFrame({
                "Day": [f"Day {i}" for i in range(1, 8)],
                "Predicted Temperature (°C)": forecast_actual.flatten().round(2)
            })
            st.table(forecast_df)

###

    except Exception as e:
        st.error(f"❌ Error during forecasting: {e}")

with st.sidebar.expander("👨‍💻 Developer"):
    st.markdown("- **Rayyan Ahmed**")
    st.markdown("- **IBM Certifed Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified in Fundamentals of Large Language Models (LLMs)**")
    st.markdown("- **Have expertise in EDA, ML, Reinforcement Learning, ANN, CNN, CV, RNN, NLP, LLMs.**")
    st.markdown("[💼Visit Rayyan's LinkedIn Profile](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

with st.sidebar.expander("📌 Model Descriptions"):
    st.markdown("🔹 **SimpleRNN**: A basic Recurrent Neural Network architecture. Suitable for modeling simple sequential patterns, but may struggle with long-term dependencies in time series data.")
    st.markdown("🔹 **LSTM (Long Short-Term Memory)**: An advanced RNN variant designed to remember information over long sequences. Effective for time series forecasting due to its ability to capture temporal patterns and trends.")
    st.markdown("🔹 **GRU (Gated Recurrent Unit)**: A lightweight alternative to LSTM with fewer parameters. Offers similar performance with faster training, making it ideal for quick and efficient sequence modeling.")

with st.sidebar.expander("📦 Real-World Applications"):
    st.markdown("🌾 **Agriculture**: Help farmers plan irrigation and crop protection based on upcoming temperatures.")
    st.markdown("🏙️ **Urban Planning**: Assist cities in managing heatwaves, energy usage, and public safety.")
    st.markdown("🏠 **Smart Homes**: Enable HVAC systems to optimize heating and cooling automatically.")
    st.markdown("✈️ **Aviation & Transport**: Support flight scheduling and road maintenance by forecasting extreme weather.")
    st.markdown("⚡ **Energy Sector**: Improve power demand forecasting and grid load balancing during temperature swings.")

with st.sidebar.expander("📈 Other Time Series Use Cases"):
    st.markdown("💹 **Stock Market Prediction**: Forecast stock prices or trends based on historical market data.")
    st.markdown("🛒 **Sales Forecasting**: Predict future sales for businesses to manage inventory and marketing.")
    st.markdown("🚦 **Traffic Flow Prediction**: Model traffic congestion trends to optimize city planning or navigation.")
    st.markdown("🩺 **Healthcare Monitoring**: Predict patient vitals or detect anomalies from wearable sensor data.")
    st.markdown("🎧 **Speech & Audio Analysis**: Model audio signals for voice recognition, music analysis, and more.")
