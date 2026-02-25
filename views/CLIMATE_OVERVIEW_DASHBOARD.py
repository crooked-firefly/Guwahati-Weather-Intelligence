import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')

import utils as ut

def show(df):
    st.title("🌤️ Climate Overview Dashboard")

    st.subheader("🌡️Daily Mean Temperatures Overview")

# --- KPIs ---
    avg_tempmax = df['tempmax'].mean()
    avg_tempmin = df['tempmin'].mean()
    avg_temp = df['temp'].mean()
    avg_feelslikemax = df['feelslikemax'].mean()
    avg_feelslikemin = df['feelslikemin'].mean()
    avg_feelslike = df['feelslike'].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ut.kpi_card(c1, "🥵 Temp Max", f"{avg_tempmax:.1f}", suffix="°C")
    ut.kpi_card(c2, "❄️ Temp Min", f"{avg_tempmin:.1f}", suffix="°C")
    ut.kpi_card(c3, "🌡️ Temp", f"{avg_temp:.1f}", suffix="°C")
    ut.kpi_card(c4, "🔥 FeelsLike Max", f"{avg_feelslikemax:.1f}", suffix="°C")
    ut.kpi_card(c5, "💧 FeelsLike Min", f"{avg_feelslikemin:.1f}", suffix="°C")
    ut.kpi_card(c6, "🌞 FeelsLike", f"{avg_feelslike:.1f}", suffix="°C")



    st.subheader("🌍 General Climatic Attributes")

# --- KPIs ---
    annual_rain = df.groupby('year')['precip'].sum().reset_index()
    avg_annual_rainfall = annual_rain['precip'].mean()

    avg_daily_windspeed = df['windspeed'].mean()  # in km/h
    avg_daily_humidity = df['humidity'].mean()  # in %
    avg_daily_pressure = df['sealevelpressure'].mean()  # in hPa
    avg_visibility = df['visibility'].mean()  # in km
    avg_temp_range = df['temprature_range'].mean()  # in °C

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ut.kpi_card(c1, "🌧️ Avg Annual Rainfall", f"{avg_annual_rainfall/1000:.2f}", suffix="K mm")
    ut.kpi_card(c2, "💨 Wind Speed/ Day", f"{avg_daily_windspeed:.1f}", suffix=" km/h")
    ut.kpi_card(c3, "💦 Avg Daily Humidity", f"{avg_daily_humidity:.1f}", suffix=" %")
    ut.kpi_card(c4, "📉 Avg SLP/ Day", f"{avg_daily_pressure:.1f}", suffix=" hPa")
    ut.kpi_card(c5, "👁️ Avg Visibility", f"{avg_visibility:.1f}", suffix=" km")
    ut.kpi_card(c6, "🌡️ Avg Temp Range", f"{avg_temp_range:.1f}", suffix=" °C")
    




    st.subheader("Visuals")
    #########row 1
    df['month'] = df['month'].astype(int)

# --- Chart 1: Mean Rainfall and Rain Hours ---
    monthly_stats = (
        df[df['precip'] > 0]
        .groupby('month')
        .agg(mean_precip=('precip', 'mean'),
             mean_precip_hours=('precip_hours', 'mean'))
        .reset_index()
    )
    
    # Fill missing months
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_stats = pd.merge(all_months, monthly_stats, on='month', how='left').fillna(0)
    
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    bars = ax1.bar(monthly_stats['month'], monthly_stats['mean_precip'], color='skyblue', label='Mean Precip (mm)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Mean Precip (mm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(range(1, 13))
    ax1.set_title("Mean Rainfall (mm) and Rain Hours on Rainy Days")
    
    # Add data labels to bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # Line chart for Mean Precip Hours
    ax2 = ax1.twinx()
    line = ax2.plot(monthly_stats['month'], monthly_stats['mean_precip_hours'],
                    color='orange', marker='s', label='Mean Precip Hours')
    ax2.set_ylabel('Mean Precip Hours', color='Orange')
    ax2.tick_params(axis='y', labelcolor='Orange')
    
    # Add data labels to line points
    for x, y in zip(monthly_stats['month'], monthly_stats['mean_precip_hours']):
        ax2.annotate(f'{y:.1f}', xy=(x, y), xytext=(0, -10), textcoords="offset points",
                     ha='center', fontsize=8, color='Orange')
    
    fig1.tight_layout()
    
    # --- Chart 2: Rainy vs Dry Days Distribution ---
    rainy_dry = df.groupby(['month', 'is_rainy_day']).size().unstack(fill_value=0)
    rainy_dry_percent = rainy_dry.div(rainy_dry.sum(axis=1), axis=0) * 100
    rainy_dry_percent = rainy_dry_percent.rename(columns={True: 'Rainy Day', False: 'Dry Day'}).reset_index()
    
    fig2, ax = plt.subplots(figsize=(5, 4))
    dry = ax.bar(rainy_dry_percent['month'], rainy_dry_percent['Dry Day'], label='Dry Day', color='wheat')
    rainy = ax.bar(rainy_dry_percent['month'], rainy_dry_percent['Rainy Day'],
                   bottom=rainy_dry_percent['Dry Day'], label='Rainy Day', color='lightblue')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Days in %')
    ax.set_title("Rainy and Dry Days Distribution")
    ax.set_xticks(range(1, 13))
    ax.legend()
    
    # Add data labels
    for i in range(len(rainy_dry_percent)):
        dry_val = rainy_dry_percent['Dry Day'][i]
        rain_val = rainy_dry_percent['Rainy Day'][i]
        ax.text(i + 1, dry_val / 2, f'{dry_val:.1f}%', ha='center', va='center', fontsize=8)
        ax.text(i + 1, dry_val + rain_val / 2, f'{rain_val:.1f}%', ha='center', va='center', fontsize=8)
    
    fig2.tight_layout()
    
    # --- Display both charts side by side ---
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)






#####row 2
    
   



# Ensure formatting
    df['month'] = df['month'].astype(int)
    df['seasons'] = df['seasons'].str.title()
    df['uv_category'] = df['uv_category'].str.title()
    
    # --- Chart 1: Mean Daily Humidity Levels ---
    monthly_humidity = df.groupby('month')['humidity'].mean().reset_index()
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_humidity = pd.merge(all_months, monthly_humidity, on='month', how='left').fillna(0)
    
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    bars = ax1.bar(monthly_humidity['month'], monthly_humidity['humidity'], color='lightblue')
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Humidity (%)")
    ax1.set_title("Mean Daily Humidity Levels")
    ax1.set_xticks(range(1, 13))
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    fig1.tight_layout()
    
    # --- Chart 2: UV Index Category Distribution by Season ---
    uv_dist = df.groupby(['seasons', 'uv_category']).size().unstack(fill_value=0)
    uv_percent = uv_dist.div(uv_dist.sum(axis=1), axis=0) * 100
    uv_percent = uv_percent[['Low', 'Medium', 'High']]  # Ensure order
    uv_percent = uv_percent.reset_index()
    
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bottom = [0] * len(uv_percent)
    colors = {'Low': 'teal', 'Medium': 'wheat', 'High': 'red'}
    
    for category in ['Low', 'Medium', 'High']:
        values = uv_percent[category]
        bars = ax2.bar(uv_percent['seasons'], values, bottom=bottom, label=category, color=colors[category])
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 1:
                ax2.text(i, btm + val / 2, f'{val:.1f}%', ha='center', va='center', fontsize=8)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax2.set_ylabel("Days in %")
    ax2.set_xlabel("Seasons")
    ax2.set_title("☀️ Days Distribution by UV Index Categories")
    ax2.legend(title="UV Index")
    fig2.tight_layout()
    
    # --- Display both charts side by side ---
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)















####row 3
   # Ensure month is numeric
    df['month'] = df['month'].astype(int)
    
    # Group by month and calculate mean temperatures
    monthly_temp = df.groupby('month').agg(
        tempmax=('tempmax', 'mean'),
        temp=('temp', 'mean'),
        tempmin=('tempmin', 'mean')
    ).reset_index()
    
    # Fill missing months
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_temp = pd.merge(all_months, monthly_temp, on='month', how='left').fillna(0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Max Temp (orange diamonds)
    ax.plot(monthly_temp['month'], monthly_temp['tempmax'], color='orange', marker='D', label='Max Temp')
    
    # Mean Temp (blue squares)
    ax.plot(monthly_temp['month'], monthly_temp['temp'], color='blue', marker='s', label='Mean Temp')
    
    # Min Temp (white diamonds)
    ax.plot(monthly_temp['month'], monthly_temp['tempmin'], color='black', marker='D', label='Min Temp', linestyle='--')
    
    # Add data labels
    for i in range(12):
        ax.text(monthly_temp['month'][i], monthly_temp['tempmax'][i] + 0.5, f"{monthly_temp['tempmax'][i]:.1f}", ha='center',     fontsize=8, color='Red')
        ax.text(monthly_temp['month'][i], monthly_temp['temp'][i] + 0.5, f"{monthly_temp['temp'][i]:.1f}", ha='center', fontsize=8,     color='blue')
        ax.text(monthly_temp['month'][i], monthly_temp['tempmin'][i] - 1.5, f"{monthly_temp['tempmin'][i]:.1f}", ha='center',     fontsize=8, color='black')
    
    ax.set_title("Mean Daily Max., Min. and Avg. Temperatures (°C)")
    ax.set_xlabel("Months")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.set_xticks(range(1, 13))
    ax.set_ylim(5, 40)
    ax.legend()
    fig.tight_layout()
    
    st.pyplot(fig)





###########row 4
    df['month'] = df['month'].astype(int)
    
    # Group by month and calculate average cloud cover
    monthly_cloud = df.groupby('month')['cloudcover'].mean().reset_index()
    
    # Fill missing months if any
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_cloud = pd.merge(all_months, monthly_cloud, on='month', how='left').fillna(0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(monthly_cloud['month'], monthly_cloud['cloudcover'], color='skyblue')
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg. Cloud Cover (%)")
    ax.set_title("Average Monthly Cloud Cover")
    ax.set_xticks(range(1, 13))
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    fig.tight_layout()
    st.pyplot(fig)

























