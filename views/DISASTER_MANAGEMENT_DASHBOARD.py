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
    st.title("🚨 DISASTER MANAGEMENT DASHBOARD")
    st.subheader("⚠️ Disaster Risk Indicators")

# Ensure datetime and year columns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['year'].astype(int)

# --- Calculations ---
    avg_rain_long_duration_days = df[df['precip_hours'] > 8].groupby('year').size().mean()
    avg_days_rainfall_above_50 = df[df['precip'] > 50].groupby('year').size().mean()
    avg_slp_alerts = df[df['sealevelpressure'] < 1000].groupby('year').size().mean()
    avg_wind_speed_extremes = df[df['windspeed'] > 30].groupby('year').size().mean()
    avg_heatwave_days = df[df['tempmax'] > 33].groupby('year').size().mean()
    avg_coldwave_days = df[df['tempmin'] <= 4].groupby('year').size().mean()
    avg_visibility_alerts = df[df['visibility'] < 2].groupby('year').size().mean()
    avg_uv_index = df['uvindex'].mean()
    
    # --- Display KPIs in 2 rows of 4 ---
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    ut.kpi_card(r1c1, "🌧️ Rain Hours > 8/day", f"{avg_rain_long_duration_days:.1f}", suffix=" days")
    ut.kpi_card(r1c2, "🌊 Rainfall > 50mm/day", f"{avg_days_rainfall_above_50:.1f}", suffix=" days")
    ut.kpi_card(r1c3, "🌡️ SLP < 1000 hPa", f"{avg_slp_alerts:.1f}", suffix=" days")
    ut.kpi_card(r1c4, "💨 Wind > 30 km/h", f"{avg_wind_speed_extremes:.1f}", suffix=" days")
    
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    ut.kpi_card(r2c1, "🔥 Heatwave Days (>33°C)", f"{avg_heatwave_days:.1f}", suffix=" days")
    ut.kpi_card(r2c2, "❄️ Coldwave Days (≤4°C)", f"{avg_coldwave_days:.1f}", suffix=" days")
    ut.kpi_card(r2c3, "🌫️ Visibility < 2 km", f"{avg_visibility_alerts:.1f}", suffix=" days")
    ut.kpi_card(r2c4, "☀️ Avg UV Index", f"{avg_uv_index:.2f}")

    

    ### row 1
    
# Group and calculate mean total rainfall per month
    monthly_totals = df.groupby(['year', 'month'])['precip'].sum().reset_index()
    
    # Step 2: Average those monthly totals across years
    mean_monthly_rainfall = monthly_totals.groupby('month')['precip'].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_monthly_rainfall['month'], mean_monthly_rainfall['precip'],
            marker='o', color='#3498db', linewidth=2)
    
    # Add value labels
    for i, row in mean_monthly_rainfall.iterrows():
        ax.text(row['month'], row['precip'] + 1, f"{row['precip']:.1f}", ha='center', fontsize=10)
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Monthly Rainfall (mm)", fontsize=11)
    ax.set_title("Mean Monthly Total Rainfall (in mm)", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)],   fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    st.pyplot(fig)





##row 2
    st.markdown("## 🌬️ Mean Monthly Wind Speed Category Distribution")
    
    # Step 1: Count days per month per year per category
    monthly_counts = df.groupby(['year', 'month', 'WindSpeedCategory']).size().reset_index(name='count')
    
    # Step 2: Average across years
    mean_monthly_counts = monthly_counts.groupby(['month', 'WindSpeedCategory'])['count'].mean().unstack(fill_value=0)
    
    # Ensure consistent category order
    category_order = ['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong', 'Severe / Damaging']
    mean_monthly_counts = mean_monthly_counts[[cat for cat in category_order if cat in mean_monthly_counts.columns]]
    mean_monthly_counts = mean_monthly_counts.reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(mean_monthly_counts)
    colors_speed = {
        'Calm': '#d5dbdb',
        'Light': '#aed6f1',
        'Moderate': '#5dade2',
        'Strong': '#2874a6',
        'Very Strong': '#1f618d',
        'Severe / Damaging': '#922b21'
    }
    
    for category in mean_monthly_counts.columns[1:]:
        values = mean_monthly_counts[category]
        ax.bar(mean_monthly_counts['month'], values, bottom=bottom, label=category,
               color=colors_speed.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=9)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Days per Month", fontsize=11)
    ax.set_title("Mean Monthly Wind Speed Category Distribution", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)],  fontsize=9)
    ax.legend(title="Wind Speed", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=True)
    fig.tight_layout()
    st.pyplot(fig)
    
    



    ##row 3
    st.markdown("## 💨 Mean Monthly Wind Gust Category Distribution (2015+)")

# Filter data from 2015 onward
    df_gust = df[df['year'] >= 2015]
    
    # Step 1: Count days per month per year per gust category
    monthly_gust_counts = df_gust.groupby(['year', 'month', 'WindGustCategory']).size().reset_index(name='count')
    
    # Step 2: Average across years
    mean_monthly_gust = monthly_gust_counts.groupby(['month', 'WindGustCategory'])['count'].mean().unstack(fill_value=0)
    
    # Ensure consistent category order
    gust_order = [
        'Weak Gust', 'Moderate Gust', 'Strong Gust',
        'Very Strong Gust', 'Severe / Damaging Gust', 'Extreme / Damaging Gust'
    ]
    mean_monthly_gust = mean_monthly_gust[[cat for cat in gust_order if cat in mean_monthly_gust.columns]]
    mean_monthly_gust = mean_monthly_gust.reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(mean_monthly_gust)
    colors_gust = {
        'Weak Gust': '#d5dbdb',
        'Moderate Gust': '#aed6f1',
        'Strong Gust': '#5dade2',
        'Very Strong Gust': '#2e86c1',
        'Severe / Damaging Gust': '#e67e22',
        'Extreme / Damaging Gust': '#c0392b'
    }
    
    for category in mean_monthly_gust.columns[1:]:
        values = mean_monthly_gust[category]
        ax.bar(mean_monthly_gust['month'], values, bottom=bottom, label=category,
               color=colors_gust.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=9)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Days per Month", fontsize=11)
    ax.set_title("Mean Monthly Wind Gust Category Distribution (2015+)", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(title="Wind Gust", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=True)
    fig.tight_layout()
    st.pyplot(fig)
    



#####row 4

# --- Row: Mean Monthly Rain Intensity Distribution ---
    st.markdown("## 🌧️ Mean Monthly Rain Intensity Distribution")
    
    # Step 1: Count days per month per year per rain intensity
    monthly_rain_counts = df.groupby(['year', 'month', 'rain_intensity']).size().reset_index(name='count')
    
    # Step 2: Average across years
    mean_monthly_rain = monthly_rain_counts.groupby(['month', 'rain_intensity'])['count'].mean().unstack(fill_value=0)
    
    # Ensure consistent category order
    rain_order = ['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']
    mean_monthly_rain = mean_monthly_rain[[cat for cat in rain_order if cat in mean_monthly_rain.columns]]
    mean_monthly_rain = mean_monthly_rain.reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(mean_monthly_rain)
    colors_rain = {
        'No Rain': '#d5dbdb',
        'Light Rain': '#aed6f1',
        'Moderate Rain': '#5dade2',
        'Heavy Rain': '#2874a6'
    }
    
    for category in mean_monthly_rain.columns[1:]:
        values = mean_monthly_rain[category]
        ax.bar(mean_monthly_rain['month'], values, bottom=bottom, label=category,
               color=colors_rain.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=8)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Days per Month", fontsize=11)
    ax.set_title("Mean Monthly Rain Intensity Distribution", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)],  fontsize=9)
    ax.legend(title="Rain Intensity", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True)
    fig.tight_layout()
    st.pyplot(fig)



#### row 5
    st.markdown("## 🌧️ Mean Monthly Rain Hours Bin Distribution (Excluding 0)")
    
    # Filter out 0-hour bin
    df_rain = df[~df['precip_hours_bucket'].isin(['0'])]
    
    # Step 1: Count days per month per year per rain hours bin
    monthly_rain_hours = df_rain.groupby(['year', 'month', 'precip_hours_bucket']).size().reset_index(name='count')
    
    # Step 2: Average across years
    mean_monthly_rain_hours = monthly_rain_hours.groupby(['month', 'precip_hours_bucket'])['count'].mean().unstack(fill_value=0)
    
    # Ensure consistent bin order
    bin_order = ['0–3', '3–9', '9–15', '>15']
    mean_monthly_rain_hours = mean_monthly_rain_hours[[b for b in bin_order if b in mean_monthly_rain_hours.columns]]
    mean_monthly_rain_hours = mean_monthly_rain_hours.reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(mean_monthly_rain_hours)
    colors_bins = {
        '0–3': '#aed6f1',
        '3–9': '#5dade2',
        '9–15': '#2874a6',
        '>15': '#922b21'
    }
    
    for category in mean_monthly_rain_hours.columns[1:]:
        values = mean_monthly_rain_hours[category]
        ax.bar(mean_monthly_rain_hours['month'], values, bottom=bottom, label=category,
               color=colors_bins.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=8)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Days per Month", fontsize=11)
    ax.set_title("Mean Monthly Rain Hours Bin Distribution by Days (Excl. 0)", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)],  fontsize=9)
    ax.legend(title="Rain Hours Bin", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True)
    fig.tight_layout()
    st.pyplot(fig)
    



#####roww 6

    st.markdown("## ⚠️ Mean Monthly Weather Alert Type Distribution")
    
    # Filter out 'Normal' alert type
    df_alerts = df[df['WeatherAlertType'] != 'Normal']
    
    # Step 1: Count days per month per year per alert type
    monthly_alert_counts = df_alerts.groupby(['year', 'month', 'WeatherAlertType']).size().reset_index(name='count')
    
    # Step 2: Average across years
    mean_monthly_alerts = monthly_alert_counts.groupby(['month', 'WeatherAlertType'])['count'].mean().unstack(fill_value=0)
    
    # Ensure consistent alert type order
    alert_order = ['Coldwave', 'Heatwave', 'Heavy Rainfall']
    mean_monthly_alerts = mean_monthly_alerts[[cat for cat in alert_order if cat in mean_monthly_alerts.columns]]
    mean_monthly_alerts = mean_monthly_alerts.reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(mean_monthly_alerts)
    colors_alerts = {
        'Coldwave': 'grey',         # White
        'Heatwave': '#e74c3c',         # Red
        'Heavy Rainfall': '#2874a6'    # Blue
    }
    
    for category in mean_monthly_alerts.columns[1:]:
        values = mean_monthly_alerts[category]
        ax.bar(mean_monthly_alerts['month'], values, bottom=bottom, label=category,
               color=colors_alerts.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 0:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=10)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Days per Month", fontsize=11)
    ax.set_title("Mean Monthly Weather Alert Type Distribution", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(title="Alert Type", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    fig.tight_layout()
    st.pyplot(fig)

    
    
    