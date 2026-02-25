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
    st.title("🏙️ URBAN INFRASTRUCTURE PLANNING DASHBOARD")
    st.subheader("🏙️ Urban Infrastructure Stress Indicators")

# Ensure datetime and year columns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    
    # 1. Avg Cooling Demand Index per Day per Year
    def cooling_proxy(group):
        valid = group[['tempmax', 'humidity']].dropna()
        return (valid['tempmax'] * (valid['humidity'] / 100)).mean()
    
    avg_cooling_proxy = df.groupby('year').apply(cooling_proxy).mean()
    
    # 2. Mean Monthly Rainfall (mm)
    yearly_rain = df.groupby('year')['precip'].sum()

    monthly_rain_per_year = yearly_rain / 12
    avg_monthly_rainfall = monthly_rain_per_year.mean()

    
    # 3. Avg Water Availability Proxy per Day per Year
    def water_proxy(group):
        valid = group[['precip', 'humidity']].dropna()
        if valid.empty:
            return None
        max_precip = valid['precip'].max()
        return ((valid['precip'] / max_precip) + (valid['humidity'] / 100)).mean()
    
    avg_water_availability_proxy = df.groupby('year').apply(water_proxy).mean()
    
    # 4. Avg Days with Rainfall > 50mm per Year
    rain_50mm_per_year = df[df['precip'] > 50].groupby('year').size()
    avg_days_rainfall_above_50 = rain_50mm_per_year.mean()
    
    # 5. Avg SLP Alerts per Year (Sea Level Pressure < 1000 hPa)
    slp_alerts_per_year = df[df['sealevelpressure'] < 1000].groupby('year').size()
    avg_slp_alerts = slp_alerts_per_year.mean()
    
    # 6. Avg Visibility Alerts per Year (Visibility < 2 km)
    visibility_alerts_per_year = df[df['visibility'] < 2].groupby('year').size()
    avg_visibility_alerts = visibility_alerts_per_year.mean()
    
    # 7. Avg Heatwave Days per Year (TempMax ≥ 40°C)
    heatwave_extremes_per_year = df[df['tempmax'] >= 40].groupby('year').size()
    avg_heatwave_extremes = heatwave_extremes_per_year.mean()
    
    # 8. Avg Wind Speed Extremes per Year (WindSpeed ≥ 70 km/h)
    wind_extremes_per_year = df[df['windspeed'] >= 70].groupby('year').size()
    avg_wind_extremes = wind_extremes_per_year.mean()
    
    # --- Display KPIs in 2 rows of 4 ---
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    ut.kpi_card(r1c1, "❄️ Cooling Demand Index/ Day/ Year", f"{avg_cooling_proxy:.2f}")
    ut.kpi_card(r1c2, "🌧️ Mean Monthly Rainfall", f"{avg_monthly_rainfall:.1f}", suffix=" mm")
    ut.kpi_card(r1c3, "💧 Water Availability Proxy/ Day/ Year", f"{avg_water_availability_proxy:.2f}")
    ut.kpi_card(r1c4, "🌊 Rainfall > 50mm/Year", f"{avg_days_rainfall_above_50:.2f}", suffix=" days")
    
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    ut.kpi_card(r2c1, "🌡️ SLP (< 1000 hPa)/ Year", f"{avg_slp_alerts:.2f}", suffix=" days")
    ut.kpi_card(r2c2, "🌫️ Visibility (< 2 km)/ Year", f"{avg_visibility_alerts:.2f}", suffix=" days")
    ut.kpi_card(r2c3, "🔥 Heatwave Days (≥40°C)/ Year", f"{avg_heatwave_extremes:.2f}", suffix=" days")
    ut.kpi_card(r2c4, "💨 Wind (≥ 70 km/h)/ Year", f"{avg_wind_extremes:.2f}", suffix=" days")




###row 1

    # --- Row: Mean Monthly Total Rainfall Distribution ---
    st.markdown("## 🌧️ Mean Monthly Total Rainfall Distribution")
    
    # Step 1: Sum rainfall per month per year
    monthly_rainfall = df.groupby(['year', 'month'])['precip'].sum().reset_index(name='total_rainfall')
    
    # Step 2: Average across years
    mean_monthly_rainfall = monthly_rainfall.groupby('month')['total_rainfall'].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(mean_monthly_rainfall['month'], mean_monthly_rainfall['total_rainfall'],
                  color='#3498db', edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Avg Total Rainfall (mm)", fontsize=11)
    ax.set_title("Mean Monthly Total Rainfall Distribution", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)],  fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    

####row 2
    # --- Row: Mean Monthly Max Temp and Cooling Demand Proxy ---
    st.markdown("## 🌡️ Mean Monthly Max Temperature & Cooling Demand Proxy")
    
    # Step 1: Calculate cooling demand proxy
    df['cooling_demand_proxy'] = df['tempmax'] * (df['humidity'] / 100)
    
    # Step 2: Group by year and month, then average
    monthly_stats = df.groupby(['year', 'month'])[['tempmax', 'cooling_demand_proxy']].mean().reset_index()
    
    # Step 3: Average across years
    mean_monthly = monthly_stats.groupby('month')[['tempmax', 'cooling_demand_proxy']].mean().reset_index()
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Bar chart for mean tempmax
    bars = ax1.bar(mean_monthly['month'], mean_monthly['tempmax'], color='#f39c12', edgecolor='black', label='Mean Temp Max')
    ax1.set_xlabel("Month", fontsize=11)
    ax1.set_ylabel("Mean Temp Max (°C)", color='#f39c12', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#f39c12')
    
    # Add value labels to bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Line chart for cooling demand proxy
    ax2 = ax1.twinx()
    line = ax2.plot(mean_monthly['month'], mean_monthly['cooling_demand_proxy'], color='#2980b9', marker='o', linewidth=2,     label='Cooling Demand Proxy')
    ax2.set_ylabel("Cooling Demand Proxy", color='#2980b9', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#2980b9')
    
    # Add value labels to line chart
    for x, y in zip(mean_monthly['month'], mean_monthly['cooling_demand_proxy']):
        ax2.text(x, y - 0.2, f'{y:.1f}', ha='center', va='top', fontsize=9, color='#2980b9')

    
    # Title and x-axis
    ax1.set_title("Mean Monthly Max Temp & Cooling Demand Proxy", fontsize=13)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    
    # Legends
    lines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)



######row 3

    # --- Row: Total Monthly Days by Wind Speed & Gust Category (Filtered) ---
    st.markdown("## 🌬️ Total Monthly Days (Extreme Wind & Gust Categories Only)")
    
    col1, col2 = st.columns(2)
    
    # --- Wind Speed Category ---
    with col1:
        
    
        # Filter for extreme wind speed categories
        extreme_speed = df[df['WindSpeedCategory'].isin(['Very Strong', 'Severe / Damaging'])]
        monthly_speed = extreme_speed.groupby(['month', 'WindSpeedCategory']).size().unstack(fill_value=0)
    
        speed_order = ['Very Strong', 'Severe / Damaging']
        monthly_speed = monthly_speed[[cat for cat in speed_order if cat in monthly_speed.columns]]
        monthly_speed = monthly_speed.reset_index()
    
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        bottom = [0] * len(monthly_speed)
        colors_speed = {
            'Very Strong': "#7fa9c5",
            'Severe / Damaging': "#e2918a"
        }
    
        for category in monthly_speed.columns[1:]:
            values = monthly_speed[category]
            ax1.bar(monthly_speed['month'], values, bottom=bottom, label=category,
                    color=colors_speed.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
            for i, (val, btm) in enumerate(zip(values, bottom)):
                if val > 0:
                    ax1.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=9)
            bottom = [btm + val for btm, val in zip(bottom, values)]
    
        ax1.set_title("Wind Speed (Extreme Categories)")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Total Days")
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels([str(m) for m in range(1, 13)])
        ax1.legend( loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)


        fig1.tight_layout()
        st.pyplot(fig1)
    
    # --- Wind Gust Category ---
    with col2:
        
    
        df_gust = df[df['year'] >= 2015]
        extreme_gust = df_gust[df_gust['WindGustCategory'].isin(['Very Strong Gust', 'Extreme / Damaging Gust'])]
        monthly_gust = extreme_gust.groupby(['month', 'WindGustCategory']).size().unstack(fill_value=0)
    
        gust_order = ['Very Strong Gust', 'Extreme / Damaging Gust']
        monthly_gust = monthly_gust[[cat for cat in gust_order if cat in monthly_gust.columns]]
        monthly_gust = monthly_gust.reset_index()
    
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bottom = [0] * len(monthly_gust)
        colors_gust = {
            'Very Strong Gust': "#81a8ce",
            'Extreme / Damaging Gust': "#ec5c4c"
        }
    
        for category in monthly_gust.columns[1:]:
            values = monthly_gust[category]
            ax2.bar(monthly_gust['month'], values, bottom=bottom, label=category,
                    color=colors_gust.get(category, '#bbb'), edgecolor='black', linewidth=0.5)
            for i, (val, btm) in enumerate(zip(values, bottom)):
                if val > 0:
                    ax2.text(i + 1, btm + val / 2, f'{val:.0f}', ha='center', va='center', fontsize=9)
            bottom = [btm + val for btm, val in zip(bottom, values)]
    
        ax2.set_title("Wind Gust (Extreme Categories)")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Total Days")
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels([str(m) for m in range(1, 13)])
        ax2.legend( loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)

        fig2.tight_layout()
        st.pyplot(fig2)

    


#####row 4
    # --- Row: Mean Monthly Humidity & Visibility ---
    st.markdown("## 💧 Mean Monthly Humidity & Visibility")
    
    col1, col2 = st.columns(2)
    
    # --- Mean Monthly Humidity ---
    with col1:
        st.markdown("### 💧 Avg Monthly Humidity (%)")
    
        # Step 1: Group by year and month, then average humidity
        monthly_humidity = df.groupby(['year', 'month'])['humidity'].mean().reset_index()
    
        # Step 2: Average across years
        mean_humidity = monthly_humidity.groupby('month')['humidity'].mean().reset_index()
    
        # Plot
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        bars = ax1.bar(mean_humidity['month'], mean_humidity['humidity'],
                       color='#76c7c0', edgecolor='black')
    
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Avg Humidity (%)")
        ax1.set_title("Mean Monthly Humidity")
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels([str(m) for m in range(1, 13)])
        fig1.tight_layout()
        st.pyplot(fig1)
    
    # --- Mean Monthly Visibility ---
    with col2:
        st.markdown("### 👁️ Avg Monthly Visibility (km)")
    
        # Step 1: Group by year and month, then average visibility
        monthly_vis = df.groupby(['year', 'month'])['visibility'].mean().reset_index()
    
        # Step 2: Average across years
        mean_vis = monthly_vis.groupby('month')['visibility'].mean().reset_index()
    
        # Plot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(mean_vis['month'], mean_vis['visibility'], color='#5dade2', marker='o', linewidth=2, label='Visibility')
    
        for x, y in zip(mean_vis['month'], mean_vis['visibility']):
            ax2.text(x, y-0.1, f'{y:.1f}', ha='center', va='top', fontsize=9, color='#5dade2')
    
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Avg Visibility (km)")
        ax2.set_title("Mean Monthly Visibility")
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels([str(m) for m in range(1, 13)])
        fig2.tight_layout()
        st.pyplot(fig2)



    ##row 5

    # --- Row: Mean Monthly Max, Min, and Avg Temperatures ---
    st.markdown("## 🌡️ Mean Monthly Max, Min, and Avg Temperatures")
    
    # Step 1: Group by year and month, then average each temperature metric
    monthly_temps = df.groupby(['year', 'month'])[['tempmax', 'tempmin', 'temp']].mean().reset_index()
    
    # Step 2: Average across years
    mean_monthly_temps = monthly_temps.groupby('month')[['tempmax', 'tempmin', 'temp']].mean().reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot each line
    ax.plot(mean_monthly_temps['month'], mean_monthly_temps['tempmax'], label='Max Temp', color='#e74c3c', marker='o')
    ax.plot(mean_monthly_temps['month'], mean_monthly_temps['tempmin'], label='Min Temp', color='#3498db', marker='o')
    ax.plot(mean_monthly_temps['month'], mean_monthly_temps['temp'], label='Avg Temp', color="#848b87", marker='o')
    
    # Add value labels below each point
    for col, color in zip(['tempmax', 'tempmin', 'temp'], ['#e74c3c', '#3498db', "#79807c"]):
        for x, y in zip(mean_monthly_temps['month'], mean_monthly_temps[col]):
            ax.text(x, y - 0.5, f'{y:.1f}', ha='center', va='top', fontsize=9, color=color)
    
    # Axis and title
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Mean Monthly Max, Min, and Avg Temperatures", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)
