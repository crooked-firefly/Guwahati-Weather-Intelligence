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
    st.title("🌾 AGRICULTURE PLANNING DASHBOARD")
    st.subheader("🌾 Agriculture Climate Indicators")

# Ensure datetime and year columns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year

# 1. Avg Rainy Days Count per Year
    rainy_days_per_year = df[df['precip'] > 0].groupby('year').size()
    avg_rainy_days_count = rainy_days_per_year.mean()

# 2. Avg Rainy Days % per Year
    rainy_days_pct_per_year = (
    df.groupby('year')
    .apply(lambda x: (x['precip'] > 0).sum() / len(x) * 100)
)
    avg_rainy_days_pct = rainy_days_pct_per_year.mean()

# 3. Avg Max Temp
    avg_max_temp = df['tempmax'].mean()

# 4. Avg Min Temp
    avg_min_temp = df['tempmin'].mean()

# 5. Avg Soil Moisture Proxy = (humidity * dew) / 100
    df['soil_moisture_proxy'] = (df['humidity'] * df['dew']) / 100
    avg_soil_moisture_proxy = df['soil_moisture_proxy'].mean()

# 6. Avg Monsoon Intensity Index per Year
    monsoon_rainy = df[(df['seasons'] == 'Monsoon') & (df['precip'] > 0)]
    monsoon_stats = monsoon_rainy.groupby('year').agg(
    monsoon_rain=('precip', 'sum'),
    rainy_days=('precip', 'count')
).reset_index()
    monsoon_stats['intensity'] = monsoon_stats['monsoon_rain'] / monsoon_stats['rainy_days']
    avg_monsoon_intensity = monsoon_stats['intensity'].mean()

# 7. Avg Rainfall per Rainy Day
    rainy_days = df[df['precip'] > 0]
    avg_rain_per_rainy_day = df['precip'].sum() / len(rainy_days) if len(rainy_days) > 0 else None

# 8. Avg Monsoon Rainfall Share per Year
    def monsoon_share(group):
        total_rain = group['precip'].sum()
        monsoon_rain = group[group['seasons'] == 'Monsoon']['precip'].sum()
        return (monsoon_rain / total_rain) * 100 if total_rain > 0 else None

    monsoon_rainfall_share = df.groupby('year').apply(monsoon_share)
    avg_monsoon_rainfall_share = monsoon_rainfall_share.mean()

# --- Display KPIs in 2 rows of 4 ---
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    ut.kpi_card(r1c1, "🌧️ Avg Rainy Days Count", f"{avg_rainy_days_count:.1f}", suffix=" days")
    ut.kpi_card(r1c2, "☔ Avg Rainy Days %", f"{avg_rainy_days_pct:.1f}", suffix=" %")
    ut.kpi_card(r1c3, "🌡️ Avg Max Temp", f"{avg_max_temp:.1f}", suffix=" °C")
    ut.kpi_card(r1c4, "❄️ Avg Min Temp", f"{avg_min_temp:.1f}", suffix=" °C")
    
    # --- Row 2: Soil & Rainfall KPIs ---
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    ut.kpi_card(r2c1, "🌱 Soil Moisture Proxy", f"{avg_soil_moisture_proxy:.2f}")
    ut.kpi_card(r2c2, "🌧️ Monsoon Intensity", f"{avg_monsoon_intensity:.1f}", suffix=" mm")
    ut.kpi_card(r2c3, "💦 Rainfall per Rainy Day", f"{avg_rain_per_rainy_day:.1f}", suffix=" mm")
    ut.kpi_card(r2c4, "🌈 Monsoon Rainfall Share", f"{avg_monsoon_rainfall_share:.1f}", suffix=" %")




###first row
    # df['years_bucket'] = df['years_bucket'].astype(str)
    df['seasons'] = df['seasons'].str.title()
    df['temp_stress_category'] = df['temp_stress_category'].str.title()
    df['month'] = df['month'].astype(int)
    
    # --- First Row: Rainfall Distribution by Season & Years Bucket ---
    st.markdown("### 🌧️ Rainfall Distribution by Season")
    
# Group by seasons only
    rain_dist = df.groupby('seasons')['precip'].sum().reset_index()
    rain_dist.columns = ['Season', 'Total Rainfall (mm)']
    
    # Sort by rainfall amount
    season_order = ['Monsoon', 'Summer', 'Post-Monsoon', 'Winter']
    rain_dist['Season'] = pd.Categorical(rain_dist['Season'], categories=season_order, ordered=True)
    rain_dist = rain_dist.sort_values('Season').reset_index(drop=True)
    
    # Colors
    colors_rain = {
        'Monsoon': '#3498db',
        'Post-Monsoon': '#95a5a6',
        'Summer': '#e59866',
        'Winter': '#d5dbdb'
    }
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    bars = ax1.bar(
        rain_dist['Season'],
        rain_dist['Total Rainfall (mm)'],
        color=[colors_rain[s] for s in rain_dist['Season']],
        width=0.5,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 50,
            f'{height:,.0f} mm',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add percentage labels inside bars
    total = rain_dist['Total Rainfall (mm)'].sum()
    for bar, val in zip(bars, rain_dist['Total Rainfall (mm)']):
        pct = (val / total) * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f'{pct:.1f}%',
            ha='center',
            va='center',
            fontsize=11,
            color='black',
            fontweight='bold'
        )
    
    ax1.set_ylabel("Total Rainfall (mm)", fontsize=11)
    ax1.set_xlabel("Season", fontsize=11)
    ax1.set_title("Total Rainfall Distribution by Season — Guwahati (1973-2024)", fontsize=13)
    ax1.set_ylim(0, rain_dist['Total Rainfall (mm)'].max() * 1.15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    fig1.tight_layout()
    st.pyplot(fig1)


    
#######row 2
# --- Second Row: Temp Stress Category by Month ---
    st.markdown("### 🌡️ Temp. Stress Category by Month")
    
    stress_dist = df.groupby(['month', 'temp_stress_category']).size().unstack(fill_value=0)
    stress_percent = stress_dist.div(stress_dist.sum(axis=1), axis=0) * 100
    stress_percent = stress_percent[['Below Optimal', 'Beyond Optimal', 'Cold Stress', 'Heat Stress', 'Optimal']]
    stress_percent = stress_percent.reset_index()
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(stress_percent)
    colors_stress = {
        'Below Optimal': '#5d6d7e',
        'Beyond Optimal': '#f4d03f',
         'Cold Stress': '#ffffff',
        'Heat Stress': '#e74c3c',
        'Optimal': '#58d68d'
    }
    
    for category in stress_percent.columns[1:]:
        values = stress_percent[category]
        ax2.bar(stress_percent['month'], values, bottom=bottom, label=category, color=colors_stress[category])
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 1:
                ax2.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=8)
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    ax2.set_ylabel("Days in %")
    ax2.set_xlabel("Month")
    ax2.set_title("Temp. Stress Category by Month and Days")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels([str(m) for m in range(1, 13)], ha='right')
    ax2.legend(title="Stress Category", loc='lower left', frameon=True)
    fig2.tight_layout()
    st.pyplot(fig2)
    



#######row 3
    col1, col2 = st.columns(2)
    
    # --- Chart 1: Mean Monthly Humidity ---
    with col1:
        st.markdown("### 💧 Mean Monthly Humidity (%)")
    
        humidity_monthly = df.groupby('month')['humidity'].mean().reset_index()
    
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        bars = ax3.bar(humidity_monthly['month'], humidity_monthly['humidity'], color='#76c7c0', edgecolor='black')
    
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.0f}%', ha='center', fontsize=8)
    
        ax3.set_xlabel("Month", fontsize=11)
        ax3.set_ylabel("Mean Humidity (%)", fontsize=11)
        ax3.set_title("Mean Monthly Relative Humidity", fontsize=13)
        ax3.set_xticks(humidity_monthly['month'])
        ax3.set_xticklabels(humidity_monthly['month'], ha='right', fontsize=9)
        ax3.set_ylim(0, 100)
        fig3.tight_layout()
        st.pyplot(fig3)
    
    # --- Chart 2: Mean Monthly Rainfall ---
    with col2:
        st.markdown("### 🌧️ Mean Monthly Rainfall (mm) Per Day")
    
        rainfall_monthly = df.groupby('month')['precip'].mean().reset_index()
    
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        bars = ax4.bar(rainfall_monthly['month'], rainfall_monthly['precip'], color='#5dade2', edgecolor='black')
    
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f'{height:.0f}', ha='center', fontsize=8)
    
        ax4.set_xlabel("Month", fontsize=11)
        ax4.set_ylabel("Mean Rainfall (mm) Per Day", fontsize=11)
        ax4.set_title("Mean Monthly Rainfall", fontsize=13)
        ax4.set_xticks(rainfall_monthly['month'])
        ax4.set_xticklabels(rainfall_monthly['month'], ha='right', fontsize=9)
        fig4.tight_layout()
        st.pyplot(fig4)




#######row 4
    st.markdown("### 🌧️ Rain Intensity Distribution by Month")
    

    

    rain_intensity = df.groupby(['month', 'rain_intensity']).size().unstack(fill_value=0)
    rain_percent = rain_intensity.div(rain_intensity.sum(axis=1), axis=0) * 100

    # Ensure consistent order
    rain_percent = rain_percent[['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain']]
    rain_percent = rain_percent.reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = [0] * len(rain_percent)
    colors_rain_intensity = {
        'No Rain': '#b2babb',
        'Light Rain': '#5dade2',
        'Moderate Rain': '#f7dc6f',
        'Heavy Rain': '#e74c3c'
    }

    for category in rain_percent.columns[1:]:
        values = rain_percent[category]
        ax.bar(
            rain_percent['month'],
            values,
            bottom=bottom,
            label=category,
            color=colors_rain_intensity[category],
            edgecolor='black',
            linewidth=0.5
        )
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 1:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=8)
        bottom = [btm + val for btm, val in zip(bottom, values)]

    ax.set_ylabel("Days Count in %", fontsize=11)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_title("Rain Intensity Distribution by Month", fontsize=13)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], ha='right', fontsize=9)
    ax.legend(title="Rain Intensity", loc='upper right', frameon=True)
    fig.tight_layout()
    st.pyplot(fig)



#########row 5

    st.markdown("## 📈 Temperature & Humidity-Rainfall Patterns")
    
    col1, col2 = st.columns(2)
    
    # --- Chart 1: Monthly Average Temperature Range ---
    with col1:
        st.markdown("### 🌡️ Monthly Avg Temperature Range")
    
        temp_range_monthly = df.groupby('month')['temprature_range'].mean().reset_index()
    
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.plot(temp_range_monthly['month'], temp_range_monthly['temprature_range'],
                 marker='o', color='#e67e22', linewidth=2)
    
        for i, row in temp_range_monthly.iterrows():
            ax5.text(row['month'], row['temprature_range'] + 0.1,
                     f"{row['temprature_range']:.1f}", ha='center', fontsize=10)
    
        ax5.set_xlabel("Month", fontsize=11)
        ax5.set_ylabel("Avg Temp Range (°C)", fontsize=11)
        ax5.set_title("Monthly Avg Temperature Range", fontsize=13)
        ax5.set_xticks(range(1, 13))
        ax5.set_xticklabels([str(m) for m in range(1, 13)],  ha='right', fontsize=10)
        ax5.grid(True, linestyle='--', alpha=0.5)
        fig5.tight_layout()
        st.pyplot(fig5)
    
    # --- Chart 2: Humidity vs Rainfall Scatter Plot ---
    with col2:
        st.markdown("### 💧 Humidity vs Rainfall")
    
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.scatter(df['humidity'], df['precip'], alpha=0.6,
                    color='#2e86c1', edgecolor='black', linewidth=0.5)
    
        ax6.set_xlabel("Humidity (%)", fontsize=11)
        ax6.set_ylabel("Rainfall (mm)", fontsize=11)
        ax6.set_title("Humidity vs Rainfall", fontsize=13)
        ax6.grid(True, linestyle='--', alpha=0.5)
        fig6.tight_layout()
        st.pyplot(fig6)








