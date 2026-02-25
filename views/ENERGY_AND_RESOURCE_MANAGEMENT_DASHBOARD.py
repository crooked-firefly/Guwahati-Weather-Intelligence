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
    st.title("⚡ ENERGY AND RESOURCE MANAGEMENT DASHBOARD")
    st.subheader("⚡ Energy & Resource Management Indicators")

# Ensure datetime and year columns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['year'].astype(int)
    
    # 1. Average Yearly Rainfall
    avg_yearly_rainfall = df.groupby('year')['precip'].sum().mean()
    
    # 2. Avg Heatwave Days per Year (TempMax > 33°C)
    heatwave_days_per_year = df[df['tempmax'] > 33].groupby('year').size()
    avg_heatwave_days = heatwave_days_per_year.mean()
    
    # 3. Avg Coldwave Days per Year (TempMin ≤ 10°C)
    coldwave_days_per_year = df[df['tempmin'] <= 10].groupby('year').size()
    avg_coldwave_days = coldwave_days_per_year.mean()
    
    # 4. Rain Consistency Index (RCI)
    def rci_per_year(group):
        monsoon_days = group[(group['seasons'] == 'Monsoon') & (~group['precip'].isna())]
        total_days = len(monsoon_days)
        dry_days = (monsoon_days['precip'] < 2.5).sum()
        return 1 - (dry_days / total_days) if total_days > 0 else None
    
    rci_by_year = df.groupby('year').apply(rci_per_year)
    avg_rci = rci_by_year.mean()
    
    # 5. Avg Solar Radiation
    avg_solar_radiation = df['solarradiation'].mean()
    
    # 6. Avg Solar Energy
    avg_solar_energy = df['solarenergy'].mean()
    
    # --- Display KPIs in 2 rows of 4 ---
    r1c1, r1c2, r1c3 = st.columns(3)
    ut.kpi_card(r1c1, "🌧️ Avg Yearly Rainfall", f"{avg_yearly_rainfall/1000:.2f}", suffix="K mm")
    ut.kpi_card(r1c2, "🔥 Heatwave Days (>33°C)/ Year", f"{avg_heatwave_days:.1f}", suffix=" days")
    ut.kpi_card(r1c3, "❄️ Coldwave Days (≤10°C)/ Year", f"{avg_coldwave_days:.1f}", suffix=" days")
    
    
    r2c1, r2c2, r2c3 = st.columns(3)
    ut.kpi_card(r2c1, "🔆 Avg Solar Radiation/Day", f"{avg_solar_radiation:.1f}", suffix=" W/m²")
    ut.kpi_card(r2c2, "🔋 Avg Solar Energy/Day", f"{avg_solar_energy:.2f}", suffix=" MJ/m²")
    ut.kpi_card(r2c3, "📈Mean Monsoon Rain Consistency Index", f"{avg_rci:.2f}")







####row 1

    st.markdown("## 🌧️ Mean Monthly Total Rainfall Distribution")
    
    # Step 1: Sum rainfall per month per year
    monthly_rainfall = df.groupby(['year', 'month'])['precip'].sum().reset_index(name='total_rainfall')
    
    # Step 2: Average across years (DAX-aligned logic: average of yearly monthly totals)
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





######row 3

    # --- Row: 100% Stacked Bar Chart for Heatwave and Coldwave Days ---
    st.markdown("## 🔥❄️ Monthly Distribution of Heatwave and Coldwave Days (100%)")
    
    # Step 1: Create flags
    df['heatwave_day'] = df['tempmax'] > 33
    df['coldwave_day'] = df['tempmin'] <= 10
    
    # Step 2: Count total, heatwave, and coldwave days per month
    monthly_total = df.groupby('month').size().reset_index(name='total_days')
    heatwave_counts = df[df['heatwave_day']].groupby('month').size().reset_index(name='heatwave_days')
    coldwave_counts = df[df['coldwave_day']].groupby('month').size().reset_index(name='coldwave_days')
    
    # Step 3: Merge and calculate neutral days
    merged = pd.merge(monthly_total, heatwave_counts, on='month', how='left').fillna(0)
    merged = pd.merge(merged, coldwave_counts, on='month', how='left').fillna(0)
    merged['neutral_days'] = merged['total_days'] - merged['heatwave_days'] - merged['coldwave_days']
    
    # Step 4: Calculate percentages
    merged['heatwave_pct'] = (merged['heatwave_days'] / merged['total_days']) * 100
    merged['coldwave_pct'] = (merged['coldwave_days'] / merged['total_days']) * 100
    merged['neutral_pct'] = (merged['neutral_days'] / merged['total_days']) * 100
    
    # Step 5: Plot 100% stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(merged)
    
    colors = {
        'coldwave_pct': '#3498db',
        'neutral_pct': '#95a5a6',
        'heatwave_pct': '#e74c3c'
    }
    
    for category in ['coldwave_pct', 'neutral_pct', 'heatwave_pct']:
        values = merged[category]
        ax.bar(merged['month'], values, bottom=bottom,
               label='Heatwave' if category == 'heatwave_pct' else
                     'Coldwave' if category == 'coldwave_pct' else
                     'Normal Days',
               color=colors[category], edgecolor='black')
        
        # Add value labels
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=8, color='Black')
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    # Final formatting
    ax.set_title("Monthly Distribution of Heatwave and Coldwave Days (100% Stacked)", fontsize=13)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Percentage of Days", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)
    








###row 2

    st.markdown("## 🌬️ Monthly Wind Speed Category Distribution (%)")
    
    # Step 1: Count occurrences of each wind speed category per month
    monthly_wind = df.groupby(['month', 'WindSpeedCategory']).size().reset_index(name='count')
    
    # Step 2: Pivot to get categories as columns
    wind_pivot = monthly_wind.pivot(index='month', columns='WindSpeedCategory', values='count').fillna(0)
    
    # Step 3: Convert counts to percentages
    wind_percent = wind_pivot.div(wind_pivot.sum(axis=1), axis=0) * 100
    wind_percent = wind_percent.reset_index()
    
    # Step 4: Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(wind_percent)
    
    # Define color palette for categories
    category_colors = {
        'Calm': '#d4efdf',
        'Light': '#aed6f1',
        'Moderate': '#5dade2',
        'Strong': '#2e86c1',
        'Very Strong': '#1b4f72',
        'Severe / Damaging': '#922b21'
    }
    
    # Ensure consistent category order
    categories = [cat for cat in category_colors if cat in wind_percent.columns]
    
    for category in categories:
        values = wind_percent[category]
        ax.bar(wind_percent['month'], values, bottom=bottom,
               label=category, color=category_colors[category], edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 5:  # Only label if significant
                ax.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=8, color='Black')
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    # Final formatting
    ax.set_title("Monthly Wind Speed Category Distribution (%)", fontsize=13)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Percentage of Days", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)



######row 3


    st.markdown("## 🧭 Monthly Wind Direction Distribution (%)")
    
    # Step 1: Count occurrences of each wind direction per month
    monthly_winddir = df.groupby(['month', 'wind_direction']).size().reset_index(name='count')
    
    # Step 2: Pivot to get wind directions as columns
    winddir_pivot = monthly_winddir.pivot(index='month', columns='wind_direction', values='count').fillna(0)
    
    # Step 3: Convert counts to percentages
    winddir_percent = winddir_pivot.div(winddir_pivot.sum(axis=1), axis=0) * 100
    winddir_percent = winddir_percent.reset_index()
    
    # Step 4: Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(winddir_percent)
    
    # Define a color palette for wind directions

    palette = sns.color_palette("tab20", len(winddir_percent.columns) - 1)
    direction_colors = dict(zip(winddir_percent.columns[1:], palette))
    
    # Plot each wind direction
    for direction in winddir_percent.columns[1:]:
        values = winddir_percent[direction]
        ax.bar(winddir_percent['month'], values, bottom=bottom,
               label=direction, color=direction_colors[direction], edgecolor='black', linewidth=0.3)
        
        # Add value labels for significant segments
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=7, color='Black')
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    # Final formatting
    ax.set_title("Monthly Wind Direction Distribution (%)", fontsize=13)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Percentage of Days", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=9, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)





######row 4


    st.markdown("## ☀️ Monthly Solar Radiation Category Distribution (%)")
    
    # Step 1: Count days per month per solar radiation category
    monthly_solar = df.groupby(['month', 'SolarRadiationCategory']).size().reset_index(name='count')
    
    # Step 2: Pivot to get categories as columns
    solar_pivot = monthly_solar.pivot(index='month', columns='SolarRadiationCategory', values='count').fillna(0)
    
    # Step 3: Convert counts to percentages
    solar_percent = solar_pivot.div(solar_pivot.sum(axis=1), axis=0) * 100
    solar_percent = solar_percent.reset_index()
    
    # Step 4: Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(solar_percent)
    
    # Define color palette for solar categories

    palette = sns.color_palette("Set2", len(solar_percent.columns) - 1)
    solar_colors = dict(zip(solar_percent.columns[1:], palette))
    
    # Plot each solar category
    for category in solar_percent.columns[1:]:
        values = solar_percent[category]
        ax.bar(solar_percent['month'], values, bottom=bottom,
               label=category, color=solar_colors[category], edgecolor='black', linewidth=0.3)
        
        # Add value labels for significant segments
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=7, color='black')
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    # Final formatting
    ax.set_title("Monthly Solar Radiation Category Distribution (%)", fontsize=13)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Percentage of Days", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)




#######row 5

    # --- Row: Monthly Solar Energy Category Distribution (in %) ---
    st.markdown("## 🔋 Monthly Solar Energy Category Distribution (%)")
    
    # Step 1: Count number of days in each solar energy category per month
    monthly_solar_energy = df.groupby(['month', 'SolarPerformanceCategory']).size().reset_index(name='count')
    
    # Step 2: Pivot to get categories as columns
    solar_energy_pivot = monthly_solar_energy.pivot(index='month', columns='SolarPerformanceCategory', values='count').fillna(0)
    
    # Step 3: Convert counts to percentages
    solar_energy_percent = solar_energy_pivot.div(solar_energy_pivot.sum(axis=1), axis=0) * 100
    solar_energy_percent = solar_energy_percent.reset_index()
    
    # Step 4: Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(solar_energy_percent)
    
    # Define color palette for solar performance categories
    
    palette = sns.color_palette("Set3", len(solar_energy_percent.columns) - 1)
    solar_perf_colors = dict(zip(solar_energy_percent.columns[1:], palette))
    
    # Plot each solar performance category
    for category in solar_energy_percent.columns[1:]:
        values = solar_energy_percent[category]
        ax.bar(solar_energy_percent['month'], values, bottom=bottom,
               label=category, color=solar_perf_colors[category], edgecolor='black', linewidth=0.3)
        
        # Add value labels for significant segments
        for i, (val, btm) in enumerate(zip(values, bottom)):
            if val > 5:
                ax.text(i + 1, btm + val / 2, f'{val:.0f}%', ha='center', va='center', fontsize=7, color='black')
        bottom = [btm + val for btm, val in zip(bottom, values)]
    
    # Final formatting
    ax.set_title("Monthly Solar Energy Category Distribution (%)", fontsize=13)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Percentage of Days", fontsize=11)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=True)
    
    fig.tight_layout()
    st.pyplot(fig)







