import streamlit as st
import pandas as pd
import auth
import Data_Loader as dl
import utils as ut

# Import all views
from views import (
    CLIMATE_OVERVIEW_DASHBOARD,
    AGRICULTURE_PLANNING_DASHBOARD,
    DISASTER_MANAGEMENT_DASHBOARD,
    URBAN_INFRASTRUCTURE_PLANNING_DASHBOARD,
    ENERGY_AND_RESOURCE_MANAGEMENT_DASHBOARD,
    RAINFALL_PREDICTION_DASHBOARD
)

# 1. Page Config
st.set_page_config(page_title="Guwahati Weather Intelligence", page_icon="🌦️", layout="wide")
ut.load_css()  # Apply soft background and card styling

# 2. Auth
if not auth.check_password():
    st.stop()

# 3. Load Data
df_weather = dl.load_data()

# 4. Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.info(f"User: {st.session_state.get('current_user', 'Guest')}")

# 5. Global Date Filter
st.sidebar.markdown("---")
date_col = [col for col in df_weather.columns if 'datetime' in col.lower()]
if date_col:
    date_field = date_col[0]
    min_date = df_weather[date_field].min()
    max_date = df_weather[date_field].max()
    start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date])

    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()

    df_weather = df_weather[
        (df_weather[date_field] >= start_date) &
        (df_weather[date_field] <= end_date)
    ]




####filter
# 5b. Additional Slicers
st.sidebar.markdown("### 📅 Temporal Filters")

# Year filter
years = sorted(df_weather['year'].dropna().unique())
selected_years = st.sidebar.multiselect("Select Years", years, default=years)
df_weather = df_weather[df_weather['year'].isin(selected_years)]

# Season filter
seasons = df_weather['seasons'].dropna().unique()
selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons)
df_weather = df_weather[df_weather['seasons'].isin(selected_seasons)]

# Month filter
months = sorted(df_weather['month'].dropna().unique())
selected_months = st.sidebar.multiselect("Select Months", months, default=months)
df_weather = df_weather[df_weather['month'].isin(selected_months)]

st.sidebar.markdown("### 🌧️ Rainfall Filters")

# Rain Hour Bins (string column)
if 'precip_hours_bucket' in df_weather.columns:
    rain_bins = df_weather['precip_hours_bucket'].dropna().unique()
    selected_rain_bins = st.sidebar.multiselect("Rain Hour Bins", rain_bins, default=rain_bins)
    df_weather = df_weather[df_weather['precip_hours_bucket'].isin(selected_rain_bins)]

# Rain Intensity
if 'rain_intensity' in df_weather.columns:
    rain_intensity = df_weather['rain_intensity'].dropna().unique()
    selected_rain_intensity = st.sidebar.multiselect("Rain Intensity", rain_intensity, default=rain_intensity)
    df_weather = df_weather[df_weather['rain_intensity'].isin(selected_rain_intensity)]

st.sidebar.markdown("### 🌡️ Temperature Filters")

# Temperature Stress
if 'temp_stress_category' in df_weather.columns:
    temp_stress = df_weather['temp_stress_category'].dropna().unique()
    selected_temp_stress = st.sidebar.multiselect("Temperature Stress", temp_stress, default=temp_stress)
    df_weather = df_weather[df_weather['temp_stress_category'].isin(selected_temp_stress)]

st.sidebar.markdown("### 🌬️ Wind & Pressure Alerts")

# Wind Alerts
if 'WindSpeedCategory' in df_weather.columns:
    wind_alerts = df_weather['WindSpeedCategory'].dropna().unique()
    selected_wind_alerts = st.sidebar.multiselect("Wind Alerts", wind_alerts, default=wind_alerts)
    df_weather = df_weather[df_weather['WindSpeedCategory'].isin(selected_wind_alerts)]

# Gust Alerts
if 'WindGustCategory' in df_weather.columns:
    gust_alerts = df_weather['WindGustCategory'].unique()
    selected_gust_alerts = st.sidebar.multiselect("Gust Alerts", gust_alerts, default=gust_alerts)
    df_weather = df_weather[df_weather['WindGustCategory'].isin(selected_gust_alerts)]

# Sea Level Pressure Alerts
if 'SeaLevelPressureCategory' in df_weather.columns:
    slp_alerts = df_weather['SeaLevelPressureCategory'].unique()
    selected_slp_alerts = st.sidebar.multiselect("SLP Alerts", slp_alerts, default=slp_alerts)
    df_weather = df_weather[df_weather['SeaLevelPressureCategory'].isin(selected_slp_alerts)]






# 6. Dashboard Selection
page = st.sidebar.radio("Go to", [
    "CLIMATE OVERVIEW DASHBOARD",
    "AGRICULTURE PLANNING DASHBOARD",
    "DISASTER MANAGEMENT DASHBOARD",
    "URBAN INFRASTRUCTURE PLANNING DASHBOARD",
    "ENERGY AND RESOURCE MANAGEMENT DASHBOARD",
    "RAINFALL PREDICTION DASHBOARD"
])

# 7. Routing Logic
if page == "CLIMATE OVERVIEW DASHBOARD":
    CLIMATE_OVERVIEW_DASHBOARD.show(df_weather)
elif page == "AGRICULTURE PLANNING DASHBOARD":
    AGRICULTURE_PLANNING_DASHBOARD.show(df_weather)
elif page == "DISASTER MANAGEMENT DASHBOARD":
    DISASTER_MANAGEMENT_DASHBOARD.show(df_weather)
elif page == "URBAN INFRASTRUCTURE PLANNING DASHBOARD":
    URBAN_INFRASTRUCTURE_PLANNING_DASHBOARD.show(df_weather)
elif page == "ENERGY AND RESOURCE MANAGEMENT DASHBOARD":
    ENERGY_AND_RESOURCE_MANAGEMENT_DASHBOARD.show(df_weather)
elif page == "RAINFALL PREDICTION DASHBOARD":
    RAINFALL_PREDICTION_DASHBOARD.show(df_weather)


# 8. Logout
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state["password_correct"] = False
    st.rerun()





