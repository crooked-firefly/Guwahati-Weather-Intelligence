# =============================================================
# feature_engineering.py
# Guwahati Daily Weather — Feature Engineering Pipeline
# =============================================================

import pandas as pd
import numpy as np
import os
from datetime import datetime as dt


# ============================================================
# STEP 1 — LOAD RAW DATA
# ============================================================
def load_raw_data(filepath):
    df = pd.read_csv(filepath)   # ← uses filepath parameter (no hardcoded path)
    print(f"✅ Raw data loaded: {df.shape}")
    return df


# ============================================================
# STEP 2 — DATETIME PARSING
# ============================================================
def parse_datetime_columns(df):
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')

    df['sunrise'] = df['sunrise'].str.replace('T', ' ')
    df['sunset']  = df['sunset'].str.replace('T', ' ')
    df['sunrise'] = pd.to_datetime(df['sunrise'], format='%Y-%m-%d %H:%M:%S')
    df['sunset']  = pd.to_datetime(df['sunset'],  format='%Y-%m-%d %H:%M:%S')

    print("✅ Datetime columns parsed")
    return df


# ============================================================
# STEP 3 — BASIC CLEANING
# ============================================================
def basic_cleaning(df):
    df['name'] = "Guwahati"
    df['precip'].fillna(0, inplace=True)
    df['preciptype'].fillna('NONE', inplace=True)
    df['winddir'] = pd.to_numeric(df['winddir'], errors='coerce')
    df['uvindex'] = pd.to_numeric(df['uvindex'], errors='coerce')
    print("✅ Basic cleaning done")
    return df


# ============================================================
# STEP 4 — DATA CLEANING & DISCREPANCY FIXES
# ============================================================
def clean_data(df):

    df = df[df['temp'].notna()]
    print(f"✅ Dropped null temp rows — shape: {df.shape}")

    df.loc[(df['snowdepth'] > 0) & (df['snow'] == 0), 'snowdepth'] = 0
    print("✅ Fixed snowdepth inconsistencies")

    df.drop(columns=['snow', 'snowdepth'], inplace=True, errors='ignore')
    print("✅ Dropped snow and snowdepth columns")

    dates = df[df['tempmin'] == df['tempmax']]['datetime'].tolist()
    for date in dates:
        week_number = date.isocalendar()[1]
        year        = date.year
        tempmax     = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['tempmax'].mean()
        tempmin     = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['tempmin'].mean()
        temp        = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['temp'].mean()
        feelslikemax= df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['feelslikemax'].mean()
        feelslikemin= df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['feelslikemin'].mean()
        feelslike   = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['feelslike'].mean()
        df.loc[df['datetime'] == date, 'tempmax']     = tempmax
        df.loc[df['datetime'] == date, 'tempmin']     = tempmin
        df.loc[df['datetime'] == date, 'temp']        = temp
        df.loc[df['datetime'] == date, 'feelslikemax']= feelslikemax
        df.loc[df['datetime'] == date, 'feelslikemin']= feelslikemin
        df.loc[df['datetime'] == date, 'feelslike']   = feelslike
    print("✅ Fixed tempmin == tempmax rows")

    negative_tempmin_dates = df[df['tempmin'] < 0]['datetime'].tolist()
    for date in negative_tempmin_dates:
        week_number  = date.isocalendar()[1]
        year         = date.year
        tempmin      = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmin'] >= 0)      & (df['datetime'].dt.year == year)]['tempmin'].mean()
        temp         = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['temp'] >= 0)         & (df['datetime'].dt.year == year)]['temp'].mean()
        tempmax      = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] >= 0)      & (df['datetime'].dt.year == year)]['tempmax'].mean()
        feelslikemax = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['feelslikemax'] >= 0) & (df['datetime'].dt.year == year)]['feelslikemax'].mean()
        feelslikemin = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['feelslikemin'] >= 0) & (df['datetime'].dt.year == year)]['feelslikemin'].mean()
        feelslike    = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['feelslike'] >= 0)    & (df['datetime'].dt.year == year)]['feelslike'].mean()
        df.loc[df['datetime'] == date, 'tempmin']     = tempmin
        df.loc[df['datetime'] == date, 'temp']        = temp
        df.loc[df['datetime'] == date, 'feelslikemax']= tempmax
        df.loc[df['datetime'] == date, 'feelslikemin']= feelslikemin
        df.loc[df['datetime'] == date, 'feelslike']   = feelslike
        df.loc[df['datetime'] == date, 'tempmax']     = feelslikemax
    print("✅ Fixed negative tempmin rows")

    anomaly_date = dt.strptime('1981-01-08 00:00:00', '%Y-%m-%d %H:%M:%S')
    week_number  = anomaly_date.isocalendar()[1]
    year         = anomaly_date.year
    tempmax      = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['tempmax'].mean()
    tempmin      = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['tempmin'].mean()
    temp         = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['temp'].mean()
    feelslikemax = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['feelslikemax'].mean()
    feelslikemin = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['feelslikemin'].mean()
    feelslike    = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['tempmax'] != df['tempmin']) & (df['datetime'].dt.year == year)]['feelslike'].mean()
    df.loc[df['datetime'] == anomaly_date, 'tempmax']     = tempmax
    df.loc[df['datetime'] == anomaly_date, 'tempmin']     = tempmin
    df.loc[df['datetime'] == anomaly_date, 'temp']        = temp
    df.loc[df['datetime'] == anomaly_date, 'feelslikemax']= feelslikemax
    df.loc[df['datetime'] == anomaly_date, 'feelslikemin']= feelslikemin
    df.loc[df['datetime'] == anomaly_date, 'feelslike']   = feelslike
    print("✅ Fixed anomaly date 1981-01-08")

    solar_radiation_1 = df[(df['datetime'].dt.month == 7) & (df['solarradiation'] != 0) & (df['datetime'].dt.year == 2016) & (df['conditions'] == 'Rain, Partially cloudy') & (df['description'] == 'Partly cloudy throughout the day with late afternoon rain.') & (df['cloudcover'] >= 80) & (df['cloudcover'] <= 90)]['solarradiation'].mean()
    solar_energy_1    = df[(df['datetime'].dt.month == 7) & (df['solarenergy'] != 0)    & (df['datetime'].dt.year == 2016) & (df['conditions'] == 'Rain, Partially cloudy') & (df['description'] == 'Partly cloudy throughout the day with late afternoon rain.') & (df['cloudcover'] >= 80) & (df['cloudcover'] <= 90)]['solarenergy'].mean()
    uv_index_1        = df[(df['datetime'].dt.month == 7) & (df['uvindex'] != 0)        & (df['datetime'].dt.year == 2016) & (df['conditions'] == 'Rain, Partially cloudy') & (df['description'] == 'Partly cloudy throughout the day with late afternoon rain.') & (df['cloudcover'] >= 80) & (df['cloudcover'] <= 90)]['uvindex'].mean()
    df.loc[df['datetime'] == dt.strptime('2016-07-13 00:00:00', '%Y-%m-%d %H:%M:%S'), 'solarradiation'] = solar_radiation_1
    df.loc[df['datetime'] == dt.strptime('2016-07-13 00:00:00', '%Y-%m-%d %H:%M:%S'), 'solarenergy']    = solar_energy_1
    df.loc[df['datetime'] == dt.strptime('2016-07-13 00:00:00', '%Y-%m-%d %H:%M:%S'), 'uvindex']        = uv_index_1
    print("✅ Fixed solar values for 2016-07-13")

    solar_radiation_2 = df[(df['datetime'].dt.month == 1) & (df['solarradiation'] != 0) & (df['datetime'].dt.year == 2018) & (df['conditions'] == 'Partially cloudy') & (df['cloudcover'] >= 35) & (df['cloudcover'] <= 45) & (df['description'] == 'Partly cloudy throughout the day.')]['solarradiation'].mean()
    solar_energy_2    = df[(df['datetime'].dt.month == 1) & (df['solarenergy'] != 0)    & (df['datetime'].dt.year == 2018) & (df['conditions'] == 'Partially cloudy') & (df['cloudcover'] >= 35) & (df['cloudcover'] <= 45) & (df['description'] == 'Partly cloudy throughout the day.')]['solarenergy'].mean()
    uv_index_2        = df[(df['datetime'].dt.month == 1) & (df['uvindex'] != 0)        & (df['datetime'].dt.year == 2018) & (df['conditions'] == 'Partially cloudy') & (df['cloudcover'] >= 35) & (df['cloudcover'] <= 45) & (df['description'] == 'Partly cloudy throughout the day.')]['uvindex'].mean()
    df.loc[df['datetime'] == dt.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), 'solarradiation'] = solar_radiation_2
    df.loc[df['datetime'] == dt.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), 'solarenergy']    = solar_energy_2
    df.loc[df['datetime'] == dt.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'), 'uvindex']        = uv_index_2
    print("✅ Fixed solar values for 2018-01-01")

    mask = df['windgust'] < df['windspeed']
    df.loc[mask, ['windgust', 'windspeed']] = df.loc[mask, ['windspeed', 'windgust']].values
    print("✅ Fixed windgust < windspeed rows")

    df.loc[df['datetime'] == dt.strptime('2003-01-08 00:00:00', '%Y-%m-%d %H:%M:%S'), 'conditions'] = 'Partially cloudy'
    target_date = '2003-01-08 00:00:00'
    week_number = df.loc[df['datetime'] == target_date, 'datetime'].dt.isocalendar().week.values[0]
    cloudcover  = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['datetime'].dt.year == 2003) & (df['conditions'] == 'Partially cloudy') & (df['precip'] == 0) & (df['description'] == 'Partly cloudy throughout the day.') & (df['cloudcover'].notna())]['cloudcover'].mean()
    df.loc[df['datetime'] == dt.strptime('2003-01-08 00:00:00', '%Y-%m-%d %H:%M:%S'), 'cloudcover'] = cloudcover
    print("✅ Fixed cloudcover for 2003-01-08")

    date        = '1984-02-01 00:00:00'
    week_number = dt.strptime(date, '%Y-%m-%d %H:%M:%S').isocalendar()[1]
    year        = dt.strptime(date, '%Y-%m-%d %H:%M:%S').year
    sealevelpressure_null = df[(df['datetime'].dt.isocalendar().week == week_number) & (df['datetime'].dt.year == year) & (df['sealevelpressure'].notna())]['sealevelpressure'].mean()
    df.loc[df['datetime'] == dt.strptime('1984-02-01 00:00:00', '%Y-%m-%d %H:%M:%S'), 'sealevelpressure'] = sealevelpressure_null
    print("✅ Fixed sealevelpressure for 1984-02-01")

    dates  = '2011-06-09 00:00:00'
    months = dt.strptime(dates, '%Y-%m-%d %H:%M:%S').month
    years  = dt.strptime(dates, '%Y-%m-%d %H:%M:%S').year
    sealevelpressure_mean = df[(df['datetime'].dt.month == months) & (df['datetime'].dt.year == years) & (df['precip'] < 10) & (df['sealevelpressure'].notna())]['sealevelpressure'].mean()
    df['sealevelpressure'].fillna(sealevelpressure_mean, inplace=True)
    print("✅ Filled remaining sealevelpressure nulls")

    visibility_date        = '1976-10-20 00:00:00'
    week_number_visibility = dt.strptime(visibility_date, '%Y-%m-%d %H:%M:%S').isocalendar()[1]
    year_visibility        = dt.strptime(visibility_date, '%Y-%m-%d %H:%M:%S').year
    visibility_null        = df[(df['datetime'].dt.isocalendar().week == week_number_visibility) & (df['datetime'].dt.year == year_visibility) & (df['visibility'].notna())][['visibility']].mean()
    df.loc[df['datetime'] == dt.strptime('1976-10-20 00:00:00', '%Y-%m-%d %H:%M:%S'), 'visibility'] = visibility_null
    print("✅ Fixed visibility for 1976-10-20")

    dates_visibility  = '2011-06-09 00:00:00'
    months_visibility = dt.strptime(dates_visibility, '%Y-%m-%d %H:%M:%S').month
    years_visibility  = dt.strptime(dates_visibility, '%Y-%m-%d %H:%M:%S').year
    visibility_mean   = df[(df['datetime'].dt.month == months_visibility) & (df['datetime'].dt.year == years_visibility) & (df['precip'] < 10) & (df['visibility'].notna())]['visibility'].mean()
    df['visibility'].fillna(visibility_mean, inplace=True)
    print("✅ Filled remaining visibility nulls")

    print(f"\n✅ All cleaning steps complete — shape: {df.shape}")
    return df


# ============================================================
# STEP 5 — STATION FEATURES
# ============================================================
def create_station_features(df):
    df['stations'] = df['stations'].str.replace('42410099999.0', '42410099999')
    df['stations'] = df['stations'].str.replace('42410099999',   'Guwahati (WMO ID)')
    df['stations'] = df['stations'].str.replace('42516099999',   'Tezpur (WMO ID)')
    df['stations'] = df['stations'].str.replace('42414099999',   'Shillong (WMO ID)')
    df['stations'] = df['stations'].str.replace('42408099999',   'North Lakhimpur (WMO ID)')
    df['stations'] = df['stations'].str.replace('42512099999',   'Silchar (WMO ID)')
    df['stations'] = df['stations'].str.replace('42409099999',   'Dhubri (WMO ID)')
    df['stations'] = df['stations'].str.replace('VEGT',          'Guwahati Airport (ICAO)')
    df['stations'] = df['stations'].str.replace('INMU0020468',   'Unknown (IMD Hybrid Code)')
    df['stations'] = df['stations'].str.replace('INI0000VEBI',   'Unknown (IMD Hybrid Code)')
    df['stations'] = df['stations'].str.replace('INU042410-1',   'Unknown (IMD Hybrid Code)')
    df['stations'] = df['stations'].str.replace('INM00042516',   'Unknown (IMD Hybrid Code)')
    df['stations'] = df['stations'].str.replace('INM00042408',   'Unknown (IMD Hybrid Code)')
    df['stations'] = df['stations'].str.replace('remote',        'Remote/Unspecified')
    df['primary_station']    = df['stations'].str.split(',').apply(
        lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown'
    )
    df['secondary_stations'] = df['stations'].str.split(',').apply(
        lambda x: ','.join(x[1:]).strip() if isinstance(x, list) and len(x) > 1 else 'N/A'
    )
    print("✅ Station features created")
    return df


# ============================================================
# STEP 6 — DATETIME FEATURES
# ============================================================
def create_datetime_features(df):
    df['day']            = df['datetime'].dt.day
    df['month']          = df['datetime'].dt.month
    df['year']           = df['datetime'].dt.year
    df['DayofWeekReal']  = df['datetime'].dt.dayofweek
    df['weekofyearReal'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['quarter']        = df['datetime'].dt.quarter
    df['is_weekend']     = df['DayofWeekReal'].apply(lambda x: 1 if x >= 5 else 0)
    df['weekday_name']   = df['datetime'].dt.day_name()
    print("✅ Datetime features created")
    return df


# ============================================================
# STEP 7 — TEMPERATURE FEATURES
# ============================================================
def create_temperature_features(df):
    df['temprature_range'] = df['tempmax'] - df['tempmin']
    df['is_heatwave']      = df['tempmax'].apply(lambda x: 1 if x >= 35 else 0)
    df['is_coldwave']      = df['tempmin'].apply(lambda x: 1 if x <= 5  else 0)

    def classify_temp_stress(row):
        tmax = row['tempmax']
        tmin = row['tempmin']
        if pd.isnull(tmax) or pd.isnull(tmin): return None
        elif tmax >= 35:                        return "Heat Stress"
        elif tmin >= 15 and tmax <= 30:         return "Optimal"
        elif tmin <= 10:                        return "Cold Stress"
        elif tmax > 30:                         return "Beyond Optimal"
        elif tmin > 10:                         return "Below Optimal"
        else:                                   return None

    df['temp_stress_category'] = df.apply(classify_temp_stress, axis=1)
    print("✅ Temperature features created")
    return df


# ============================================================
# STEP 8 — DEW & HUMIDITY FEATURES
# ============================================================
def create_humidity_features(df):
    df['dew_point_diff'] = df['temp'] - df['dew']
    df['is_humidday']    = df['humidity'].apply(lambda x: 1 if x >= 85 else 0)
    print("✅ Humidity features created")
    return df


# ============================================================
# STEP 9 — WIND FEATURES
# ============================================================
def create_wind_features(df):
    df['is_windyday'] = df['windspeed'].apply(lambda x: 1 if x >= 30 else 0)

    def classify_wind_speed(speed):
        if pd.isnull(speed): return None
        elif speed <= 5:     return "Calm"
        elif speed <= 15:    return "Light"
        elif speed <= 25:    return "Moderate"
        elif speed <= 40:    return "Strong"
        elif speed <= 60:    return "Very Strong"
        elif speed > 60:     return "Severe / Damaging"
        else:                return None

    df['WindSpeedCategory'] = df['windspeed'].apply(classify_wind_speed)

    def classify_wind_gust(gust):
        if pd.isnull(gust): return "N/A"
        elif gust < 20:     return "Weak Gust"
        elif gust <= 35:    return "Moderate Gust"
        elif gust <= 50:    return "Strong Gust"
        elif gust <= 70:    return "Very Strong Gust"
        elif gust > 70:     return "Extreme / Damaging Gust"
        else:               return "N/A"

    df['WindGustCategory'] = df['windgust'].apply(classify_wind_gust)
    df['gust_factor']      = df['windgust'] / df['windspeed']

    def wind_dir_label(degree, speed):
        if pd.isna(degree) or speed == 0:       return 'NONE'
        elif degree <= 22.5 or degree >= 337.5: return 'N'
        elif degree > 22.5  and degree < 67.5:  return 'NE'
        elif degree >= 67.5 and degree < 112.5: return 'E'
        elif degree >= 112.5 and degree < 157.5:return 'SE'
        elif degree >= 157.5 and degree < 202.5:return 'S'
        elif degree >= 202.5 and degree < 247.5:return 'SW'
        elif degree >= 247.5 and degree < 292.5:return 'W'
        else:                                   return 'NW'

    df['wind_direction'] = df.apply(
        lambda row: wind_dir_label(row['winddir'], row['windspeed']), axis=1
    )
    df['winddir'] = df['winddir'].fillna(0)
    print("✅ Wind features created")
    return df


# ============================================================
# STEP 10 — PRESSURE FEATURES
# ============================================================
def create_pressure_features(df):
    df['pressure_drop_flag'] = df['sealevelpressure'].apply(
        lambda x: 1 if x < 1000 else 0
    )

    def classify_slp_category(slp):
        if pd.isnull(slp): return None
        elif slp < 990:    return "Very Low Pressure (Deep Cyclonic System)"
        elif slp <= 1000:  return "Low Pressure"
        elif slp <= 1010:  return "Slightly Below Normal"
        elif slp <= 1020:  return "Normal Pressure"
        elif slp <= 1030:  return "Slightly Above Normal"
        elif slp <= 1040:  return "High Pressure"
        elif slp > 1040:   return "Very High Pressure (Strong Anticyclone)"
        else:              return None

    df['SeaLevelPressureCategory'] = df['sealevelpressure'].apply(classify_slp_category)
    print("✅ Pressure features created")
    return df


# ============================================================
# STEP 11 — VISIBILITY FEATURES
# ============================================================
def create_visibility_features(df):
    df['visibility_level'] = df['visibility'].apply(
        lambda x: 'Low' if x <= 1 else ('Medium' if x > 1 and x <= 5 else 'High')
    )
    print("✅ Visibility features created")
    return df


# ============================================================
# STEP 12 — SEASON FEATURES
# ============================================================
def create_season_features(df):
    df['seasons'] = df['month'].apply(
        lambda x: 'Winter'       if x in [12, 1, 2]   else
                  'Summer'       if x in [3, 4, 5]    else
                  'Monsoon'      if x in [6, 7, 8, 9] else
                  'Post-Monsoon'
    )
    df['is_remote_reading'] = df['secondary_stations'].apply(
        lambda x: 1 if x == 'Remote/Unspecified' else 0
    )
    print("✅ Season features created")
    return df


# ============================================================
# STEP 13 — SOLAR & UV FEATURES
# ============================================================
def create_solar_features(df):
    def classify_solar_radiation(sr):
        if pd.isnull(sr): return None
        elif sr < 100:    return "Low"
        elif sr < 200:    return "Medium"
        elif sr >= 200:   return "High"
        else:             return None

    def classify_solar_performance(se):
        if pd.isnull(se): return None
        elif se < 10:     return "Stress Day (<10)"
        elif se <= 14:    return "Moderate Solar Day[10-14]"
        elif se <= 20:    return "Normal Operational Day[15-20]"
        elif se > 20:     return "Good Solar Day(>20)"
        else:             return None

    def classify_uv(uv):
        if pd.isnull(uv): return None
        elif uv <= 2:     return 'Low'
        elif uv <= 5:     return 'Medium'
        else:             return 'High'

    df['SolarRadiationCategory']   = df['solarradiation'].apply(classify_solar_radiation)
    df['SolarPerformanceCategory'] = df['solarenergy'].apply(classify_solar_performance)
    df['uv_category']              = df['uvindex'].apply(classify_uv)
    print("✅ Solar & UV features created")
    return df


# ============================================================
# STEP 14 — PRECIPITATION FEATURES
# ============================================================
def create_precipitation_features(df):
    df['rain_intensity'] = df['precip'].apply(
        lambda x: 'No Rain'       if x == 0            else
                  'Light Rain'    if x > 0  and x <= 15 else
                  'Moderate Rain' if x > 15 and x <= 60 else
                  'Heavy Rain'
    )

    precipitation_cover_mapping = {
        0.0: 0.0,    4.17: 1.0,   8.33: 2.0,   12.5: 3.0,
        16.67: 4.0,  20.83: 5.0,  25: 6.0,     29.17: 7.0,
        33.33: 8.0,  37.5: 9.0,   41.67: 10.0, 45.83: 11.0,
        50: 12.0,    54.17: 13.0, 58.33: 14.0, 62.5: 15.0,
        66.67: 16.0, 70.83: 17.0, 75: 18.0,    79.17: 19.0,
        83.33: 20.0
    }
    df['precip_hours'] = df['precipcover'].map(precipitation_cover_mapping)

    def categorize_precip_hours(x):
        if pd.isnull(x): return None
        elif x == 0:     return "0"
        elif x <= 3:     return "0–3"
        elif x <= 9:     return "3–9"
        elif x <= 15:    return "9–15"
        else:            return ">15"

    df['precip_hours_bucket'] = df['precip_hours'].apply(categorize_precip_hours)

    def classify_rain_day(p):
        if pd.isnull(p): return None
        elif p > 0:      return "Rainy Day"
        else:            return "Dry Day"

    df['is_rainy_day'] = df['precip'].apply(classify_rain_day)
    print("✅ Precipitation features created")
    return df


# ============================================================
# STEP 15 — WEATHER ALERT FEATURES
# ============================================================
def create_weather_alert_features(df):
    def classify_weather_alert(row):
        if pd.isnull(row['tempmax']) or pd.isnull(row['tempmin']) or pd.isnull(row['precip']):
            return None
        elif row['tempmin'] <= 10: return "Coldwave"
        elif row['tempmax'] >= 33: return "Heatwave"
        elif row['precip'] > 50:   return "Heavy Rainfall"
        else:                      return "Normal"

    df['WeatherAlertType'] = df.apply(classify_weather_alert, axis=1)
    print("✅ Weather Alert features created")
    return df


# ============================================================
# MASTER PIPELINE FUNCTION
# ============================================================
def run_feature_engineering(filepath, save_path):
    print("🚀 Starting Feature Engineering Pipeline...\n")

    # Create output directory if needed
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df = load_raw_data(filepath)
    df = parse_datetime_columns(df)
    df = basic_cleaning(df)
    df = clean_data(df)
    df = create_station_features(df)
    df = create_datetime_features(df)
    df = create_temperature_features(df)
    df = create_humidity_features(df)
    df = create_wind_features(df)
    df = create_pressure_features(df)
    df = create_visibility_features(df)
    df = create_season_features(df)
    df = create_solar_features(df)
    df = create_precipitation_features(df)
    df = create_weather_alert_features(df)

    df.to_csv(save_path, index=False)

    print(f"\n✅ Feature Engineering Complete!")
    print(f"📁 Saved to: {save_path}")
    print(f"📊 Final shape: {df.shape}")
    return df


# ============================================================
# RUN — only when run directly on local machine
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = run_feature_engineering(
        filepath  = os.path.join(BASE_DIR, '..', 'data', 'guwahati_weather_1972_2025.csv'),
        save_path = os.path.join(BASE_DIR, '..', 'data', 'processed_data.csv')
    )
    print("\nColumn list:")
    print(df.columns.tolist())