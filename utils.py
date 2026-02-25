import streamlit as st
import plotly.graph_objects as go

def load_css():
    st.markdown("""
    <style>
        /* Main App Background */
        .stApp {
            background-color: #f4f4f4; /* Soft light grey */
            color: #000000;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            color: #000000;
            border-right: 1px solid #dddddd;
        }

        /* KPI Card Styling */
        .kpi-card {
            background-color: #ffffff;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #cccccc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
            margin-bottom: 10px;
        }

        .kpi-title {
            font-size: 14px;
            color: #555555;
            margin-bottom: 5px;
        }

        .kpi-value {
            font-size: 24px;
            font-weight: bold;
            color: #000000;
        }

        /* Input Fields */
        .stTextInput input {
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
        }

        /* Metrics */
        [data-testid="stMetricLabel"] { color: #555555; }
        [data-testid="stMetricValue"] { color: #000000; }
    </style>
    """, unsafe_allow_html=True)

def kpi_card(col, title, value, prefix="", suffix=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{prefix}{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)

def style_chart(fig):
    fig.update_layout(
        paper_bgcolor='#f4f4f4',
        plot_bgcolor='#ffffff',
        font_color='#000000',
        title_font_color='#000000',
        xaxis=dict(showgrid=True, color='#333333'),
        yaxis=dict(showgrid=True, gridcolor='#dddddd', color='#333333'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
