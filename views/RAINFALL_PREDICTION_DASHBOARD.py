


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

import utils as ut


def show(df):
    st.title("🌧️ Rainfall Prediction Dashboard")
    st.markdown("Machine Learning insights using Logistic Regression trained on Guwahati weather data (1973–2024)")

    #  Check if predictions exist
    if 'predicted_rain' not in df.columns or 'rain_probability' not in df.columns:
        st.warning("⚠️ Predictions not found in data. Please run predict.py first to generate predictions.")
        return

    # Prepare binary labels for evaluation 
    df = df.copy()
    df['actual_binary']    = (df['is_rainy_day'] == 'Rainy Day').astype(int)
    df['predicted_binary'] = (df['predicted_rain'] == 'Rainy Day').astype(int)
    df['rain_prob_decimal'] = df['rain_probability'] / 100

    # ============================================================
    # KPI CARDS
    # ============================================================
    st.subheader("📊 Model Performance Summary")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy  = accuracy_score(df['actual_binary'], df['predicted_binary'])
    precision = precision_score(df['actual_binary'], df['predicted_binary'])
    recall    = recall_score(df['actual_binary'], df['predicted_binary'])
    f1        = f1_score(df['actual_binary'], df['predicted_binary'])
    roc_auc   = roc_auc_score(df['actual_binary'], df['rain_prob_decimal'])
    avg_prob  = df['rain_probability'].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ut.kpi_card(c1, "✅ Accuracy",    f"{accuracy*100:.1f}",  suffix="%")
    ut.kpi_card(c2, "🎯 Precision",   f"{precision*100:.1f}", suffix="%")
    ut.kpi_card(c3, "🌧️ Rain Recall", f"{recall*100:.1f}",   suffix="%")
    ut.kpi_card(c4, "⚖️ F1 Score",    f"{f1*100:.1f}",       suffix="%")
    ut.kpi_card(c5, "📈 ROC-AUC",     f"{roc_auc:.4f}",      suffix="")
    ut.kpi_card(c6, "💧 Avg Rain Prob",f"{avg_prob:.1f}",    suffix="%")

    st.markdown("---")

    # ============================================================
    # ROW 1 — Actual vs Predicted + Confusion Matrix
    # ============================================================
    st.subheader("🔍 Actual vs Predicted Comparison")

    col1, col2 = st.columns(2)

    # ── Chart 1: Actual vs Predicted monthly comparison ──────
    with col1:
        df['month'] = df['month'].astype(int)

        actual_monthly    = df.groupby('month')['actual_binary'].sum().reset_index()
        predicted_monthly = df.groupby('month')['predicted_binary'].sum().reset_index()
        monthly_compare   = pd.merge(actual_monthly, predicted_monthly, on='month')
        monthly_compare.columns = ['month', 'Actual Rainy Days', 'Predicted Rainy Days']

        all_months = pd.DataFrame({'month': range(1, 13)})
        monthly_compare = pd.merge(all_months, monthly_compare, on='month', how='left').fillna(0)

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        x     = np.arange(len(monthly_compare['month']))
        width = 0.35

        bars1 = ax1.bar(x - width/2, monthly_compare['Actual Rainy Days'],    width, label='Actual',    color='steelblue')
        bars2 = ax1.bar(x + width/2, monthly_compare['Predicted Rainy Days'], width, label='Predicted', color='lightcoral')

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Rainy Days')
        ax1.set_title('Actual vs Predicted Rainy Days by Month')
        ax1.set_xticks(x)
        ax1.set_xticklabels(range(1, 13))
        ax1.legend()

        for bar in bars1:
            ax1.annotate(f'{int(bar.get_height())}',
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
        for bar in bars2:
            ax1.annotate(f'{int(bar.get_height())}',
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)

        fig1.tight_layout()
        st.pyplot(fig1)

    # ── Chart 2: Confusion Matrix ─────────────────────────────
    with col2:
        cm = confusion_matrix(df['actual_binary'], df['predicted_binary'])

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Dry', 'Predicted Rain'],
            yticklabels=['Actual Dry', 'Actual Rain'],
            ax=ax2, linewidths=0.5
        )
        ax2.set_title('Confusion Matrix')
        ax2.set_ylabel('Actual Label')
        ax2.set_xlabel('Predicted Label')

        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                ax2.text(j + 0.5, i + 0.75, f'({cm[i,j]/total*100:.1f}%)',
                         ha='center', va='center', fontsize=9, color='gray')

        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # ============================================================
    # ROW 2 — ROC AUC Curve + Rain Probability Distribution
    # ============================================================
    st.subheader("📈 ROC AUC Curve & Probability Distribution")

    col1, col2 = st.columns(2)

    # ── Chart 3: ROC AUC Curve ───────────────────────────────
    with col1:
        fpr, tpr, thresholds = roc_curve(df['actual_binary'], df['rain_prob_decimal'])
        roc_auc_val          = auc(fpr, tpr)

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
        ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
        ax3.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC AUC Curve')
        ax3.legend(loc='lower right')
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1.02])

        fig3.tight_layout()
        st.pyplot(fig3)

    # ── Chart 4: Rain Probability Distribution ────────────────
    with col2:
        fig4, ax4 = plt.subplots(figsize=(5, 4))

        dry_probs   = df[df['actual_binary'] == 0]['rain_probability']
        rainy_probs = df[df['actual_binary'] == 1]['rain_probability']

        ax4.hist(dry_probs,   bins=30, alpha=0.6, color='wheat',     label='Actual Dry Day',   edgecolor='orange')
        ax4.hist(rainy_probs, bins=30, alpha=0.6, color='steelblue', label='Actual Rainy Day', edgecolor='blue')

        ax4.axvline(x=50, color='red', linestyle='--', lw=1.5, label='Decision Threshold (50%)')
        ax4.set_xlabel('Rain Probability (%)')
        ax4.set_ylabel('Number of Days')
        ax4.set_title('Rain Probability Distribution by Actual Class')
        ax4.legend()

        fig4.tight_layout()
        st.pyplot(fig4)

    st.markdown("---")

    # ============================================================
    # ROW 3 — Feature Importance + Prediction by Season
    # ============================================================
    st.subheader("🔑 Feature Importance & Seasonal Predictions")

    col1, col2 = st.columns(2)

    # ── Chart 5: Feature Importance (Model Coefficients) ─────
    with col1:
        import joblib, os

        
        BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'model_artifacts', 'rainfall_model.pkl')

        try:
            model = joblib.load(MODEL_PATH)

            feature_names = [
                'tempmax', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover',
                'visibility', 'is_heatwave', 'is_coldwave', 'is_humidday', 'is_windyday',
                'pressure_drop_flag', 'wind_direction_E', 'wind_direction_NE',
                'wind_direction_NONE', 'wind_direction_NW', 'wind_direction_S',
                'wind_direction_SE', 'wind_direction_SW', 'seasons_Post-Monsoon',
                'seasons_Summer'
            ]

            coefficients = model.coef_[0]
            feat_imp_df  = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
            feat_imp_df  = feat_imp_df.reindex(feat_imp_df['Coefficient'].abs().sort_values(ascending=True).index)

            fig5, ax5 = plt.subplots(figsize=(5, 6))
            colors = ['steelblue' if c > 0 else 'lightcoral' for c in feat_imp_df['Coefficient']]
            ax5.barh(feat_imp_df['Feature'], feat_imp_df['Coefficient'], color=colors)
            ax5.axvline(x=0, color='black', lw=0.8)
            ax5.set_xlabel('Coefficient Value')
            ax5.set_title('Feature Importance (Model Coefficients)\n🔵 Increases Rain   🔴 Decreases Rain')

            fig5.tight_layout()
            st.pyplot(fig5)

        except Exception as e:
            st.warning(f"⚠️ Could not load model for feature importance: {e}")

    # ── Chart 6: Predicted Rainy Days by Season ──────────────
    with col2:
        seasonal_compare = df.groupby('seasons').agg(
            Actual_Rainy=('actual_binary', 'sum'),
            Predicted_Rainy=('predicted_binary', 'sum'),
            Total_Days=('actual_binary', 'count')
        ).reset_index()

        seasonal_compare['Actual_%']    = (seasonal_compare['Actual_Rainy']    / seasonal_compare['Total_Days'] * 100).round(1)
        seasonal_compare['Predicted_%'] = (seasonal_compare['Predicted_Rainy'] / seasonal_compare['Total_Days'] * 100).round(1)

        fig6, ax6 = plt.subplots(figsize=(5, 4))
        x     = np.arange(len(seasonal_compare['seasons']))
        width = 0.35

        bars1 = ax6.bar(x - width/2, seasonal_compare['Actual_%'],    width, label='Actual %',    color='steelblue')
        bars2 = ax6.bar(x + width/2, seasonal_compare['Predicted_%'], width, label='Predicted %', color='lightcoral')

        ax6.set_xlabel('Season')
        ax6.set_ylabel('Rainy Days (%)')
        ax6.set_title('Actual vs Predicted Rainy Days % by Season')
        ax6.set_xticks(x)
        ax6.set_xticklabels(seasonal_compare['seasons'], rotation=15)
        ax6.legend()

        for bar in bars1:
            ax6.annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
        for bar in bars2:
            ax6.annotate(f'{bar.get_height():.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

        fig6.tight_layout()
        st.pyplot(fig6)

    st.markdown("---")

    # ============================================================
    # ROW 4 — Avg Rain Probability by Month + Yearly Accuracy
    # ============================================================
    st.subheader("📅 Monthly & Yearly Prediction Trends")

    col1, col2 = st.columns(2)

    # ── Chart 7: Avg Rain Probability by Month ───────────────
    with col1:
        monthly_prob = df.groupby('month')['rain_probability'].mean().reset_index()
        all_months   = pd.DataFrame({'month': range(1, 13)})
        monthly_prob = pd.merge(all_months, monthly_prob, on='month', how='left').fillna(0)

        fig7, ax7 = plt.subplots(figsize=(5, 4))
        bars = ax7.bar(monthly_prob['month'], monthly_prob['rain_probability'], color='steelblue')
        ax7.set_xlabel('Month')
        ax7.set_ylabel('Avg Rain Probability (%)')
        ax7.set_title('Average Rain Probability by Month')
        ax7.set_xticks(range(1, 13))
        ax7.set_ylim(0, 100)
        ax7.axhline(y=50, color='red', linestyle='--', lw=1, label='50% Threshold')
        ax7.legend()

        for bar in bars:
            height = bar.get_height()
            ax7.annotate(f'{height:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

        fig7.tight_layout()
        st.pyplot(fig7)

    # ── Chart 8: Yearly Accuracy Trend ───────────────────────
    with col2:
        yearly_accuracy = df.groupby('year').apply(
            lambda x: (x['actual_binary'] == x['predicted_binary']).mean() * 100
        ).reset_index()
        yearly_accuracy.columns = ['year', 'accuracy']

        fig8, ax8 = plt.subplots(figsize=(5, 4))
        ax8.plot(yearly_accuracy['year'], yearly_accuracy['accuracy'],
                 color='steelblue', marker='o', markersize=3, lw=1.5)
        ax8.axhline(y=accuracy*100, color='red', linestyle='--', lw=1,
                    label=f'Overall Accuracy ({accuracy*100:.1f}%)')
        ax8.fill_between(yearly_accuracy['year'], yearly_accuracy['accuracy'],
                         alpha=0.1, color='steelblue')
        ax8.set_xlabel('Year')
        ax8.set_ylabel('Accuracy (%)')
        ax8.set_title('Yearly Prediction Accuracy Trend')
        ax8.set_ylim(50, 100)
        ax8.legend()

        fig8.tight_layout()
        st.pyplot(fig8)

    st.markdown("---")

    # ============================================================
    # ROW 5 — Detailed Prediction Table
    # ============================================================
    st.subheader("🗂️ Detailed Prediction Table")

    display_cols = ['datetime', 'seasons', 'is_rainy_day', 'predicted_rain', 'rain_probability']
    available_cols = [col for col in display_cols if col in df.columns]

    # Color coding
    def highlight_prediction(row):
        if row['is_rainy_day'] == row['predicted_rain']:
            return ['background-color: #d4edda'] * len(row)  # green for correct
        else:
            return ['background-color: #f8d7da'] * len(row)  # red for wrong

    st.dataframe(
        df[available_cols].sort_values('datetime', ascending=False).head(100),
        use_container_width=True
    )
    st.caption("Showing last 100 records. Green = Correct Prediction, Red = Wrong Prediction")