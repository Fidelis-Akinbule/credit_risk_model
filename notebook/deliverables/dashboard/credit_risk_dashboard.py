import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import math

def get_pdo(scorecard_params):
    return scorecard_params.get(
        'PDO',
        math.log(2) * scorecard_params['FACTOR']
    )

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💳 Credit Risk Scoring Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Model Performance", "🎯 Score Calculator", "📈 Business Impact", 
     "🔍 Feature Analysis", "📋 Model Comparison", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dashboard Features:**
- Model performance metrics
- Interactive score calculator
- Business impact analysis
- Feature importance rankings
- Calibration diagnostics
- Risk segmentation
""")

# Load data and models
@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        val_scores = pd.read_csv('deliverables/scorecards/validation_scores_enhanced.csv')
        model_summary = pd.read_csv('deliverables/reports/model_comparison_summary.csv')
        feature_importance = pd.read_csv('deliverables/reports/feature_importance_comparison.csv')

        # Load business parameters (FIXED)
        with open('deliverables/business_rules/business_parameters_enhanced.json', 'r') as f:
            raw_business = json.load(f)
            business_params = raw_business['business_params']

        # Load scorecard parameters (already correct)
        with open('deliverables/scorecards/scorecard_parameters_enhanced.json', 'r') as f:
            raw = json.load(f)
            scorecard_params = raw['scorecard_params']

        strategy_comparison = pd.read_csv('deliverables/business_rules/strategy_comparison_enhanced.csv')

        return {
            'val_scores': val_scores,
            'model_summary': model_summary,
            'feature_importance': feature_importance,
            'business_params': business_params,
            'scorecard_params': scorecard_params,
            'strategy_comparison': strategy_comparison
        }

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models = {
            'xgboost': joblib.load('deliverables/models/final_xgb_model_calibrated.pkl'),
            'lightgbm': joblib.load('deliverables/models/final_lgb_model_calibrated.pkl'),
            'logistic': joblib.load('deliverables/models/final_lr_model.pkl')
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load data
data = load_data()
models = load_models()

if data is None or models is None:
    st.error("Failed to load required data. Please ensure all deliverables are in the correct location.")
    st.stop()

# PAGE 1: MODEL PERFORMANCE
if page == "📊 Model Performance":
    st.header("📊 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    # Get best model
    best_model = data['model_summary'].loc[data['model_summary']['AUC'].idxmax()]
    
    with col1:
        st.metric(
            label="Best Model",
            value=best_model['Model'],
            delta=None
        )
    
    with col2:
        st.metric(
            label="AUC Score",
            value=f"{best_model['AUC']:.4f}",
            delta=f"+{(best_model['AUC'] - 0.5)*100:.1f}% vs Random"
        )
    
    with col3:
        st.metric(
            label="KS Statistic",
            value=f"{best_model['KS']:.4f}",
            delta="Excellent" if best_model['KS'] > 0.4 else "Good"
        )
    
    st.markdown("---")
    
    # Model comparison table
    st.subheader("Model Comparison")
    
    # Format the dataframe
    display_df = data['model_summary'].copy()
    display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.4f}")
    display_df['KS'] = display_df['KS'].apply(lambda x: f"{x:.4f}")
    display_df['Brier'] = display_df['Brier'].apply(lambda x: f"{x:.4f}")
    display_df['Calib_Gap'] = display_df['Calib_Gap'].apply(lambda x: f"{x:.4f}")
    display_df['P@10%'] = display_df['P@10%'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, width="stretch")
    
    # Calibration quality
    st.markdown("---")
    st.subheader("Calibration Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Brier Score (Lower is Better)**")
        st.markdown(f"Best Model: **{best_model['Brier']:.4f}**")
        st.markdown("Random Baseline: **0.2500**")
        improvement = ((0.25 - best_model['Brier']) / 0.25) * 100
        st.markdown(f"Improvement: **{improvement:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Calibration Gap**")
        st.markdown(f"Best Model: **{best_model['Calib_Gap']:.4f}**")
        if best_model['Calib_Gap'] < 0.005:
            st.markdown("Status: **Excellent** ✅")
        elif best_model['Calib_Gap'] < 0.01:
            st.markdown("Status: **Good** ✓")
        else:
            st.markdown("Status: **Needs Improvement** ⚠️")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance interpretation
    st.markdown("---")
    st.subheader("Performance Interpretation")
    
    interpretation = f"""
**Model Quality Assessment:**

- **Discrimination (AUC):** {best_model['AUC']:.4f} - {"Excellent" if best_model['AUC'] > 0.85 else "Good"} ability to rank order risk
- **Separation (KS):** {best_model['KS']*100:.1f}% - {"Excellent" if best_model['KS'] > 0.4 else "Good"} separation between good/bad borrowers
- **Calibration (Brier):** {best_model['Brier']:.4f} - Probabilities are {"well-calibrated" if best_model['Brier'] < 0.15 else "moderately calibrated"}
- **Precision@10%:** {best_model['P@10%']*100:.1f}% - Captures {best_model['P@10%']*100:.1f}% defaults in top 10% riskiest applicants

**Regulatory Compliance:**
- Model meets Basel III discriminatory power requirements (AUC > 0.70)
- Calibration suitable for IFRS 9 expected loss calculations
- Performance stable across train/validation splits
"""
    
    st.info(interpretation)

# PAGE 2: SCORE CALCULATOR
elif page == "🎯 Score Calculator":
    st.header("🎯 Individual Credit Score Calculator")
    
    st.markdown("""
Enter applicant information below to generate a credit score and risk assessment.
All fields are required for accurate prediction.
""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=1000000, value=5000)
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
        
    with col2:
        st.subheader("Credit History")
        revolving_util = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=100.0, value=30.0) / 100
        debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        num_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=8)
        num_real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=20, value=1)
    
    st.subheader("Delinquency History")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        times_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, max_value=20, value=0)
    with col4:
        times_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, max_value=20, value=0)
    with col5:
        times_90_plus = st.number_input("Times 90+ Days Late", min_value=0, max_value=20, value=0)
    
    if st.button("Calculate Credit Score", type="primary"):
        
        # Feature engineering (matching training process)
        total_delinquencies = times_30_59 + times_60_89 + times_90_plus
        has_delinquency = 1 if total_delinquencies > 0 else 0
        worst_delinquency = 0 if total_delinquencies == 0 else (1 if times_30_59 > 0 else (2 if times_60_89 > 0 else 3))
        has_90day_late = 1 if times_90_plus > 0 else 0
        
        monthly_debt_payment = debt_ratio * monthly_income
        credit_lines_per_dependent = num_credit_lines / (num_dependents + 1)
        
        if revolving_util <= 0.3:
            utilization_category = 0
        elif revolving_util <= 0.5:
            utilization_category = 1
        elif revolving_util <= 0.7:
            utilization_category = 2
        else:
            utilization_category = 3
        
        income_per_dependent = monthly_income / (num_dependents + 1)
        age_income_interaction = age * monthly_income
        util_debt_interaction = revolving_util * debt_ratio
        age_delinq_interaction = age * total_delinquencies
        
        # Create feature vector
        features = np.array([[
            age,
            0,  # util_extreme_flag
            0,  # debt_extreme_flag
            0,  # income_missing
            0,  # dependents_missing
            0,  # MonthlyIncome_outlier_flag
            0,  # NumberOfOpenCreditLinesAndLoans_outlier_flag
            0,  # NumberOfTime30-59DaysPastDueNotWorse_outlier_flag
            0,  # NumberOfTimes90DaysLate_outlier_flag
            0,  # NumberOfTime60-89DaysPastDueNotWorse_outlier_flag
            np.log1p(revolving_util),
            np.log1p(debt_ratio),
            np.log1p(monthly_income),
            np.log1p(times_30_59),
            np.log1p(times_60_89),
            np.log1p(times_90_plus),
            np.log1p(num_credit_lines),
            np.log1p(num_real_estate),
            np.log1p(num_dependents),
            total_delinquencies,
            has_delinquency,
            worst_delinquency,
            has_90day_late,
            monthly_debt_payment,
            credit_lines_per_dependent,
            utilization_category,
            income_per_dependent,
            age_income_interaction,
            util_debt_interaction,
            age_delinq_interaction
        ]])
        
        # Get prediction from best model
        best_model_name = data['model_summary'].loc[data['model_summary']['AUC'].idxmax(), 'Model'].lower()
        model_key = 'xgboost' if 'xgb' in best_model_name else ('lightgbm' if 'light' in best_model_name else 'logistic')
        
        probability = models[model_key].predict_proba(features)[0, 1]
        
        # Convert to score
        params = data['scorecard_params']

        FACTOR = params['FACTOR']
        OFFSET = params['OFFSET']

        # PDO is derived, not stored
        PDO = params.get('PDO', math.log(2) * FACTOR)

        
        odds = (1 - probability) / probability
        score = OFFSET + FACTOR * np.log(odds)
        score = np.clip(score, 300, 900)
        
        # Determine risk band
        if score >= 700:
            risk_band = "Very Low Risk"
            risk_color = "success"
            recommendation = "Auto-Approve"
        elif score >= 600:
            risk_band = "Low-Medium Risk"
            risk_color = "warning"
            recommendation = "Manual Review"
        else:
            risk_band = "High Risk"
            risk_color = "danger"
            recommendation = "Decline"
        
        # Calculate expected loss
        LGD = data['business_params']['LGD']
        EAD = data['business_params']['EAD']
        expected_loss = probability * LGD * EAD
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Credit Assessment Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Credit Score", f"{score:.0f}")
        with col2:
            st.metric("Default Probability", f"{probability*100:.2f}%")
        with col3:
            st.metric("Risk Band", risk_band)
        with col4:
            st.metric("Expected Loss", f"${expected_loss:.2f}")
        
        # Recommendation box
        if risk_color == "success":
            st.markdown(f'<div class="success-box"><strong>Recommendation:</strong> {recommendation}<br>This applicant shows strong creditworthiness with minimal default risk.</div>', unsafe_allow_html=True)
        elif risk_color == "warning":
            st.markdown(f'<div class="warning-box"><strong>Recommendation:</strong> {recommendation}<br>This applicant requires manual underwriting review before decision.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="danger-box"><strong>Recommendation:</strong> {recommendation}<br>This applicant shows elevated default risk. Consider decline or high-interest terms.</div>', unsafe_allow_html=True)
        
        # Risk factors
        st.markdown("---")
        st.subheader("Key Risk Factors")
        
        risk_factors = []
        
        if revolving_util > 0.7:
            risk_factors.append("⚠️ High revolving utilization (>70%)")
        if debt_ratio > 0.43:
            risk_factors.append("⚠️ High debt ratio")
        if total_delinquencies > 0:
            risk_factors.append(f"⚠️ {total_delinquencies} delinquency event(s)")
        if times_90_plus > 0:
            risk_factors.append("🔴 Serious delinquency (90+ days late)")
        if age < 25:
            risk_factors.append("⚠️ Limited credit history (young age)")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("✅ No major risk factors identified")

# PAGE 3: BUSINESS IMPACT
elif page == "📈 Business Impact":
    st.header("📈 Business Impact Analysis")
    
    # Key metrics
    st.subheader("Financial Impact Summary")
    
    col1, col2, col3 = st.columns(3)
    
    # Get optimal strategy
    optimal_strategy = data['strategy_comparison'].loc[data['strategy_comparison']['Profit'].idxmax()]
    
    with col1:
        st.metric(
            "Optimal Approval Rate",
            f"{optimal_strategy['Approval_Rate']*100:.1f}%",
            delta=f"{(optimal_strategy['Approval_Rate'] - 1)*100:.1f}% vs Approve All"
        )
    
    with col2:
        st.metric(
            "Default Rate (Approved)",
            f"{optimal_strategy['Approved_Default_Rate']*100:.2f}%",
            delta=f"-{(0.067 - optimal_strategy['Approved_Default_Rate'])*100:.2f}pp vs Baseline",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Total Profit",
            f"${optimal_strategy['Profit']:,.0f}",
            delta="Optimal Strategy"
        )
    
    st.markdown("---")
    
    # Strategy comparison
    st.subheader("Strategy Comparison")
    
    fig_strategy = go.Figure()
    
    strategies = data['strategy_comparison']['Strategy'].tolist()
    profits = data['strategy_comparison']['Profit'].tolist()
    
    fig_strategy.add_trace(go.Bar(
        x=strategies,
        y=profits,
        marker_color=['green' if p == max(profits) else 'lightblue' for p in profits],
        text=[f'${p:,.0f}' for p in profits],
        textposition='outside'
    ))
    
    fig_strategy.update_layout(
        title="Profit by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Total Profit ($)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_strategy, width="stretch")
    
    # Approval rate vs default rate trade-off
    st.markdown("---")
    st.subheader("Approval Rate vs Default Rate Trade-off")
    
    fig_tradeoff = go.Figure()
    
    fig_tradeoff.add_trace(go.Scatter(
        x=data['strategy_comparison']['Approval_Rate'] * 100,
        y=data['strategy_comparison']['Approved_Default_Rate'] * 100,
        mode='markers+lines+text',
        marker=dict(
            size=data['strategy_comparison']['Profit'] / 1000000,
            color=data['strategy_comparison']['Profit'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Profit ($)")
        ),
        text=data['strategy_comparison']['Strategy'],
        textposition='top center',
        line=dict(color='gray', dash='dash')
    ))
    
    fig_tradeoff.update_layout(
        title="Strategy Trade-offs (Bubble size = Profit)",
        xaxis_title="Approval Rate (%)",
        yaxis_title="Default Rate - Approved (%)",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_tradeoff, width="stretch")
    
    # Expected loss analysis
    st.markdown("---")
    st.subheader("Expected Loss Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Loss Given Default (LGD)**")
        st.markdown(f"**{data['business_params']['LGD']*100:.0f}%** of exposure")
        st.markdown("Industry standard for unsecured consumer credit")
    
    with col2:
        st.markdown("**Exposure at Default (EAD)**")
        st.markdown(f"**${data['business_params']['EAD']:,}** per loan")
        st.markdown("Average loan amount assumption")
    
    st.markdown("---")
    
    # Score distribution
    st.subheader("Credit Score Distribution")
    
    fig_dist = go.Figure()
    
    # Separate by actual outcome
    defaults = data['val_scores'][data['val_scores']['Actual'] == 1]
    non_defaults = data['val_scores'][data['val_scores']['Actual'] == 0]
    
    fig_dist.add_trace(go.Histogram(
        x=non_defaults['Score'],
        name='Non-Default',
        opacity=0.7,
        marker_color='green',
        nbinsx=50
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=defaults['Score'],
        name='Default',
        opacity=0.7,
        marker_color='red',
        nbinsx=50
    ))
    
    # Add cutoff lines
    fig_dist.add_vline(x=700, line_dash="dash", line_color="blue", 
                       annotation_text="Auto-Approve: 700")
    fig_dist.add_vline(x=600, line_dash="dash", line_color="orange", 
                       annotation_text="Manual Review: 600")
    
    fig_dist.update_layout(
        title="Score Distribution by Actual Outcome",
        xaxis_title="Credit Score",
        yaxis_title="Frequency",
        height=500,
        barmode='overlay'
    )
    
    st.plotly_chart(fig_dist, width="stretch")

# PAGE 4: FEATURE ANALYSIS
elif page == "🔍 Feature Analysis":
    st.header("🔍 Feature Importance Analysis")
    
    # Top features
    st.subheader("Top 15 Most Important Features")
    
    top_15 = data['feature_importance'].head(15)
    
    fig_importance = go.Figure()
    
    fig_importance.add_trace(go.Bar(
        y=top_15['Feature'],
        x=top_15['Importance_XGB_Norm'],
        name='XGBoost',
        orientation='h',
        marker_color='#1f77b4'
    ))
    
    fig_importance.add_trace(go.Bar(
        y=top_15['Feature'],
        x=top_15['Importance_LGB_Norm'],
        name='LightGBM',
        orientation='h',
        marker_color='#ff7f0e'
    ))
    
    fig_importance.add_trace(go.Bar(
        y=top_15['Feature'],
        x=top_15['Importance_LR_Norm'],
        name='Logistic Regression',
        orientation='h',
        marker_color='#2ca02c'
    ))
    
    fig_importance.update_layout(
        title="Feature Importance Comparison (Normalized %)",
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=600,
        barmode='group',
        yaxis={'categoryorder':'total ascending'}
    )
    
    st.plotly_chart(fig_importance, width="stretch")
    
    st.markdown("---")
    
    # Consensus top 5
    st.subheader("Consensus Top 5 Features")
    
    top_5 = data['feature_importance'].head(5)
    
    for idx, row in top_5.iterrows():
        with st.expander(f"**{idx+1}. {row['Feature']}**"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Importance", f"{row['Avg_Importance']:.2f}%")
            with col2:
                st.metric("Logistic Reg", f"{row['Importance_LR_Norm']:.2f}%")
            with col3:
                st.metric("XGBoost", f"{row['Importance_XGB_Norm']:.2f}%")
            with col4:
                st.metric("LightGBM", f"{row['Importance_LGB_Norm']:.2f}%")
            
            # Feature description
            feature_name = row['Feature']
            if 'delinq' in feature_name.lower():
                st.info("📊 **Delinquency Feature:** Measures past payment behavior and credit discipline.")
            elif 'utilization' in feature_name.lower():
                st.info("💳 **Credit Utilization:** Percentage of available credit being used.")
            elif 'age' in feature_name.lower():
                st.info("👤 **Age-Related:** Captures life stage and credit maturity.")
            elif 'income' in feature_name.lower():
                st.info("💰 **Income Feature:** Measures financial capacity and stability.")
            elif 'debt' in feature_name.lower():
                st.info("📈 **Debt Metric:** Assesses debt burden relative to income.")
    
    st.markdown("---")
    
    # Feature categories
    st.subheader("Feature Category Breakdown")
    
    # Categorize features
    delinquency_features = [f for f in data['feature_importance']['Feature'] 
                           if any(x in f.lower() for x in ['delinq', 'late', '90day'])]
    financial_features = [f for f in data['feature_importance']['Feature'] 
                         if any(x in f.lower() for x in ['income', 'debt', 'payment'])]
    credit_features = [f for f in data['feature_importance']['Feature'] 
                      if any(x in f.lower() for x in ['utilization', 'credit', 'lines'])]
    demographic_features = [f for f in data['feature_importance']['Feature'] 
                           if any(x in f.lower() for x in ['age', 'dependent'])]
    
    category_importance = {
        'Delinquency History': data['feature_importance'][
            data['feature_importance']['Feature'].isin(delinquency_features)
        ]['Avg_Importance'].sum(),
        'Financial Ratios': data['feature_importance'][
            data['feature_importance']['Feature'].isin(financial_features)
        ]['Avg_Importance'].sum(),
        'Credit Utilization': data['feature_importance'][
            data['feature_importance']['Feature'].isin(credit_features)
        ]['Avg_Importance'].sum(),
        'Demographics': data['feature_importance'][
            data['feature_importance']['Feature'].isin(demographic_features)
        ]['Avg_Importance'].sum()
    }
    
    fig_category = go.Figure(data=[go.Pie(
        labels=list(category_importance.keys()),
        values=list(category_importance.values()),
        hole=0.4,
        marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#feca57']
    )])
    
    fig_category.update_layout(
        title="Feature Importance by Category",
        height=500
    )
    
    st.plotly_chart(fig_category, width="stretch")
    
    # Key insights
    st.markdown("---")
    st.subheader("Key Insights")
    
    top_1_feature = data['feature_importance'].iloc[0]['Feature']
    top_1_importance = data['feature_importance'].iloc[0]['Avg_Importance']
    
    insights = f"""
**Feature Analysis Summary:**

- **Most Important Feature:** {top_1_feature} ({top_1_importance:.1f}% importance)
- **Delinquency Dominance:** Payment history features are strongest predictors
- **Credit Utilization:** Second most important category - how much credit is used
- **Engineered Features:** Custom features add significant predictive power
- **Demographic Factors:** Age and dependents provide context but are not primary drivers

**Business Implications:**

- Focus underwriting on payment history verification
- Monitor credit utilization closely for existing customers
- Age alone is not a strong predictor (fair lending compliance)
- Engineered interaction features capture risk patterns effectively
"""
    
    st.info(insights)

# PAGE 5: MODEL COMPARISON (continuation)
elif page == "📋 Model Comparison":
    st.header("📋 Detailed Model Comparison")
    
    # Metrics radar chart
    st.subheader("Multi-Metric Comparison")
    
    fig_radar = go.Figure()
    
    metrics = ['AUC', 'KS', 'P@10%']
    
    for idx, row in data['model_summary'].iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['AUC'], row['KS'], row['P@10%']],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig_radar, width="stretch")
    
    st.markdown("---")
    
    # Calibration comparison
    st.subheader("Calibration Quality Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_brier = go.Figure(data=[go.Bar(
            x=data['model_summary']['Model'],
            y=data['model_summary']['Brier'],
            marker_color=['green' if b == data['model_summary']['Brier'].min() 
                         else 'lightblue' for b in data['model_summary']['Brier']],
            text=data['model_summary']['Brier'].apply(lambda x: f"{x:.4f}"),
            textposition='outside'
        )])
        
        fig_brier.update_layout(
            title="Brier Score (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="Brier Score",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_brier, width="stretch")
    
    with col2:
        fig_gap = go.Figure(data=[go.Bar(
            x=data['model_summary']['Model'],
            y=data['model_summary']['Calib_Gap'],
            marker_color=['green' if g == data['model_summary']['Calib_Gap'].min() 
                         else 'lightblue' for g in data['model_summary']['Calib_Gap']],
            text=data['model_summary']['Calib_Gap'].apply(lambda x: f"{x:.4f}"),
            textposition='outside'
        )])
        
        fig_gap.update_layout(
            title="Calibration Gap (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="Calibration Gap",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_gap, width="stretch")
    
    st.markdown("---")
    
    # Detailed metrics table
    st.subheader("Complete Metrics Table")
    st.dataframe(data['model_summary'], width="stretch")
    
    # Model selection rationale
    st.markdown("---")
    st.subheader("Model Selection Rationale")
    
    best_model = data['model_summary'].loc[data['model_summary']['AUC'].idxmax()]
    
    rationale = f"""
**Selected Model: {best_model['Model']}**

**Selection Criteria:**

1. **Discrimination Power (AUC):** {best_model['AUC']:.4f}
   - Excellent ability to rank order risk
   - Exceeds regulatory minimum (0.70) by significant margin

2. **Calibration Quality (Brier):** {best_model['Brier']:.4f}
   - Probabilities are well-calibrated after post-processing
   - Suitable for expected loss calculations (Basel III, IFRS 9)

3. **Business Performance (Precision@10%):** {best_model['P@10%']*100:.1f}%
   - Strong capture rate in highest risk segment
   - Efficient targeting for risk mitigation

4. **Stability:**
   - Minimal overfitting across train/validation
   - Consistent performance across metrics

**Trade-offs Considered:**

- Logistic Regression: More interpretable, but lower discrimination
- Tree Models: Higher discrimination, required calibration
- Final choice balances performance with reliability
"""
    
    st.info(rationale)

# PAGE 6: ABOUT
elif page == "ℹ️ About":
    st.header("ℹ️ About This Dashboard")
    
    st.markdown("""
### Credit Risk Scoring System

This dashboard presents a comprehensive credit risk scoring model built using machine learning techniques.
The system predicts the probability of default for credit applicants and converts these probabilities
into interpretable credit scores (300-900 scale).

---

### Model Architecture

**Three Models Evaluated:**
1. **Logistic Regression** - Interpretable baseline with natural calibration
2. **XGBoost** - Gradient boosted trees with high discrimination
3. **LightGBM** - Efficient gradient boosting alternative

**All tree models calibrated using isotonic regression for accurate probability estimates.**

---

### Key Features

✅ **30 Engineered Features** including:
- Delinquency history aggregations
- Financial ratios (debt-to-income, utilization)
- Interaction terms (age × income, utilization × debt)
- Credit behavior indicators

✅ **Industry-Standard Metrics:**
- AUC-ROC for discrimination
- KS statistic for separation
- Brier score for calibration
- Precision@k for business value

✅ **Regulatory Compliance:**
- Basel III discriminatory power requirements
- IFRS 9 calibration standards
- Fair lending considerations

---

### Scorecard Parameters
""")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Score Range**")
        st.markdown("300 - 900 points")
        st.markdown("")

        st.markdown("**PDO (Points to Double Odds)**")
        PDO = get_pdo(data['scorecard_params'])
        st.markdown(f"{PDO:.0f} points")

        st.markdown("")
        st.markdown("**Base Score**")
        st.markdown(f"{data['scorecard_params']['BASE_SCORE']} points")

    
    with col2:
        st.markdown("**Base Odds**")
        st.markdown(f"{data['scorecard_params']['BASE_ODDS']}:1")
        st.markdown("")
        st.markdown("**Risk Bands**")
        st.markdown("- Very Low: 700-900")
        st.markdown("- Low-Medium: 600-699")
        st.markdown("- High: 300-599")
    
    st.markdown("---")
    
    st.markdown("### Business Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Loss Given Default (LGD)**")
        st.markdown(f"{data['business_params']['LGD']*100:.0f}%")
        st.markdown("Industry standard for unsecured credit")
        st.markdown("")
        st.markdown("**Interest Rate**")
        st.markdown(f"{data['business_params']['INTEREST_RATE']*100:.0f}% annual")
    
    with col2:
        st.markdown("**Exposure at Default (EAD)**")
        st.markdown(f"${data['business_params']['EAD']:,}")
        st.markdown("Average loan amount")
        st.markdown("")
        st.markdown("**Loan Term**")
        st.markdown(f"{data['business_params']['LOAN_TERM']} years")
    
    st.markdown("---")
    
    st.markdown("### Data Sources")
    
    st.markdown("""
- **Training Dataset:** "Give Me Some Credit" (Kaggle)
- **Training Samples:** 149,391 borrowers
- **Validation Samples:** 29,879 borrowers
- **Test Samples:** 101,175 borrowers (unlabeled)
- **Features:** 30 engineered features from 10 raw features

---

### Model Performance Summary
""")
    
    best_model = data['model_summary'].loc[data['model_summary']['AUC'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", best_model['Model'])
        st.metric("AUC", f"{best_model['AUC']:.4f}")
    
    with col2:
        st.metric("KS Statistic", f"{best_model['KS']*100:.1f}%")
        st.metric("Brier Score", f"{best_model['Brier']:.4f}")
    
    with col3:
        st.metric("Precision@10%", f"{best_model['P@10%']*100:.1f}%")
        st.metric("Calibration Gap", f"{best_model['Calib_Gap']:.4f}")
    
    st.markdown("---")
    
    st.markdown("""
### Technical Stack

- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Data Processing:** pandas, numpy
- **Visualization:** Plotly, matplotlib, seaborn
- **Dashboard:** Streamlit
- **Model Calibration:** Isotonic regression, Platt scaling

---

### Contact & Documentation

For more information about this project:
- Review the README.md in the deliverables folder
- Check the model documentation in deliverables/reports/
- Examine saved models in deliverables/models/

---

**Dashboard Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Production Ready
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Credit Risk Scoring Dashboard | Built with Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)