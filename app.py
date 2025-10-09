import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Milling Machine Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for industrial theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --dark-gray: #2C3E50;
        --steel-blue: #34495E;
        --safety-yellow: #F39C12;
        --light-gray: #ECF0F1;
        --warning-red: #E74C3C;
        --success-green: #27AE60;
    }
    
    /* Main container styling */
    .main {
        background-color: #1a1a1a;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 5px solid #F39C12;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        color: #F39C12;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .subtitle {
        color: #ECF0F1;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #F39C12;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        color: #BDC3C7;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #F39C12;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-unit {
        color: #ECF0F1;
        font-size: 1rem;
        margin-left: 0.5rem;
    }
    
    /* Status indicators */
    .status-healthy {
        background: #27AE60;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-warning {
        background: #F39C12;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .status-critical {
        background: #E74C3C;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2C3E50;
    }
    
    /* Input labels */
    .stSlider label, .stNumberInput label {
        color: #F39C12 !important;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(243, 156, 18, 0.1);
        border: 2px solid #F39C12;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-title {
        color: #F39C12;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .warning-text {
        color: #ECF0F1;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Info box */
    .info-box {
        background: rgba(52, 73, 94, 0.3);
        border: 2px solid #34495E;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.75rem 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #E67E22 0%, #D35400 100%);
        box-shadow: 0 4px 8px rgba(243, 156, 18, 0.4);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #F39C12, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Calculate derived features
def calculate_derived_features(torque, rpm, process_temp, air_temp):
    temperature_difference = process_temp - air_temp
    mechanical_power = (torque * rpm * 2 * np.pi) / 60
    return temperature_difference, mechanical_power

# Get status based on prediction
def get_status_html(prediction, probability):
    if prediction == 0:
        if probability < 0.7:
            status_class = "status-warning"
            status_text = "Caution"
        else:
            status_class = "status-healthy"
            status_text = "Healthy"
    else:
        status_class = "status-critical"
        status_text = "Failure Risk"
    
    return f'<div class="{status_class}">{status_text}</div>'

# Create gauge chart
def create_gauge_chart(value, title, max_value=100):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'color': '#ECF0F1', 'size': 18}},
        number = {'font': {'color': '#F39C12', 'size': 32}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickcolor': '#ECF0F1'},
            'bar': {'color': '#F39C12'},
            'bgcolor': '#34495E',
            'borderwidth': 2,
            'bordercolor': '#2C3E50',
            'steps': [
                {'range': [0, max_value*0.6], 'color': '#27AE60'},
                {'range': [max_value*0.6, max_value*0.85], 'color': '#F39C12'},
                {'range': [max_value*0.85, max_value], 'color': '#E74C3C'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ECF0F1'},
        height=250
    )
    
    return fig

# Create bar chart for feature importance
def create_feature_chart(features_dict):
    fig = go.Figure(data=[
        go.Bar(
            x=list(features_dict.values()),
            y=list(features_dict.keys()),
            orientation='h',
            marker=dict(
                color=['#F39C12', '#E67E22', '#D35400', '#C0392B', '#A93226', '#922B21'],
                line=dict(color='#2C3E50', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title={'text': 'Input Parameters', 'font': {'color': '#F39C12', 'size': 20}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(44, 62, 80, 0.3)',
        font={'color': '#ECF0F1'},
        xaxis={'gridcolor': '#34495E'},
        yaxis={'gridcolor': '#34495E'},
        height=400
    )
    
    return fig

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">⚙️ Milling Machine Health Monitor</h1>
    <p class="subtitle">Real-time Predictive Maintenance System</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error("⚠️ Model could not be loaded. Please ensure 'xgboost_model.pkl' is in the correct directory.")
    st.stop()

# Create main tabs
tab1, tab2 = st.tabs(["🔧 Machine Health", "📊 Dataset Statistics"])

with tab2:
    try:
        # Load dataset
        df = pd.read_csv("ai4i2020.csv")
        # Focused columns
        focus_cols = [
            "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"
        ]
        df_focus = df[focus_cols].copy()
        
        # Data quality
        total_samples = len(df_focus)
        num_features = len(focus_cols) - 1
        num_targets = 1
        missing = df_focus.isnull().sum().sum()
        data_quality = 100 * (1 - missing / (df_focus.size))
        
        # Overview metrics
        st.markdown("#### 📊 Dataset Overview")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Total Samples", total_samples)
        colB.metric("Features", num_features)
        colC.metric("Target Variables", num_targets)
        colD.metric("Data Quality", f"{data_quality:.1f}%")
        st.markdown("---")
        
        # Statistical summary
        st.markdown("#### 📋 Statistical Summary")
        
        # Select specific variables for statistical summary
        summary_cols = [
            "Type", 
            "Air temperature [K]", 
            "Process temperature [K]", 
            "Rotational speed [rpm]", 
            "Torque [Nm]", 
            "Tool wear [min]",
            "Machine failure"
        ]
        
        # Create separate summaries for categorical and numerical variables
        categorical_cols = ["Type", "Machine failure"]
        numerical_cols = [col for col in summary_cols if col not in categorical_cols]
        
        # Display numerical variables statistics
        st.markdown("##### Numerical Variables Statistics")
        numerical_stats = df_focus[numerical_cols].describe()
        st.dataframe(numerical_stats, use_container_width=True)
        
        # Display categorical variables statistics
        st.markdown("##### Categorical Variables Distribution")
        cat_col1, cat_col2 = st.columns(2)
        
        with cat_col1:
            st.markdown("**Machine Type Distribution:**")
            type_counts = df_focus["Type"].value_counts()
            type_percentages = df_focus["Type"].value_counts(normalize=True) * 100
            type_summary = pd.DataFrame({
                'Count': type_counts,
                'Percentage': type_percentages.round(2)
            })
            st.dataframe(type_summary, use_container_width=True)
        
        with cat_col2:
            st.markdown("**Machine Failure Distribution:**")
            failure_counts = df_focus["Machine failure"].value_counts()
            failure_percentages = df_focus["Machine failure"].value_counts(normalize=True) * 100
            failure_summary = pd.DataFrame({
                'Count': failure_counts,
                'Percentage': failure_percentages.round(2)
            })
            failure_summary.index = ['Healthy (0)', 'Failure (1)']
            st.dataframe(failure_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Target Distributions", "Feature Distributions", "Correlations", "Normalized Comparison"
        ])
        
        with viz_tab1:
            st.markdown("##### Machine Failure Distribution")
            failure_counts = df_focus["Machine failure"].value_counts()
            fig = px.bar(
                x=['Healthy (0)', 'Failure (1)'], 
                y=[failure_counts[0], failure_counts[1]], 
                color=['Healthy (0)', 'Failure (1)'],
                color_discrete_map={'Healthy (0)': "#27AE60", 'Failure (1)': "#E74C3C"},
                title="Machine Failure Distribution"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'},
                xaxis={'gridcolor': '#34495E'},
                yaxis={'gridcolor': '#34495E'},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with viz_tab2:
            st.markdown("##### Feature Distributions")
            for col in focus_cols[:-1]:  # Exclude target variable
                if col == "Type":
                    # Handle categorical Type column
                    type_counts = df_focus[col].value_counts()
                    fig = px.bar(
                        x=type_counts.index, 
                        y=type_counts.values, 
                        title=f"{col} Distribution",
                        color=type_counts.index,
                        color_discrete_map={'L': "#3498DB", 'M': "#F39C12", 'H': "#E74C3C"}
                    )
                else:
                    # Handle numerical columns
                    fig = px.histogram(df_focus, x=col, nbins=30, title=f"{col} Distribution")
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(44, 62, 80, 0.3)',
                    font={'color': '#ECF0F1'},
                    xaxis={'gridcolor': '#34495E'},
                    yaxis={'gridcolor': '#34495E'},
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with viz_tab3:
            st.markdown("##### Feature-Target Correlations")
            # Create numeric version of dataframe for correlation
            df_numeric = df_focus.copy()
            # Convert Type to numeric (L=0, M=1, H=2)
            df_numeric['Type'] = df_numeric['Type'].map({'L': 0, 'M': 1, 'H': 2})
            
            corr = df_numeric.corr()
            fig = px.imshow(
                corr, 
                text_auto=True, 
                color_continuous_scale="RdBu", 
                aspect="auto",
                title="Feature-Target Correlation Matrix"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with viz_tab4:
            st.markdown("##### Normalized Feature Comparison")
            # Create normalized version for comparison
            norm_df = df_focus.copy()
            
            # Normalize only numerical features
            numerical_cols = [col for col in focus_cols[:-1] if col != "Type"]
            for col in numerical_cols:
                norm_df[col] = (norm_df[col] - norm_df[col].mean()) / norm_df[col].std()
            
            # Create box plots for numerical features
            fig = px.box(
                norm_df, 
                y=numerical_cols, 
                points="outliers", 
                title="Normalized Feature Comparison (Numerical Features Only)"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'},
                xaxis={'gridcolor': '#34495E'},
                yaxis={'gridcolor': '#34495E'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        
        # Data Insights
        st.markdown("#### 💡 Data Insights")
        info1, info2 = st.columns(2)
        
        with info1:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #F39C12;">Scaling Strategy</h4>
                <p style="color: #ECF0F1;">Features are normalized using standard deviation scaling for fair comparison and visualization. This helps highlight outliers and trends across different units.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with info2:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #F39C12;">Score Interpretation</h4>
                <p style="color: #ECF0F1;">Performance scores are interpreted as: <br> <b>Excellent</b> (&gt;90%), <b>Good</b> (80-90%), <b>Average</b> (60-80%), <b>Poor</b> (&lt;60%). Use these to guide maintenance actions.</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Dataset error: {e}. Please ensure 'ai4i2020.csv' is present and formatted correctly.")

with tab1:
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 🔧 Machine Parameters")
        st.markdown("---")
        
        st.markdown("#### Primary Inputs")
        
        # Machine Type selection
        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            index=1,
            help="Machine quality variant: L (Low), M (Medium), H (High)"
        )
        
        torque = st.slider(
            "Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1,
            help="Operating torque of the milling machine"
        )
        
        rpm = st.slider(
            "Rotational Speed (rpm)",
            min_value=0,
            max_value=3000,
            value=1500,
            step=10,
            help="Spindle rotation speed"
        )
        
        process_temp = st.slider(
            "Process Temperature (K)",
            min_value=273.0,
            max_value=373.0,
            value=310.0,
            step=0.1,
            help="Temperature during machining process"
        )
        
        air_temp = st.slider(
            "Air Temperature (K)",
            min_value=273.0,
            max_value=323.0,
            value=298.0,
            step=0.1,
            help="Ambient air temperature"
        )
        
        tool_wear = st.slider(
            "Tool Wear (min)",
            min_value=0,
            max_value=300,
            value=100,
            step=1,
            help="Tool wear time in minutes"
        )
        
        st.markdown("---")
        
        # Calculate derived features
        temp_diff, mech_power = calculate_derived_features(torque, rpm, process_temp, air_temp)
        
        st.markdown("#### 📊 Derived Features")
        st.metric("Temperature Difference", f"{temp_diff:.2f} K")
        st.metric("Mechanical Power", f"{mech_power:.2f} W")
        
        st.markdown("---")
        
        predict_button = st.button("🔍 ANALYZE MACHINE HEALTH", use_container_width=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Current Operating Conditions")
        
        # Create feature visualization
        features_dict = {
            'Torque': torque,
            'Rotational Speed': rpm / 10,  # Scale for better visualization
            'Process Temp': process_temp - 273,  # Convert to Celsius for viz
            'Air Temp': air_temp - 273,
            'Temp Difference': temp_diff,
            'Mech Power': mech_power / 10  # Scale for better visualization
        }
        
        fig_features = create_feature_chart(features_dict)
        st.plotly_chart(fig_features, use_container_width=True)

    with col2:
        st.markdown("### ⚡ Key Metrics")
        
        # Display gauges
        gauge_torque = create_gauge_chart(torque, "Torque", 100)
        st.plotly_chart(gauge_torque, use_container_width=True)

    st.markdown("---")

    # Prediction section
    if predict_button:
        with st.spinner("🔄 Analyzing machine condition..."):
            # Simulate processing time for effect
            time.sleep(1)
            
            # Prepare input data
            input_data = pd.DataFrame({
                'Type': [machine_type],
                'Air temperature (K)': [air_temp],
                'Process temperature (K)': [process_temp],
                'Rotational speed (rpm)': [rpm],
                'Torque (Nm)': [torque],
                'Tool wear (min)': [tool_wear],
                'temperature_difference': [temp_diff],
                'Mechanical Power (W)': [mech_power]
            })
            
            # Convert Type to category for XGBoost compatibility
            input_data['Type'] = input_data['Type'].astype('category')
            
            try:
                # Make prediction with categorical feature enabled
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("### 🎯 Diagnostic Results")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown("#### Machine Status")
                    status_html = get_status_html(prediction, prediction_proba[prediction])
                    st.markdown(status_html, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown("#### Confidence Level")
                    confidence = prediction_proba[prediction] * 100
                    st.markdown(f'<div class="metric-value">{confidence:.1f}%</div>', unsafe_allow_html=True)
                
                with result_col3:
                    st.markdown("#### Prediction Time")
                    current_time = datetime.now().strftime("%H:%M:%S")
                    st.markdown(f'<div class="metric-value" style="font-size: 1.5rem;">{current_time}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed analysis
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("#### 📊 Probability Distribution")
                    
                    prob_fig = go.Figure(data=[
                        go.Bar(
                            x=['Healthy', 'Failure Risk'],
                            y=prediction_proba * 100,
                            marker=dict(
                                color=['#27AE60' if prediction == 0 else '#E74C3C', 
                                       '#E74C3C' if prediction == 1 else '#27AE60'],
                                line=dict(color='#2C3E50', width=2)
                            ),
                            text=[f"{p*100:.1f}%" for p in prediction_proba],
                            textposition='auto',
                        )
                    ])
                    
                    prob_fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(44, 62, 80, 0.3)',
                        font={'color': '#ECF0F1'},
                        xaxis={'gridcolor': '#34495E'},
                        yaxis={'title': 'Probability (%)', 'gridcolor': '#34495E'},
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(prob_fig, use_container_width=True)
                
                with detail_col2:
                    st.markdown("#### 📋 Recommendations")
                    
                    if prediction == 0:
                        if confidence > 85:
                            st.markdown("""
                            <div class="info-box">
                                <p style="color: #27AE60; font-weight: bold;">✅ Optimal Operating Condition</p>
                                <p style="color: #ECF0F1;">• Continue normal operations</p>
                                <p style="color: #ECF0F1;">• Maintain current parameters</p>
                                <p style="color: #ECF0F1;">• Schedule routine inspection</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="warning-box">
                                <p class="warning-title">⚠️ Monitor Closely</p>
                                <p class="warning-text">• Machine is operational but approaching limits</p>
                                <p class="warning-text">• Consider parameter adjustment</p>
                                <p class="warning-text">• Increase monitoring frequency</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box" style="border-color: #E74C3C; background: rgba(231, 76, 60, 0.1);">
                            <p class="warning-title" style="color: #E74C3C;">🚨 IMMEDIATE ACTION REQUIRED</p>
                            <p class="warning-text">• High risk of equipment failure detected</p>
                            <p class="warning-text">• Stop machine and perform inspection</p>
                            <p class="warning-text">• Contact maintenance team immediately</p>
                            <p class="warning-text">• Review operating parameters</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.info("Please check that the model is compatible with the input features.")

    # Real-time Simulation Section
    st.markdown("---")
    st.markdown("### 🔬 Real-Time Simulation Mode")
    st.markdown("Simulate continuous machine operation with gradually degrading conditions until failure is predicted.")

    sim_col1, sim_col2 = st.columns([3, 1])

    with sim_col1:
        st.markdown("""
        <div class="info-box">
            <p style="color: #F39C12; font-weight: bold;">How Simulation Works:</p>
            <p style="color: #ECF0F1;">
                • Starts with current parameter values<br>
                • Gradually increases torque, temperature, and tool wear<br>
                • Runs predictions every iteration until failure is detected<br>
                • Automatically stops when failure risk is predicted
            </p>
        </div>
        """, unsafe_allow_html=True)

    with sim_col2:
        simulation_button = st.button("▶️ START SIMULATION", use_container_width=True, type="primary")

    if simulation_button:
        st.markdown("#### 📡 Simulation Running...")
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Initialize simulation parameters
    sim_torque = torque
    sim_rpm = rpm
    sim_process_temp = process_temp
    sim_air_temp = air_temp
    sim_tool_wear = tool_wear
    
    # Storage for simulation history
    simulation_history = []
    iteration = 0
    max_iterations = 100
    failure_detected = False
    
    # Progress bar
    progress_bar = st.progress(0)
    
    while iteration < max_iterations and not failure_detected:
        iteration += 1
        
        # Gradually degrade parameters (simulate wear and tear)
        sim_torque = min(sim_torque + np.random.uniform(0.5, 1.5), 100)
        sim_rpm = min(sim_rpm + np.random.uniform(5, 15), 3000)
        sim_process_temp = min(sim_process_temp + np.random.uniform(0.2, 0.8), 373)
        sim_tool_wear = min(sim_tool_wear + np.random.uniform(2, 5), 300)
        
        # Calculate derived features
        sim_temp_diff, sim_mech_power = calculate_derived_features(
            sim_torque, sim_rpm, sim_process_temp, sim_air_temp
        )
        
        # Prepare input data
        sim_input_data = pd.DataFrame({
            'Type': [machine_type],
            'Air temperature (K)': [sim_air_temp],
            'Process temperature (K)': [sim_process_temp],
            'Rotational speed (rpm)': [sim_rpm],
            'Torque (Nm)': [sim_torque],
            'Tool wear (min)': [sim_tool_wear],
            'temperature_difference': [sim_temp_diff],
            'Mechanical Power (W)': [sim_mech_power]
        })
        
        # Convert Type to category
        sim_input_data['Type'] = sim_input_data['Type'].astype('category')
        
        try:
            # Make prediction
            sim_prediction = model.predict(sim_input_data)[0]
            sim_proba = model.predict_proba(sim_input_data)[0]
            
            # Store history
            simulation_history.append({
                'iteration': iteration,
                'torque': sim_torque,
                'rpm': sim_rpm,
                'process_temp': sim_process_temp,
                'tool_wear': sim_tool_wear,
                'prediction': sim_prediction,
                'failure_probability': sim_proba[1] * 100
            })
            
            # Update progress
            progress_bar.progress(min(iteration / max_iterations, 1.0))
            
            # Update status
            if sim_prediction == 1:
                failure_detected = True
                status_placeholder.markdown(f"""
                <div class="warning-box" style="border-color: #E74C3C; background: rgba(231, 76, 60, 0.2);">
                    <p class="warning-title" style="color: #E74C3C;">🚨 FAILURE DETECTED AT ITERATION {iteration}</p>
                    <p class="warning-text" style="font-size: 1.1rem;">
                        Simulation stopped: Machine failure predicted with {sim_proba[1]*100:.1f}% confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                status_placeholder.markdown(f"""
                <div style="background: rgba(39, 174, 96, 0.1); border: 2px solid #27AE60; border-radius: 10px; padding: 1rem;">
                    <p style="color: #27AE60; font-weight: bold; margin: 0;">
                        ✅ Iteration {iteration}: Machine Operating Normally (Failure Risk: {sim_proba[1]*100:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Small delay for visualization
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Simulation error at iteration {iteration}: {e}")
            break
    
    # Display final results
    if failure_detected:
        st.success(f"✅ Simulation completed! Failure detected after {iteration} iterations.")
    else:
        st.warning(f"⚠️ Simulation completed {max_iterations} iterations without detecting failure.")
    
        # Create visualization of simulation history
    if simulation_history:
        hist_df = pd.DataFrame(simulation_history)
        
        # Create subplots
        st.markdown("#### 📈 Simulation Results")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Failure probability over time
            prob_fig = go.Figure()
            prob_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=hist_df['failure_probability'],
                mode='lines+markers',
                name='Failure Probability',
                line=dict(color='#E74C3C', width=3),
                marker=dict(size=6, color='#F39C12')
            ))
            
            # Add threshold line
            prob_fig.add_hline(y=50, line_dash="dash", line_color="#F39C12", 
                              annotation_text="50% Threshold")
            
            prob_fig.update_layout(
                title={'text': 'Failure Probability Over Time', 'font': {'color': '#F39C12'}},
                xaxis={'title': 'Iteration', 'gridcolor': '#34495E'},
                yaxis={'title': 'Failure Probability (%)', 'gridcolor': '#34495E', 'range': [0, 100]},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'},
                height=350
            )
            
            st.plotly_chart(prob_fig, use_container_width=True)
        
        with viz_col2:
            # Parameter degradation - Torque & Tool Wear
            param_fig = go.Figure()
            
            param_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=hist_df['torque'],
                mode='lines',
                name='Torque (Nm)',
                line=dict(color='#F39C12', width=2),
                yaxis='y'
            ))
            
            param_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=hist_df['tool_wear'],
                mode='lines',
                name='Tool Wear (min)',
                line=dict(color='#E74C3C', width=2),
                yaxis='y2'
            ))
            
            param_fig.update_layout(
                title={'text': 'Torque & Tool Wear Progression', 'font': {'color': '#F39C12'}},
                xaxis={'title': 'Iteration', 'gridcolor': '#34495E'},
                yaxis={
                    'title': {'text': 'Torque (Nm)', 'font': {'color': '#F39C12'}},
                    'gridcolor': '#34495E',
                    'tickfont': {'color': '#F39C12'}
                },
                yaxis2={
                    'title': {'text': 'Tool Wear (min)', 'font': {'color': '#E74C3C'}},
                    'overlaying': 'y',
                    'side': 'right',
                    'tickfont': {'color': '#E74C3C'}
                },
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'},
                height=350,
                showlegend=True,
                legend=dict(bgcolor='rgba(44, 62, 80, 0.5)', x=0.5, y=1.15, orientation='h')
            )
            
            st.plotly_chart(param_fig, use_container_width=True)
        
        # Additional detailed parameter charts
        st.markdown("#### 🔍 Detailed Parameter Evolution")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            # RPM and Temperature chart
            rpm_temp_fig = go.Figure()
            
            rpm_temp_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=hist_df['rpm'],
                mode='lines',
                name='Rotational Speed (rpm)',
                line=dict(color='#3498DB', width=2),
                yaxis='y'
            ))
            
            rpm_temp_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=hist_df['process_temp'],
                mode='lines',
                name='Process Temp (K)',
                line=dict(color='#E67E22', width=2),
                yaxis='y2'
            ))
            
            rpm_temp_fig.update_layout(
                title={'text': 'RPM & Temperature Changes', 'font': {'color': '#F39C12'}},
                xaxis={'title': 'Iteration', 'gridcolor': '#34495E'},
                yaxis={
                    'title': {'text': 'RPM', 'font': {'color': '#3498DB'}},
                    'gridcolor': '#34495E',
                    'tickfont': {'color': '#3498DB'}
                },
                yaxis2={
                    'title': {'text': 'Temperature (K)', 'font': {'color': '#E67E22'}},
                    'overlaying': 'y',
                    'side': 'right',
                    'tickfont': {'color': '#E67E22'}
                },
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'},
                height=350,
                showlegend=True,
                legend=dict(bgcolor='rgba(44, 62, 80, 0.5)', x=0.5, y=1.15, orientation='h')
            )
            
            st.plotly_chart(rpm_temp_fig, use_container_width=True)
        
        with detail_col2:
            # All parameters normalized view
            all_params_fig = go.Figure()
            
            # Normalize all parameters to 0-100 scale for comparison
            all_params_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=(hist_df['torque'] / 100) * 100,
                mode='lines',
                name='Torque %',
                line=dict(color='#F39C12', width=2)
            ))
            
            all_params_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=(hist_df['rpm'] / 3000) * 100,
                mode='lines',
                name='RPM %',
                line=dict(color='#3498DB', width=2)
            ))
            
            all_params_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=(hist_df['tool_wear'] / 300) * 100,
                mode='lines',
                name='Tool Wear %',
                line=dict(color='#E74C3C', width=2)
            ))
            
            all_params_fig.add_trace(go.Scatter(
                x=hist_df['iteration'],
                y=((hist_df['process_temp'] - 273) / 100) * 100,
                mode='lines',
                name='Process Temp %',
                line=dict(color='#E67E22', width=2)
            ))
            
            all_params_fig.update_layout(
                title={'text': 'All Parameters (Normalized %)', 'font': {'color': '#F39C12'}},
                xaxis={'title': 'Iteration', 'gridcolor': '#34495E'},
                yaxis={'title': 'Percentage of Maximum', 'gridcolor': '#34495E', 'range': [0, 100]},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                font={'color': '#ECF0F1'},
                height=350,
                showlegend=True,
                legend=dict(bgcolor='rgba(44, 62, 80, 0.5)')
            )
            
            st.plotly_chart(all_params_fig, use_container_width=True)        # Summary metrics
        st.markdown("#### 📊 Simulation Summary")
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        with sum_col1:
            st.metric("Total Iterations", iteration)
        
        with sum_col2:
            final_torque = hist_df.iloc[-1]['torque']
            torque_increase = final_torque - torque
            st.metric("Final Torque", f"{final_torque:.1f} Nm", f"+{torque_increase:.1f} Nm")
        
        with sum_col3:
            final_tool_wear = hist_df.iloc[-1]['tool_wear']
            wear_increase = final_tool_wear - tool_wear
            st.metric("Final Tool Wear", f"{final_tool_wear:.0f} min", f"+{wear_increase:.0f} min")
        
        with sum_col4:
            final_prob = hist_df.iloc[-1]['failure_probability']
            st.metric("Final Failure Risk", f"{final_prob:.1f}%")

    else:
        # Show placeholder when no prediction is made
        st.info("👈 Adjust the machine parameters in the sidebar and click 'ANALYZE MACHINE HEALTH' to get predictions, or start a simulation to see real-time degradation.")
        
        # Show some helpful information
        st.markdown("### 📖 System Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #F39C12;">About This System</h4>
                <p style="color: #ECF0F1;">
                    This predictive maintenance system uses machine learning to monitor 
                    the health of industrial milling machines in real-time. By analyzing 
                    key operational parameters, it can predict potential failures before 
                    they occur, reducing downtime and maintenance costs.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #F39C12;">How to Use</h4>
                <p style="color: #ECF0F1;">
                    1. Adjust the machine parameters using the sliders in the sidebar<br>
                    2. Derived features are automatically calculated<br>
                    3. Click the 'ANALYZE MACHINE HEALTH' button<br>
                    4. Review the diagnostic results and recommendations
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7F8C8D; padding: 1rem;">
    <p>Milling Machine Predictive Maintenance System | Powered by XGBoost & Streamlit</p>
    <p style="font-size: 0.8rem;">⚙️ Industrial Equipment Monitoring | Real-time Analysis | Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)
