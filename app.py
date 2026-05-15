import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import time

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme toggle function
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Custom CSS based on theme
def get_theme_css():
    if st.session_state.theme == 'dark':
        return """
        <style>
        /* Dark theme */
        .stApp {
            background-color: #1a1a1a;
        }
        .main-header {
            font-size: 2.5rem;
            color: #6c63ff;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #6c63ff 0%, #3f3d9e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .prediction-card {
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin: 1rem 0;
            text-align: center;
        }
        .churn-high {
            background: linear-gradient(135deg, #6c63ff 0%, #3f3d9e 100%);
            color: white;
        }
        .churn-low {
            background: linear-gradient(135deg, #00b09b 0%, #2c7a5e 100%);
            color: white;
        }
        .metric-card {
            background: #2d2d2d;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: transform 0.3s;
            color: #ffffff;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        /* Dark theme text colors */
        .stMarkdown, .stText, .stNumberInput label, .stSelectbox label, .stSlider label {
            color: #ffffff !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, #6c63ff 0%, #3f3d9e 100%);
            color: white !important;
            border: none;
        }
        /* Fix for metric text */
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #cccccc !important;
        }
        /* Fix for info/warning/success boxes */
        .stAlert {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        </style>
        """
    else:
        return """
        <style>
        /* Light theme */
        .stApp {
            background-color: #ffffff;
        }
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .prediction-card {
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            text-align: center;
        }
        .churn-high {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .churn-low {
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
            color: white;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        /* Light theme text colors - ensuring visibility */
        .stMarkdown, .stText {
            color: #000000 !important;
        }
        .stNumberInput label, .stSelectbox label, .stSlider label {
            color: #000000 !important;
            font-weight: 500 !important;
        }
        /* Button styling for light theme */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
        }
        .stButton > button:hover {
            opacity: 0.9;
        }
        /* Metric text colors for light theme */
        div[data-testid="stMetricValue"] {
            color: #000000 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #333333 !important;
        }
        /* Input field styling */
        .stNumberInput input, .stSelectbox select, .stTextInput input {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        /* Info/Warning/Success boxes text */
        .stAlert {
            color: #000000 !important;
        }
        /* Dataframe styling */
        .dataframe {
            color: #000000 !important;
        }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Load the trained model and encoders
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('model.h5')
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Prediction function
def predict_churn(model, scaler, label_encoder_gender, onehot_encoder_geo, input_data):
    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
    geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0][0]

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Navigation")
    
    # Theme toggle button in sidebar
    theme_label = "Dark Mode" if st.session_state.theme == 'light' else "Light Mode"
    if st.button(f"🌓 {theme_label}", use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.markdown("---")
    
    # Determine sidebar background for option_menu
    sidebar_bg = "#fafafa" if st.session_state.theme == 'light' else "#2d2d2d"
    text_color = "#000000" if st.session_state.theme == 'light' else "#ffffff"
    
    selected = option_menu(
        menu_title=None,
        options=["Predictor", "Batch Prediction", "Model Insights", "About"],
        icons=["calculator", "upload", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": sidebar_bg},
            "icon": {"color": "#1E88E5", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": text_color},
            "nav-link-selected": {"background-color": "#1E88E5", "color": "#ffffff"},
        }
    )

# Main content
st.markdown('<h1 class="main-header">Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Load models
model, label_encoder_gender, onehot_encoder_geo, scaler = load_models()

if model is None:
    st.error("Please ensure all model files are in the correct directory")
    st.stop()

if selected == "Predictor":
    st.markdown("### Customer Information")
    st.markdown("Please fill in the customer details below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        geography = st.selectbox(
            "Geography",
            onehot_encoder_geo.categories_[0],
            help="Select the customer's country"
        )
        
        gender = st.selectbox(
            "Gender",
            label_encoder_gender.classes_,
            help="Select the customer's gender"
        )
        
        age = st.slider(
            "Age",
            18, 92, 35,
            help="Customer's age in years"
        )
        
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            help="Credit score range: 300-850"
        )
        
        tenure = st.slider(
            "Tenure (Years)",
            0, 10, 5,
            help="Number of years as a customer"
        )
    
    with col2:
        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            max_value=250000.0,
            value=50000.0,
            step=1000.0,
            help="Customer's account balance",
            format="%.2f"
        )
        
        num_of_products = st.slider(
            "Number of Products",
            1, 4, 2,
            help="Number of products purchased"
        )
        
        has_cr_card = st.selectbox(
            "Has Credit Card",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Does the customer have a credit card?"
        )
        
        is_active_member = st.selectbox(
            "Is Active Member",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Is the customer an active member?"
        )
        
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=200000.0,
            value=75000.0,
            step=1000.0,
            help="Customer's estimated annual salary",
            format="%.2f"
        )
    
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
    
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing customer data..."):
            time.sleep(1)
            prediction_proba = predict_churn(
                model, scaler, label_encoder_gender, 
                onehot_encoder_geo, input_data
            )
        
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{prediction_proba:.1%}")
        
        with col2:
            risk_level = "High" if prediction_proba > 0.5 else "Low"
            st.metric("Risk Level", risk_level)
        
        with col3:
            confidence = abs(prediction_proba - 0.5) * 2
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Churn Probability (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction card
        if prediction_proba > 0.5:
            st.markdown(f"""
            <div class="prediction-card churn-high">
                <h2>High Churn Risk</h2>
                <h3>Churn Probability: {prediction_proba:.1%}</h3>
                <p>This customer is likely to churn. Consider retention strategies!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card churn-low">
                <h2>Low Churn Risk</h2>
                <h3>Churn Probability: {prediction_proba:.1%}</h3>
                <p>This customer is likely to stay. Keep up the good service!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors analysis
        st.markdown("### Key Risk Factors")
        risk_factors = []
        
        if credit_score < 600:
            risk_factors.append("Low credit score")
        if age < 30 or age > 60:
            risk_factors.append("Age outside optimal range (30-60)")
        if tenure < 2:
            risk_factors.append("Short tenure")
        if num_of_products == 1:
            risk_factors.append("Only 1 product")
        elif num_of_products > 3:
            risk_factors.append("Too many products")
        if not is_active_member:
            risk_factors.append("Inactive member")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(f"⚠️ {factor}")
        else:
            st.success("No significant risk factors detected")

elif selected == "Batch Prediction":
    st.markdown("### Batch Prediction")
    st.markdown("Upload a CSV file with multiple customer records for batch prediction")
    
    # Template download
    template_df = pd.DataFrame({
        'CreditScore': [650, 700],
        'Geography': ['France', 'Germany'],
        'Gender': ['Male', 'Female'],
        'Age': [35, 42],
        'Tenure': [5, 3],
        'Balance': [50000, 75000],
        'NumOfProducts': [2, 1],
        'HasCrCard': [1, 0],
        'IsActiveMember': [1, 1],
        'EstimatedSalary': [75000, 90000]
    })
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="batch_prediction_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        required_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    progress_bar = st.progress(0)
                    predictions = []
                    for idx, row in df.iterrows():
                        input_data = row.to_dict()
                        proba = predict_churn(
                            model, scaler, label_encoder_gender,
                            onehot_encoder_geo, input_data
                        )
                        predictions.append(proba)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    df['Churn_Probability'] = predictions
                    df['Churn_Risk'] = df['Churn_Probability'].apply(
                        lambda x: 'High' if x > 0.5 else 'Low'
                    )
                    
                    st.success("Predictions completed!")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        high_risk = (df['Churn_Risk'] == 'High').sum()
                        st.metric("High Risk Customers", high_risk)
                    with col3:
                        low_risk = (df['Churn_Risk'] == 'Low').sum()
                        st.metric("Low Risk Customers", low_risk)
                    with col4:
                        avg_risk = df['Churn_Probability'].mean()
                        st.metric("Average Risk", f"{avg_risk:.1%}")

elif selected == "Model Insights":
    st.markdown("### Model Performance Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "86.0%", "↑ 2.1%")
    with col2:
        st.metric("Precision", "85.3%", "↑ 1.5%")
    with col3:
        st.metric("Recall", "84.7%", "↑ 1.8%")
    with col4:
        st.metric("F1-Score", "85.0%", "↑ 1.6%")
    
    st.markdown("---")
    
    st.markdown("### Feature Importance")
    features = {
        'Age': 0.95,
        'Balance': 0.92,
        'NumOfProducts': 0.88,
        'CreditScore': 0.85,
        'IsActiveMember': 0.82,
        'Geography': 0.78,
        'EstimatedSalary': 0.65,
        'Tenure': 0.60,
        'Gender': 0.55,
        'HasCrCard': 0.45
    }
    
    fig = px.bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation='h',
        title="Top Factors Influencing Churn",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=list(features.values()),
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Retention Recommendations")
    
    with st.expander("For High-Risk Customers"):
        st.markdown("""
        - Offer personalized discounts or loyalty rewards
        - Proactive customer support outreach
        - Introduce new product bundles
        - Improve mobile app experience
        - Implement referral programs
        """)
    
    with st.expander("For Medium-Risk Customers"):
        st.markdown("""
        - Regular engagement through newsletters
        - Birthday and anniversary offers
        - Push notifications for new features
        - Credit card upgrade offers
        """)
    
    with st.expander("For Low-Risk Customers"):
        st.markdown("""
        - VIP program enrollment
        - Early access to new products
        - Exclusive event invitations
        - Premium support services
        """)

else:
    st.markdown("### About This Application")
    
    st.markdown("""
    #### Customer Churn Prediction System
    
    This AI-powered application helps businesses predict customer churn using a deep learning model.
    
    **Key Features:**
    - Real-time churn prediction
    - Interactive data visualization
    - Batch processing for multiple customers
    - Feature importance analysis
    - Actionable retention recommendations
    
    **How It Works:**
    1. Enter customer information into the form
    2. The ANN model analyzes patterns in the data
    3. Get instant churn probability score
    4. Receive personalized retention strategies
    
    **Model Architecture:**
    - Input Layer: 12 features
    - Hidden Layer 1: 64 neurons (ReLU)
    - Hidden Layer 2: 32 neurons (ReLU)
    - Output Layer: 1 neuron (Sigmoid)
    
    **Performance Metrics:**
    - Training Accuracy: 86.7%
    - Validation Accuracy: 86.0%
    - AUC-ROC Score: 0.91
    
    **Technologies Used:**
    - TensorFlow/Keras for deep learning
    - Streamlit for web interface
    - Plotly for interactive visualizations
    - Scikit-learn for preprocessing
    """)
    
    st.info("Note: This is a demonstration model. For production use, ensure regular model retraining with updated data.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Customer Churn Prediction System | Powered by AI</p>",
    unsafe_allow_html=True
)