import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .feature-description {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Path configuration
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "customer_churn_business_dataset.csv"

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Usia pelanggan dalam tahun',
    'gender': 'Jenis kelamin pelanggan',
    'country': 'Negara tempat tinggal pelanggan',
    'city': 'Kota tempat tinggal pelanggan',
    'customer_segment': 'Segmen pelanggan (Enterprise/SME/Individual)',
    'tenure_months': 'Lama berlangganan dalam bulan',
    'signup_channel': 'Channel pendaftaran (Web/Mobile/Referral/etc)',
    'contract_type': 'Tipe kontrak (Monthly/Yearly)',
    'monthly_logins': 'Rata-rata login per bulan',
    'weekly_active_days': 'Rata-rata hari aktif per minggu',
    'features_used': 'Jumlah fitur produk yang digunakan',
    'usage_growth_rate': 'Tingkat pertumbuhan penggunaan',
    'last_login_days_ago': 'Berapa hari sejak login terakhir',
    'monthly_fee': 'Biaya berlangganan bulanan (USD)',
    'total_revenue': 'Total revenue dari pelanggan (USD)',
    'payment_method': 'Metode pembayaran yang digunakan',
    'payment_failures': 'Jumlah kegagalan pembayaran',
    'discount_applied': 'Apakah pelanggan mendapat diskon',
    'price_increase_last_3m': 'Apakah ada kenaikan harga 3 bulan terakhir',
    'support_tickets': 'Jumlah tiket support yang dibuat',
    'complaint_type': 'Tipe komplain utama',
    'escalations': 'Jumlah eskalasi masalah',
    'email_open_rate': 'Tingkat pembukaan email (0-1)',
    'marketing_click_rate': 'Tingkat klik email marketing (0-1)',
    'nps_score': 'Net Promoter Score (0-100)',
    'survey_response': 'Respon terhadap survey kepuasan',
    'referral_count': 'Jumlah referral yang dilakukan',
    'csat_score': 'Customer Satisfaction Score (1-5)'
}

# Load models and scaler
# Memuat model dan scaler
@st.cache_resource
def load_models():
    """Memuat semua model yang telah dilatih dan scaler"""
    models = {}
    errors = []
    
    try:
        models['MLP'] = tf.keras.models.load_model(BASE_DIR / 'mlp_model.h5')
    except Exception as e:
        errors.append(f"MLP: {str(e)}")
    
    try:
        models['TabNet'] = joblib.load(BASE_DIR / 'tabnet_model.pkl')
    except Exception as e:
        errors.append(f"TabNet: {str(e)}")
    
    try:
        models['FT-Transformer'] = tf.keras.models.load_model(BASE_DIR / 'ft_transformer_model.h5')
    except Exception as e:
        errors.append(f"FT-Transformer: {str(e)}")
    
    try:
        scaler = joblib.load(BASE_DIR / 'scaler.pkl')
    except Exception as e:
        errors.append(f"Scaler: {str(e)}")
        scaler = None
    
    return models, scaler, errors

# Debugging info di sidebar atau halaman utama
models, scaler, model_errors = load_models()
if model_errors:
    st.error(f"Beberapa model tidak bisa dimuat:")
    for error in model_errors:
        st.warning(error)


# Load dataset
@st.cache_data
def load_dataset():
    """Load the customer churn dataset"""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def preprocess_input(data, scaler, dataset):
    """Preprocess input data for prediction"""
    input_df = pd.DataFrame([data])
    
    # Get categorical columns from dataset
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['customer_id', 'churn']]
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in input_df.columns:
            le = LabelEncoder()
            le.fit(dataset[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))
    
    # Ensure all features are present
    feature_cols = [col for col in dataset.columns if col not in ['customer_id', 'churn']]
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_cols]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    return input_scaled

def main():
 st.title("📊 Customer Churn Prediction System")
 st.markdown("### Prediksi Churn Pelanggan menggunakan Multiple AI Models")
 
 # Load models and data
 models, scaler, model_errors = load_models()
 dataset = load_dataset()
 
 if not models or scaler is None or dataset is None:
     st.error("⚠️ Failed to load required resources.")
     if model_errors:
         with st.expander("View Error Details"):
             for error in model_errors:
                 st.text(error)
     st.info("Please ensure these files exist in the src folder:\n- mlp_model.h5\n- tabnet_model.pkl\n- ft_transformer_model.h5\n- scaler.pkl\n- customer_churn_business_dataset.csv")
     return
 
 # Show which models are loaded
 if model_errors:
     with st.expander("⚠️ Some models could not be loaded"):
         for error in model_errors:
             st.warning(error)
 
 # Sidebar
 with st.sidebar:
     st.header("⚙️ Configuration")
     
     # Model selection
     available_models = list(models.keys())
     
     if available_models:
         selected_model = st.selectbox(
             "Select Model",
             available_models,
             help="Choose the AI model for prediction"
         )
         
         # Show model info
         model_info = {
             'MLP': '🔷 Neural Network - Fast & Reliable',
             'TabNet': '🔶 Tabular Model - Feature Importance',
             'FT-Transformer': '🔸 Transformer - Complex Patterns'
         }
         
         if selected_model in model_info:
             st.info(model_info[selected_model])
     else:
         st.error("No models available!")
         selected_model = None
     
     st.divider()
     
     # Navigation
     page = st.radio(
         "Navigation",
         ["🔮 Prediction", "📊 Dataset Overview", "📈 Batch Prediction", "ℹ️ About Models"],
         label_visibility="collapsed"
     )
 
 # Main content
 if page == "🔮 Prediction":
     if selected_model:
         show_prediction_page(models[selected_model], selected_model, scaler, dataset)
     else:
         st.error("No model selected or available!")
 elif page == "📊 Dataset Overview":
     show_dataset_overview(dataset)
 elif page == "📈 Batch Prediction":
     show_batch_prediction_page(models, scaler, dataset)
 else:
     show_about_page()


def show_prediction_page(model, model_name, scaler, dataset):
    st.header("🔮 Customer Churn Prediction")
    
    # Information box
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ Cara Menggunakan Aplikasi</h4>
        <p><strong>Tujuan:</strong> Aplikasi ini memprediksi apakah seorang pelanggan akan berhenti menggunakan layanan (churn) atau tidak.</p>
        <p><strong>Cara Kerja:</strong></p>
        <ol>
            <li>Isi informasi pelanggan di form di bawah ini</li>
            <li>Klik tombol "Predict Churn" di bagian bawah</li>
            <li>Sistem akan menganalisis data dan memberikan prediksi</li>
            <li>Anda akan mendapat rekomendasi aksi berdasarkan hasil prediksi</li>
        </ol>
        <p><strong>Tips:</strong> Anda bisa menggunakan nilai default yang sudah tersedia atau mengubahnya sesuai data pelanggan yang ingin diprediksi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Get feature columns
    feature_cols = [col for col in dataset.columns if col not in ['customer_id', 'churn']]
    categorical_cols = dataset.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['customer_id', 'churn']]
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]
    
    # Create input form
    st.subheader("📝 Enter Customer Information")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Basic Info", 
        "📊 Usage Metrics", 
        "💰 Financial Info", 
        "📞 Support & Satisfaction"
    ])
    
    input_data = {}
    
    with tab1:
        st.markdown("### Informasi Dasar Pelanggan")
        col1, col2 = st.columns(2)
        
        basic_features = ['age', 'gender', 'country', 'city', 'customer_segment', 'tenure_months', 'signup_channel', 'contract_type']
        
        for i, feature in enumerate(basic_features):
            if feature in feature_cols:
                col = col1 if i % 2 == 0 else col2
                with col:
                    if feature in categorical_cols:
                        unique_values = dataset[feature].unique()
                        try:
                            unique_values = sorted(unique_values)
                        except:
                            unique_values = sorted(unique_values, key=str)
                        input_data[feature] = st.selectbox(
                            f"**{feature.replace('_', ' ').title()}**", 
                            unique_values,
                            help=FEATURE_DESCRIPTIONS.get(feature, '')
                        )
                    else:
                        min_val = float(dataset[feature].min())
                        max_val = float(dataset[feature].max())
                        mean_val = float(dataset[feature].mean())
                        input_data[feature] = st.number_input(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=FEATURE_DESCRIPTIONS.get(feature, '')
                        )
    
    with tab2:
        st.markdown("### Metrik Penggunaan Layanan")
        col1, col2 = st.columns(2)
        
        usage_features = ['monthly_logins', 'weekly_active_days', 'features_used', 'usage_growth_rate', 'last_login_days_ago']
        
        for i, feature in enumerate(usage_features):
            if feature in feature_cols:
                col = col1 if i % 2 == 0 else col2
                with col:
                    min_val = float(dataset[feature].min())
                    max_val = float(dataset[feature].max())
                    mean_val = float(dataset[feature].mean())
                    input_data[feature] = st.number_input(
                        f"**{feature.replace('_', ' ').title()}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=FEATURE_DESCRIPTIONS.get(feature, '')
                    )
    
    with tab3:
        st.markdown("### Informasi Finansial")
        col1, col2 = st.columns(2)
        
        financial_features = ['monthly_fee', 'total_revenue', 'payment_method', 'payment_failures', 'discount_applied', 'price_increase_last_3m']
        
        for i, feature in enumerate(financial_features):
            if feature in feature_cols:
                col = col1 if i % 2 == 0 else col2
                with col:
                    if feature in categorical_cols:
                        unique_values = dataset[feature].unique()
                        try:
                            unique_values = sorted(unique_values)
                        except:
                            unique_values = sorted(unique_values, key=str)
                        input_data[feature] = st.selectbox(
                            f"**{feature.replace('_', ' ').title()}**",
                            unique_values,
                            help=FEATURE_DESCRIPTIONS.get(feature, '')
                        )
                    else:
                        min_val = float(dataset[feature].min())
                        max_val = float(dataset[feature].max())
                        mean_val = float(dataset[feature].mean())
                        input_data[feature] = st.number_input(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=FEATURE_DESCRIPTIONS.get(feature, '')
                        )
    
    with tab4:
        st.markdown("### Support & Kepuasan Pelanggan")
        col1, col2 = st.columns(2)
        
        support_features = ['support_tickets', 'complaint_type', 'escalations', 'email_open_rate', 
                           'marketing_click_rate', 'nps_score', 'survey_response', 'referral_count', 'csat_score']
        
        for i, feature in enumerate(support_features):
            if feature in feature_cols:
                col = col1 if i % 2 == 0 else col2
                with col:
                    if feature in categorical_cols:
                        unique_values = dataset[feature].unique()
                        try:
                            unique_values = sorted(unique_values)
                        except:
                            unique_values = sorted(unique_values, key=str)
                        input_data[feature] = st.selectbox(
                            f"**{feature.replace('_', ' ').title()}**",
                            unique_values,
                            help=FEATURE_DESCRIPTIONS.get(feature, '')
                        )
                    else:
                        min_val = float(dataset[feature].min())
                        max_val = float(dataset[feature].max())
                        mean_val = float(dataset[feature].mean())
                        input_data[feature] = st.number_input(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=FEATURE_DESCRIPTIONS.get(feature, '')
                        )
    
    # Fill remaining features
    for feature in feature_cols:
        if feature not in input_data:
            if feature in categorical_cols:
                unique_values = dataset[feature].unique()
                try:
                    unique_values = sorted(unique_values)
                except:
                    unique_values = sorted(unique_values, key=str)
                input_data[feature] = unique_values[0]
            else:
                input_data[feature] = float(dataset[feature].mean())
    
    st.divider()
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Predict Churn", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner(f"Analyzing with {model_name} model..."):
            # Preprocess input
            input_processed = preprocess_input(input_data, scaler, dataset)
            
            # Make prediction based on selected model
            try:
                if model_name == "MLP" or model_name == "FT-Transformer":
                    prediction_proba = model.predict(input_processed, verbose=0)
                    prediction = (prediction_proba > 0.5).astype(int)[0][0]
                    probability = float(prediction_proba[0][0])
                elif model_name == "TabNet":
                    prediction = int(model.predict(input_processed)[0])
                    probability = float(model.predict_proba(input_processed)[0][1])
                else:
                    st.error("Unknown model type")
                    return
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.metric(
                            "Prediction",
                            "⚠️ CHURN",
                            delta="High Risk",
                            delta_color="inverse"
                        )
                    else:
                        st.metric(
                            "Prediction",
                            "✅ NO CHURN",
                            delta="Safe",
                            delta_color="normal"
                        )
                
                with col2:
                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%",
                        delta=None
                    )
                
                with col3:
                    risk_level = "🔴 High" if probability > 0.7 else "🟡 Medium" if probability > 0.4 else "🟢 Low"
                    st.metric(
                        "Risk Level",
                        risk_level,
                        delta=None
                    )
                
                # Probability gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Churn Probability<br><span style='font-size:0.8em'>Model: {model_name}</span>"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if prediction == 1:
                    st.error("""
                    **⚠️ High Churn Risk Detected!**
                    
                    **Immediate Actions:**
                    - 📞 Contact customer immediately with personalized retention offer
                    - 💬 Schedule one-on-one call to understand concerns
                    - 🎁 Offer special discount or loyalty reward (10-20% off)
                    - ⭐ Upgrade to premium support tier temporarily
                    
                    **Analysis Points:**
                    - Review recent usage patterns and engagement metrics
                    - Check support ticket history for unresolved issues
                    - Analyze payment history for any failures
                    - Compare with similar customers who churned
                    
                    **Prevention Strategy:**
                    - Send personalized email highlighting unused features
                    - Offer product training or onboarding session
                    - Create custom success plan with milestones
                    - Assign dedicated account manager
                    """)
                else:
                    st.success("""
                    **✅ Low Churn Risk - Customer is Stable**
                    
                    **Maintenance Actions:**
                    - ✉️ Continue regular engagement through newsletters
                    - 📊 Monitor usage metrics monthly
                    - 🎯 Send relevant product updates and tips
                    - 💪 Encourage feature adoption
                    
                    **Growth Opportunities:**
                    - 📈 Consider upselling premium features
                    - 🤝 Request testimonial or case study
                    - 👥 Encourage referrals with incentive program
                    - 🌟 Invite to beta test new features
                    
                    **Best Practices:**
                    - Maintain excellent customer service
                    - Keep them informed about product roadmap
                    - Recognize loyalty milestones (anniversary, etc)
                    - Regular satisfaction surveys (quarterly)
                    """)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

def show_batch_prediction_page(models, scaler, dataset):
 st.header("📊 Batch Customer Churn Prediction")
 
 st.markdown("""
 <div class="info-box">
     <h4>ℹ️ Cara Menggunakan Batch Prediksi</h4>
     <p><strong>Tujuan:</strong> Menggunakan aplikasi ini untuk memprediksi churn pada banyak pelanggan sekaligus dengan mengunggah file CSV.</p>
     <p><strong>Cara Kerja:</strong></p>
     <ol>
         <li>Unggah file CSV yang berisi data pelanggan</li>
         <li>Prediksi churn akan dilakukan untuk semua data yang ada di dalam file tersebut</li>
         <li>Hasil prediksi akan ditampilkan dalam tabel dan bisa diunduh dalam format CSV</li>
     </ol>
 </div>
 """, unsafe_allow_html=True)
 
 # Unggah file CSV
 uploaded_file = st.file_uploader("Upload file CSV untuk prediksi batch", type=["csv"])
 
 if uploaded_file is not None:
     try:
         # Membaca file CSV yang diunggah
         df = pd.read_csv(uploaded_file)
         st.write("Data yang diunggah:")
         st.dataframe(df.head())

         # Preprocessing data untuk prediksi batch
         input_data = df.copy()
         feature_cols = [col for col in dataset.columns if col not in ['customer_id', 'churn']]
         categorical_cols = dataset.select_dtypes(include=['object']).columns
         categorical_cols = [col for col in categorical_cols if col not in ['customer_id', 'churn']]

         # Encode kategori dan preprocessing
         for col in categorical_cols:
             if col in feature_cols:
                 le = LabelEncoder()
                 le.fit(dataset[col].astype(str))
                 input_data[col] = le.transform(input_data[col].astype(str))

         # Scaling data
         input_data_scaled = scaler.transform(input_data[feature_cols])

         # Prediksi untuk batch
         predictions = models['TabNet'].predict(input_data_scaled)  # Pilih model sesuai
         df['Churn Prediction'] = predictions
         churn_probabilities = models['TabNet'].predict_proba(input_data_scaled)[:, 1]
         df['Churn Probability'] = churn_probabilities

         # Tampilkan hasil prediksi
         st.subheader("Hasil Prediksi Batch")
         st.dataframe(df[['customer_id', 'Churn Prediction', 'Churn Probability']])

         # Tambahkan tombol untuk mengunduh file hasil prediksi
         st.download_button(
             label="Unduh Hasil Prediksi CSV",
             data=df.to_csv(index=False).encode('utf-8'),
             file_name='prediksi_churn_batch.csv',
             mime='text/csv'
         )

     except Exception as e:
         st.error(f"Error saat memproses file: {str(e)}")

def show_dataset_overview(dataset):
    st.header("📈 Dataset Overview")
    
    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ Tentang Dataset</h4>
        <p>Dataset ini berisi informasi pelanggan yang digunakan untuk melatih model prediksi churn. 
        Data mencakup informasi demografis, pola penggunaan, data finansial, dan metrik kepuasan pelanggan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", f"{len(dataset):,}")
    with col2:
        st.metric("🔢 Total Features", len(dataset.columns) - 2)
    with col3:
        churn_count = dataset['churn'].sum()
        st.metric("⚠️ Churned Customers", f"{churn_count:,}")
    with col4:
        churn_rate = (churn_count / len(dataset)) * 100
        st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    
    st.divider()
    
    # Churn distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Churn")
        churn_counts = dataset['churn'].value_counts()
        fig = px.pie(
            values=churn_counts.values,
            names=['No Churn', 'Churn'],
            title="Customer Churn Distribution",
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Data Sample")
        st.dataframe(dataset.head(10), use_container_width=True, height=400)
    
    st.divider()
    
    # Feature statistics
    st.subheader("📊 Feature Statistics")
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'churn']
    
    if len(numerical_cols) > 0:
        st.dataframe(dataset[numerical_cols].describe(), use_container_width=True)
    
    st.divider()
    
    # Feature distribution
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox("Select feature to visualize", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(dataset, x=selected_feature, nbins=30, 
                          title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(dataset, y=selected_feature, x='churn',
                    title=f"{selected_feature} by Churn Status")
        st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.header("ℹ️ About Models & System")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Tujuan Sistem</h4>
        <p>Sistem ini dirancang untuk memprediksi kemungkinan pelanggan akan berhenti menggunakan layanan (churn). 
        Dengan prediksi yang akurat, perusahaan dapat mengambil tindakan preventif untuk mempertahankan pelanggan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🤖 Model Architecture Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔷 MLP Model
        **Multilayer Perceptron**
        
        - **Type:** Non-Pretrained Deep Neural Network
        - **Architecture:** 
          - Input → 128 → 64 → 32 → 16 → 1
          - BatchNormalization after each layer
          - Dropout (0.5) for regularization
        - **Optimizer:** Adam (lr=0.0001)
        - **Training:** 100 epochs with early stopping
        
        **Advantages:**
        - ✅ Fast inference
        - ✅ Good generalization
        - ✅ Reliable predictions
        - ✅ Easy to deploy
        
        **Best For:** General-purpose predictions
        """)
    
    with col2:
        st.markdown("""
        #### 🔶 TabNet Model
        **Attention-based Tabular Network**
        
        - **Type:** Pretrained Specialized Model
        - **Architecture:**
          - Sequential attention mechanism
          - Feature selection capability
          - Instance-wise feature learning
        - **Batch Size:** 256 (virtual: 64)
        - **Training:** 100 epochs with patience=20
        
        **Advantages:**
        - ✅ Feature interpretability
        - ✅ Handles tabular data well
        - ✅ Automatic feature selection
        - ✅ High accuracy
        
        **Best For:** Structured/tabular data analysis
        """)
    
    with col3:
        st.markdown("""
        #### 🔸 FT-Transformer
        **Feature Tokenizer Transformer**
        
        - **Type:** Pretrained Transformer Model
        - **Architecture:**
          - Feature embedding (dim=64)
          - Multi-head attention (4 heads)
          - Feed-forward network (128 dim)
          - Layer normalization
        - **Optimizer:** Adam (lr=0.0005)
        - **Training:** 100 epochs with early stopping
        
        **Advantages:**
        - ✅ Captures complex patterns
        - ✅ Feature interactions
        - ✅ State-of-the-art architecture
        - ✅ High performance
        
        **Best For:** Complex feature relationships
        """)
    
    st.divider()
    
    st.subheader("📊 Model Performance Metrics")
    st.info("""
    **Training Configuration:**
    - **Data Split:** 80% Training, 20% Testing
    - **Stratification:** Yes (balanced classes)
    - **Scaling:** StandardScaler normalization
    - **Validation:** Hold-out validation set
    - **Early Stopping:** Prevents overfitting
    - **Metrics:** Accuracy, Precision, Recall, F1-Score
    
    All models were trained on the same dataset with careful hyperparameter tuning to ensure optimal performance.
    """)
    
    st.divider()
    
    st.subheader("🎯 Business Use Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Primary Applications
        1. **Customer Retention**
           - Identify at-risk customers early
           - Proactive intervention strategies
           - Reduce customer acquisition costs
        
        2. **Marketing Optimization**
           - Targeted retention campaigns
           - Personalized offers
           - Resource allocation
        
        3. **Revenue Protection**
           - Prevent revenue loss
           - Customer lifetime value optimization
           - Churn cost reduction
        """)
    
    with col2:
        st.markdown("""
        #### Business Benefits
        1. **Cost Savings**
           - Lower customer acquisition costs
           - Reduced marketing spend
           - Better ROI on retention efforts
        
        2. **Customer Insights**
           - Understand churn drivers
           - Improve product/service
           - Enhance customer experience
        
        3. **Strategic Planning**
           - Data-driven decisions
           - Resource optimization
           - Competitive advantage
        """)
    
    st.divider()
    
    st.subheader("🛠️ Technical Stack")
    st.code("""
    - Python 3.x
    - TensorFlow / Keras (Deep Learning)
    - PyTorch TabNet (Tabular Learning)
    - Scikit-learn (Preprocessing)
    - Streamlit (Web Interface)
    - Plotly (Visualizations)
    - Pandas & NumPy (Data Processing)
    """, language="python")

if __name__ == "__main__":
    main()