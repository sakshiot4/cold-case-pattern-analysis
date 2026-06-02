import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION & THEME ---
st.set_page_config(page_title="Cold Case Risk AI Predictor", layout="centered")

# Custom CSS to match your dark navy Power BI style theme
st.markdown("""
    <style>
    .stApp { background-color: #1F3864; color: white; }
    div[data-testid="stMetricValue"] { color: #C00000 !important; }
    h1, h2, h3, label, .stMarkdown p { color: white !important; }
    .stButton>button { border: 1px solid #C00000; border-radius: 8px; }
    .stButton>button:hover { border: 1px solid #ff4b4b; color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("🔴 Investigative Intelligence Predictor")
st.write("Input case details below to query the Random Forest model for real-time risk classification.")

# --- STEP 1: LOAD SERIALIZED ARTIFACTS ---
MODEL_PATH = os.path.join('models', 'cold_case_rf_model.pkl')
COLUMNS_PATH = os.path.join('models', 'feature_columns.pkl')

@st.cache_resource
def load_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        columns = joblib.load(COLUMNS_PATH)
        return model, columns
    return None, None

rf_model, expected_columns = load_assets()

if rf_model is None:
    st.error("⚠️ Model asset files (`cold_case_rf_model.pkl` / `feature_columns.pkl`) not found in the /models directory. Run your notebook to export them.")
else:
    # --- STEP 2: MAGIC BUTTON (LOAD HISTORICAL DATA) ---
    st.markdown("---")
    st.markdown("💡 **Testing Tool:** Load a real historical case from your dataset to see how the model reacts.")
    if st.button("🎲 Load Random Historical Case", use_container_width=True):
        try:
            # Load your cleaned export data
            df_historical = pd.read_csv(os.path.join('outputs', 'final_portfolio_export.csv'))
            
            # Pick one random row
            random_case = df_historical.sample(1).iloc[0]
            
            # Save the random case attributes into Streamlit's session memory
            st.session_state['random_state'] = random_case['State']
            st.session_state['random_month'] = random_case['Month']
            st.session_state['random_sex'] = random_case['Victim Sex']
            st.session_state['random_race'] = random_case['Victim Race']
            st.session_state['random_count'] = random_case['Victim Count']
            st.session_state['random_decade'] = random_case['Decade']
            st.session_state['random_age'] = random_case['Victim Age Group']
            st.session_state['random_weapon'] = random_case['Weapon Grouped']
            st.session_state['random_agency'] = random_case['Agency Grouped']
            st.session_state['random_relation'] = random_case['Relationship Grouped']
        except Exception as e:
            st.error(f"Could not load historical data. Ensure 'outputs/final_portfolio_export.csv' exists. Error: {e}")

    # --- STEP 3: USER INPUT FORM LAYOUT ---
    st.markdown("---")
    st.subheader("📋 Case Profile Attributes")
    
    col1, col2 = st.columns(2)
    with col1:
        state = st.text_input("Jurisdiction State", value=st.session_state.get('random_state', 'California'))
        month = st.text_input("Incident Month", value=st.session_state.get('random_month', 'January'))
        
        sex_options = ["Male", "Female", "Unknown"]
        default_sex = sex_options.index(st.session_state.get('random_sex', 'Male')) if st.session_state.get('random_sex', 'Male') in sex_options else 0
        victim_sex = st.selectbox("Victim Sex", sex_options, index=default_sex)
        
        race_options = ["White", "Black", "Asian", "Unknown", "Native American/Alaska Native"]
        default_race = race_options.index(st.session_state.get('random_race', 'White')) if st.session_state.get('random_race', 'White') in race_options else 0
        victim_race = st.selectbox("Victim Race", race_options, index=default_race)
        
        victim_count = st.number_input("Victim Count", min_value=1, value=max(1, int(st.session_state.get('random_count', 1))))
        
    with col2:
        decade = st.text_input("Incident Decade", value=st.session_state.get('random_decade', '2010s'))
        victim_age_group = st.text_input("Victim Age Group", value=st.session_state.get('random_age', 'Adult'))
        
        weapon_options = ["Firearm", "Knife", "Blunt Object", "Unknown", "Personal Violence", "Other"]
        default_weapon = weapon_options.index(st.session_state.get('random_weapon', 'Firearm')) if st.session_state.get('random_weapon', 'Firearm') in weapon_options else 0
        weapon_grouped = st.selectbox("Weapon Configuration", weapon_options, index=default_weapon)
        
        agency_options = ["Municipal", "Sheriff", "County", "State", "Other"]
        default_agency = agency_options.index(st.session_state.get('random_agency', 'Municipal')) if st.session_state.get('random_agency', 'Municipal') in agency_options else 0
        agency_grouped = st.selectbox("Agency Handling Case", agency_options, index=default_agency)
        
    relation_options = ["Unknown", "Stranger", "Known", "Intimate Partner", "Family", "Work Related"]
    default_relation = relation_options.index(st.session_state.get('random_relation', 'Unknown')) if st.session_state.get('random_relation', 'Unknown') in relation_options else 0
    relationship_grouped = st.selectbox(
        "Victim-Offender Relationship (Critical Driver)", 
        relation_options,
        index=default_relation
    )

    # --- STEP 4: PREDICTION PIPELINE INFERENCE ---
    st.markdown("---")
    if st.button("Run AI Risk Diagnosis 🚀", type="primary", use_container_width=True):
        # Package into raw dataframe row
        raw_data = {
            'State': state, 'Month': month, 'Victim Sex': victim_sex,
            'Victim Race': victim_race, 'Victim Count': int(victim_count),
            'Decade': decade, 'Victim Age Group': victim_age_group,
            'Weapon Grouped': weapon_grouped, 'Agency Grouped': agency_grouped,
            'Relationship Grouped': relationship_grouped
        }
        
        # 1. Create a blank row with all 91 expected columns set to 0
        final_features = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        # 2. Manually map the inputs to the correct dummy columns
        for column_name, user_input in raw_data.items():
            # Handle numerical columns directly
            if column_name == 'Victim Count':
                if 'Victim Count' in final_features.columns:
                    final_features['Victim Count'] = user_input
                continue
            
            # Handle categorical columns by recreating the Pandas dummy name (e.g., "Weapon Grouped_Firearm")
            dummy_column_name = f"{column_name}_{user_input}"
            
            # If this specific category exists in our trained features, turn it 'ON' (set to 1)
            if dummy_column_name in final_features.columns:
                final_features[dummy_column_name] = 1
        
        # Calculate evaluation arrays
        probabilities = rf_model.predict_proba(final_features)[0]
        cold_risk_score = probabilities[0] * 100
        solve_probability = probabilities[1] * 100
        prediction = rf_model.predict(final_features)[0]

# --- DISPLAY RESULTS ON SCREEN ---
        st.subheader("🧠 AI Diagnostics Inference")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if prediction == 0:
                st.error("⚠️ HIGH COLD CASE RISK")
                st.caption("The model evaluates this signature profile as highly challenging to clear structurally.")
            else:
                st.success("✅ HIGH SOLVE PROBABILITY")
                st.caption("Case signature contains high-signal clearing attributes.")
                
        with res_col2:
            st.metric(label="Cold Case Risk Index", value=f"{cold_risk_score:.1f}%")
            st.metric(label="Solve Probability Index", value=f"{solve_probability:.1f}%")