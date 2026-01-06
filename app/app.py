# app/app.py
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# We can import directly from bias_debugger because we will
# install our package in editable mode.
try:
    from fairness_troops import BiasAuditor
except ImportError:
    st.error(
        "Could not import BiasAuditor. "
        "Please install the package in editable mode by running: "
        "uv pip install -e ."
    )
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Bias & Fairness Debugger",
    page_icon="🚀",
    layout="wide"
)

# --- Helper Function ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main App ---
st.title("🚀 Bias & Fairness Debugger")
st.markdown(
    "Upload your trained model and test dataset to audit it for fairness."
)

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("1. Upload Your Files")
    
    uploaded_model = st.file_uploader(
        "Upload trained model (.joblib or .pkl)", 
        type=["joblib", "pkl"]
    )
    uploaded_data = st.file_uploader(
        "Upload test dataset (.csv)", 
        type=["csv"]
    )
    
    # Placeholders for configuration
    # --- Initialize Session State for Model/Data if not present ---
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'data' not in st.session_state:
        st.session_state['data'] = None

    # Helpers to clean state when new files are uploaded
    def clear_example_state():
        st.session_state['model'] = None
        st.session_state['data'] = None
        # reset config keys if needed, or let them stay

    # --- Load Example Data Button ---
    st.markdown("---")
    st.subheader("Or Try an Example")
    if st.button("Load Adult Census Example"):
        try:
            import os
            # Paths relative to app/app.py
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, '..', 'data', 'adult_model.joblib')
            data_path = os.path.join(base_dir, '..', 'data', 'adult_test_data.csv')
            
            # Load into session state
            st.session_state['model'] = joblib.load(model_path)
            st.session_state['data'] = pd.read_csv(data_path)
            
            # Set default keys in session state to pre-fill configuration
            st.session_state['target_col'] = 'income'
            st.session_state['sensitive_col'] = 'sex'
            st.session_state['privileged_group'] = 'Male'
            st.session_state['unprivileged_group'] = 'Female'
            
            st.success("Adult Census Example loaded!")
            st.rerun() # Rerun to update the UI immediately
        except Exception as e:
            st.error(f"Error loading example: {e}")

    if uploaded_model:
        try:
            # Use BytesIO to load the model from the uploaded file
            model_bytes = BytesIO(uploaded_model.getvalue())
            st.session_state['model'] = joblib.load(model_bytes)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
    if uploaded_data:
        try:
            st.session_state['data'] = pd.read_csv(uploaded_data)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    # Assign local variables from session state for use in main page
    model = st.session_state['model']
    data = st.session_state['data']


# --- Main Page ---
if model and data is not None:
    st.header("2. Configure Audit")
    
    # Get column names
    all_columns = data.columns.tolist()
    
    # Create two columns for configuration
    col1, col2 = st.columns(2)
    
    with col1:
        target_default_idx = 0
        if 'target_col' in st.session_state and st.session_state['target_col'] in all_columns:
            target_default_idx = all_columns.index(st.session_state['target_col'])

        target_col = st.selectbox(
            "Select Target Variable (e.g., 'loan_approved')", 
            all_columns,
            index=target_default_idx
        )
        
    with col2:
        sensitive_options = [col for col in all_columns if col != target_col]
        sensitive_default_idx = 0
        if 'sensitive_col' in st.session_state and st.session_state['sensitive_col'] in sensitive_options:
            sensitive_default_idx = sensitive_options.index(st.session_state['sensitive_col'])

        sensitive_col = st.selectbox(
            "Select Sensitive Attribute (e.g., 'gender')",
            sensitive_options,
            index=sensitive_default_idx
        )
        
    # Get unique values for the selected sensitive column
    if sensitive_col:
        unique_groups = data[sensitive_col].unique()
        
        if len(unique_groups) < 2:
            st.warning(f"Sensitive attribute '{sensitive_col}' must have at least 2 unique groups.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                priv_default_idx = 0
                if 'privileged_group' in st.session_state and st.session_state['privileged_group'] in unique_groups:
                    # unique_groups is a numpy array, need to handle index carefully or convert to list
                    # It's safer to rely on automatic matching or explicit index finding if unique_groups is list-like
                    pass 
                    # Simpler to just let user select or rely on default logic we implemented.
                    # Actually, let's try to set it if possible:
                    try:
                         # unique_groups might be numpy array
                         priv_default_idx = list(unique_groups).index(st.session_state['privileged_group'])
                    except ValueError:
                        pass

                privileged_group = st.selectbox(
                    "Select Privileged Group",
                    unique_groups,
                    index=priv_default_idx
                )
            with col2:
                unpriv_options = [g for g in unique_groups if g != privileged_group]
                unpriv_default_idx = 0
                if 'unprivileged_group' in st.session_state and st.session_state['unprivileged_group'] in unpriv_options:
                     try:
                         unpriv_default_idx = unpriv_options.index(st.session_state['unprivileged_group'])
                     except ValueError:
                         pass

                unprivileged_group = st.selectbox(
                    "Select Unprivileged Group",
                    unpriv_options,
                    index=unpriv_default_idx
                )

            # --- 3. Run Audit ---
            st.header("3. Run Audit")
            if st.button("Run Fairness Audit", type="primary"):
                
                with st.spinner("Running audit... 🕵️"):
                    try:
                        # Initialize the auditor
                        auditor = BiasAuditor(
                            model=model,
                            dataset=data,
                            target_col=target_col,
                            sensitive_col=sensitive_col,
                            privileged_group=privileged_group,
                            unprivileged_group=unprivileged_group
                        )
                        
                        # Run calculations
                        report = auditor.run_audit()
                        visuals = auditor.get_visuals()
                        
                        st.subheader("📊 Fairness Metrics Report")
                        st.markdown(
                            f"Comparing **{unprivileged_group}** (Unprivileged) "
                            f"vs. **{privileged_group}** (Privileged)"
                        )
                        
                        # Display metrics
                        m1, m2 = st.columns(2)
                        di_val = report.get('disparate_impact')
                        eod_val = report.get('equal_opportunity_diff')
                        
                        m1.metric(
                            label="Disparate Impact (DI)",
                            value=f"{di_val:.3f}",
                            help="Ratio of favorable outcomes for unprivileged vs. privileged. "
                                 "Ideal: 1.0. Flagged if < 0.8."
                        )
                        
                        m2.metric(
                            label="Equal Opportunity Difference (EOD)",
                            value=f"{eod_val:.3f}",
                            help="Difference in True Positive Rates (TPR_unpriv - TPR_priv). "
                                 "Ideal: 0.0."
                        )
                        
                        # Display Visuals
                        st.subheader("Visual Analysis")
                        v1, v2 = st.columns(2)
                        with v1:
                            st.pyplot(visuals.get('group_outcomes'))
                        with v2:
                            st.pyplot(visuals.get('tpr_by_group'))
                            
                        # --- 4. Mitigation ---
                        st.header("4. Mitigation Suggestions")
                        st.subheader("Pre-processing: Reweighting")
                        st.markdown(
                            "You can retrain your model using these `sample_weight` values "
                            "to mitigate bias. This technique gives more weight to "
                            "under-represented groups and outcomes."
                        )
                        
                        weights = auditor.get_mitigation_weights()
                        
                        # Add weights to the original dataframe for context
                        data_with_weights = data.copy()
                        data_with_weights['sample_weight'] = weights
                        
                        st.dataframe(data_with_weights.head())
                        
                        st.download_button(
                            label="Download All Sample Weights as CSV",
                            data=convert_df_to_csv(data_with_weights),
                            file_name="data_with_sample_weights.csv",
                            mime="text/csv",
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred during the audit: {e}")
                        st.exception(e)  # Print full traceback

else:
    st.info(
        "Welcome! Please upload a model and a dataset using the sidebar to begin."
    )
    st.subheader("Don't have files?")
    st.markdown(
        """
        1.  Make sure you have run `uv pip install -e .` in your terminal.
        2.  Run the example script to create a sample model:
            ```bash
            python examples/01_train_sample_model.py
            ```
        3.  Upload `data/sample_model.joblib` and `data/sample_loan_data.csv`.
        """
    )