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
    model = None
    data = None
    
    if uploaded_model:
        try:
            # Use BytesIO to load the model from the uploaded file
            model_bytes = BytesIO(uploaded_model.getvalue())
            model = joblib.load(model_bytes)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
    if uploaded_data:
        try:
            data = pd.read_csv(uploaded_data)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")

# --- Main Page ---
if model and data is not None:
    st.header("2. Configure Audit")
    
    # Get column names
    all_columns = data.columns.tolist()
    
    # Create two columns for configuration
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox(
            "Select Target Variable (e.g., 'loan_approved')", 
            all_columns
        )
        
    with col2:
        sensitive_col = st.selectbox(
            "Select Sensitive Attribute (e.g., 'gender')",
            [col for col in all_columns if col != target_col]
        )
        
    # Get unique values for the selected sensitive column
    if sensitive_col:
        unique_groups = data[sensitive_col].unique()
        
        if len(unique_groups) < 2:
            st.warning(f"Sensitive attribute '{sensitive_col}' must have at least 2 unique groups.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                privileged_group = st.selectbox(
                    "Select Privileged Group",
                    unique_groups,
                    index=0
                )
            with col2:
                unprivileged_group = st.selectbox(
                    "Select Unprivileged Group",
                    [g for g in unique_groups if g != privileged_group],
                    index=0
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