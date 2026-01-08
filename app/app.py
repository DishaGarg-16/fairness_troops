# app/app.py
import streamlit as st
import pandas as pd
import skops.io as sio
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
    
    st.markdown("**1. Upload trained model (.skops)**")
    st.info("For security, we only accept `.skops` files. [Learn how to save your model](#don-t-have-files)")
    uploaded_model = st.file_uploader(
        "Drag and drop file here", 
        type=['skops'], 
        key="model_uploader"
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
            # We assume the user has generated this with skops, or we will generate it on the fly if needed
            # For now, let's point to .skops
            model_path = os.path.join(base_dir, '..', 'data', 'adult_model.skops')
            data_path = os.path.join(base_dir, '..', 'data', 'adult_test_data.csv')
            
            # Load into session state
            # sio.load returns the model object directly when trusted=True
            # Note: For the example we use trusted=True as we trust our own file
            # Retrieve untrusted types from the file and trust them explicitly
            unknown_types = sio.get_untrusted_types(file=model_path)
            st.session_state['model'] = sio.load(model_path, trusted=unknown_types)
            st.session_state['data'] = pd.read_csv(data_path)
            
            # Set default keys in session state to pre-fill configuration
            st.session_state['target_col'] = 'income'
            st.session_state['sensitive_col'] = 'sex'
            st.session_state['privileged_group'] = 'Male'
            st.session_state['unprivileged_group'] = 'Female'
            
            st.success("Adult Census Example loaded!")
            st.rerun() # Rerun to update the UI immediately
        except Exception as e:
            st.error(f"Error loading example: {e}\n(Make sure you have run the updated example generation script!)")

    if uploaded_model:
        try:
            # We don't load the model here in the frontend to avoid security risks on the frontend container.
            # We just pass the bytes to the API. 
            st.session_state['model_file'] = uploaded_model
            # We might still want to load it into session state to unlock the "Configure Audit" UI
            # But we can assume it's valid if uploaded.
            st.session_state['model'] = "Uploaded Model" # Dummy value to satisfy check
            st.success("Model ready to upload!")
        except Exception as e:
            st.error(f"Error preparing model: {e}")
            
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
                
                import requests
                import os
                
                API_URL = os.getenv("API_URL", "http://localhost:8000")
                
                # Variables to store results
                audit_success = False
                audit_results = None
                audit_error = None

                with st.spinner(f"Running audit via {API_URL}... 🕵️"):
                    try:
                        # Prepare files for API
                        # Reset pointers
                        # We use the raw uploaded file object from session state or the uploader
                        # If using example, we might need to serialize it?
                        # Wait, if we use Example, st.session_state['model'] is the actual object.
                        # We need to handle BOTH cases: Uploaded File (bytes) OR Example Object (sklearn object).
                        
                        model_obj = st.session_state.get('model')
                        model_file_content = None
                        
                        if isinstance(model_obj, str) and model_obj == "Uploaded Model":
                             # It was uploaded via file uploader
                             model_file_content = st.session_state.get('model_file', uploaded_model)
                             model_file_content.seek(0)
                        else:
                             # It is a loaded object (from example)
                             # We need to serialize it to skops bytes
                             if model_obj:
                                 buf = BytesIO()
                                 sio.dump(model_obj, buf)
                                 buf.seek(0)
                                 model_file_content = buf
                        
                        if model_file_content is None:
                            st.error("No model found!")
                            st.stop()
                        
                        # Serialize data
                        data_buffer = BytesIO()
                        data.to_csv(data_buffer, index=False)
                        data_buffer.seek(0)
                        
                        files = {
                            'model_file': ('model.skops', model_file_content, 'application/octet-stream'),
                            'data_file': ('data.csv', data_buffer, 'text/csv')
                        }
                        
                        payload = {
                            'target_col': target_col,
                            'sensitive_col': sensitive_col,
                            'privileged_group': str(privileged_group),
                            'unprivileged_group': str(unprivileged_group)
                        }
                        
                        response = requests.post(f"{API_URL}/audit", files=files, data=payload)
                        
                        if response.status_code != 200:
                            audit_error = f"API Error: {response.text}"
                        else:
                            # START ASYNC POLLING
                            task_info = response.json()
                            task_id = task_info.get("task_id")
                            
                            if not task_id:
                                audit_error = "Failed to get task ID from API."
                            else:
                                status_placeholder = st.empty()
                                import time
                                
                                # Poll every 2 seconds
                                while True:
                                    result_response = requests.get(f"{API_URL}/result/{task_id}")
                                    if result_response.status_code != 200:
                                        audit_error = "Error checking task status."
                                        break
                                    
                                    result_data = result_response.json()
                                    state = result_data.get("state")
                                    status_msg = result_data.get("status", "Processing...")
                                    
                                    if state in ["PENDING", "PROGRESS", "STARTED"]:
                                        status_placeholder.info(f"Status: {status_msg}")
                                        time.sleep(2)
                                        
                                    elif state == "SUCCESS":
                                        result_payload = result_data.get("result", {})
                                        if result_payload.get("status") == "error":
                                            audit_error = f"Task Failed: {result_payload.get('error')}"
                                            break
                                        else:
                                            status_placeholder.success("Audit Complete!")
                                            audit_results = result_payload
                                            audit_success = True
                                            break
                                        
                                    elif state == "FAILURE":
                                        audit_error = f"Task Failed: {result_data.get('error')}"
                                        break
                    except Exception as e:
                        audit_error = f"An error occurred during the audit: {e}"
                        # import traceback
                        # st.text(traceback.format_exc())

                # --- Render Results (Outside Spinner) ---
                if audit_error:
                    st.error(audit_error)
                
                if audit_success and audit_results:
                    report = audit_results.get('metrics')
                    predictions = audit_results.get('predictions')
                    mitigation_weights = audit_results.get('mitigation_weights')
                    pdf_b64 = audit_results.get('pdf_report_b64')
                    visuals_res = audit_results.get('visuals', {})
                    
                    perm_imp_b64 = visuals_res.get('feature_importance')
                    pdp_b64 = visuals_res.get('pdp_plot')
                    
                    # PDF Download Button
                    if pdf_b64:
                        import base64
                        pdf_bytes = base64.b64decode(pdf_b64)
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name="fairness_audit_report.pdf",
                            mime="application/pdf"
                        )
                    
                    st.subheader("📊 Fairness Metrics Report")

                    if not report:
                        st.stop()

                    st.markdown(
                        f"Comparing **{unprivileged_group}** (Unprivileged) "
                        f"vs. **{privileged_group}** (Privileged)"
                    )
                    
                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.metric(
                            label="Disparate Impact (DI)",
                            value=f"{report.get('disparate_impact', 0):.3f}",
                            help="Ratio of favorable outcomes for unprivileged vs. privileged. Ideal: 1.0. Flagged if < 0.8."
                        )
                        st.metric(
                            label="Statistical Parity Diff",
                            value=f"{report.get('statistical_parity_diff', 0):.3f}",
                            help="Difference in positive outcome rates. Ideal: 0."
                        )
                        st.metric(
                            label="Theil Index",
                            value=f"{report.get('theil_index', 0):.3f}",
                            help="Individual fairness metric. Lower is better (0=perfect equality)."
                        )

                    with m2:
                        st.metric(
                            label="Equal Opp. Diff (EOD)",
                            value=f"{report.get('equal_opportunity_diff', 0):.3f}",
                            help="Difference in True Positive Rates. Ideal: 0.0."
                        )
                        st.metric(
                            label="Avg Abs Odds Diff",
                            value=f"{report.get('avg_abs_odds_diff', 0):.3f}",
                            help="Average of absolute difference in FPR and TPR."
                        )

                    with m3:
                        st.metric(
                            label="False Positive Rate Diff",
                            value=f"{report.get('false_positive_rate_diff', 0):.3f}",
                            help="Difference in FPR (Unpriv - Priv)."
                        )
                        st.metric(
                            label="False Negative Rate Diff",
                            value=f"{report.get('false_negative_rate_diff', 0):.3f}",
                            help="Difference in FNR (Unpriv - Priv)."
                        )
                    
                    # Display Visuals
                    # We need to reconstruct y_pred Series and sensitive_features Series
                    # to use the local visuals library
                    from fairness_troops import visuals
                    
                    # Reconstruct y_pred
                    y_pred_series = pd.Series(
                        predictions, 
                        index=data.index, 
                        name="predictions"
                    )
                    
                    # Reconstruct y_true and sensitive_features
                    y_true = data[target_col]
                    sensitive_features = data[sensitive_col]
                    
                    st.subheader("Visual Analysis")
                    v1, v2 = st.columns(2)
                    with v1:
                        fig1 = visuals.plot_group_outcomes(
                            y_pred_series,
                            sensitive_features,
                            title=f"Favorable Outcome Rate ({target_col}=1)"
                        )
                        st.pyplot(fig1)
                    with v2:
                        fig2 = visuals.plot_tpr_by_group(
                            y_true,
                            y_pred_series,
                            sensitive_features,
                            title=f"True Positive Rate by {sensitive_col}"
                        )
                        st.pyplot(fig2)
                        
                    # Display Explanations
                    st.subheader("🔍 Model Explainability")
                    import base64
                    
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        if perm_imp_b64:
                            st.markdown("**Permutation Feature Importance**")
                            st.markdown("Shows which features most affect model performance when shuffled.")
                            perm_img = base64.b64decode(perm_imp_b64)
                            st.image(perm_img, use_column_width=True)
                    
                    with c2:
                        if pdp_b64:
                            st.markdown(f"**Partial Dependence Plot (PDP)**")
                            st.markdown(f"Shows relationship between *{sensitive_col}* and predicted outcome.")
                            pdp_img = base64.b64decode(pdp_b64)
                            st.image(pdp_img, use_column_width=True)
                        
                    # --- 4. Mitigation ---
                    st.header("4. Mitigation Suggestions")
                    st.subheader("Pre-processing: Reweighting")
                    st.markdown(
                        "You can retrain your model using these `sample_weight` values "
                        "to mitigate bias. This technique gives more weight to "
                        "under-represented groups and outcomes."
                    )
                    
                    # Add weights to the original dataframe for context
                    data_with_weights = data.copy()
                    data_with_weights['sample_weight'] = mitigation_weights
                    
                    st.dataframe(data_with_weights.head())
                    
                    st.download_button(
                        label="Download All Sample Weights as CSV",
                        data=convert_df_to_csv(data_with_weights),
                        file_name="data_with_sample_weights.csv",
                        mime="text/csv",
                    )

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
        3.  Upload `data/sample_model.skops` and `data/sample_loan_data.csv`.
        """
    )