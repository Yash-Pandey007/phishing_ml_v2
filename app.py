# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import socket
import re
import tldextract # Import tldextract

# --- Load Preprocessing Artifacts and Model ---
# (Keep the loading functions exactly the same as before)
@st.cache_resource
def load_model(model_path='best_model.pkl'):
    """Loads the pickled machine learning model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler(scaler_path='robust_scaler.pkl'):
    """Loads the pickled RobustScaler object."""
    try:
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at '{scaler_path}'. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_resource
def load_pca(pca_path='pca_transformer.pkl'):
    """Loads the pickled PCA transformer object."""
    try:
        with open(pca_path, 'rb') as file:
            pca = pickle.load(file)
        return pca
    except FileNotFoundError:
        st.error(f"Error: PCA file not found at '{pca_path}'. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading PCA transformer: {e}")
        return None

@st.cache_data # Use cache_data for data like lists/dicts
def load_encoders(encoder_path='label_encoders.pkl'):
    """Loads the pickled dictionary of LabelEncoders."""
    try:
        with open(encoder_path, 'rb') as file:
            encoders = pickle.load(file)
        return encoders
    except FileNotFoundError:
        st.error(f"Error: Label encoders file not found at '{encoder_path}'. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        return None

@st.cache_data
def load_features_kept(features_path='features_kept.pkl'):
    """Loads the pickled list of feature names kept after VIF removal."""
    try:
        with open(features_path, 'rb') as file:
            features_kept = pickle.load(file)
        return features_kept
    except FileNotFoundError:
        st.error(f"Error: Features list file ('{features_path}') not found. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading features list: {e}")
        return None


# --- Feature Extraction Function (Modified to use tldextract) ---
def extract_url_features(url):
    """
    Extracts features from a URL string using urlparse and tldextract.
    MUST generate features consistent with the training data *before* VIF removal.
    Features not derivable from URL alone are assigned default values (0 or -1).
    """
    features = {}
    try:
        # Handle potential missing scheme by adding http:// if necessary
        if not re.match(r'^[a-zA-Z]+://', url):
             url = 'http://' + url
             st.warning(f"Assuming 'http://' for URL: {url}", icon="⚠️")

        parsed = urlparse(url)
        # Use tldextract to get subdomain, domain, suffix
        extracted_domain_parts = tldextract.extract(url)
        domain_name = extracted_domain_parts.domain
        subdomain_name = extracted_domain_parts.subdomain
        suffix_name = extracted_domain_parts.suffix
        registered_domain = extracted_domain_parts.registered_domain # domain + suffix

        # --- Populate features ---
        # Basic parsing features needed by encoders or kept features
        features['scheme'] = parsed.scheme if parsed.scheme else 'unknown'
        features['netloc'] = parsed.netloc if parsed.netloc else 'unknown' # Keep original netloc if needed
        features['path'] = parsed.path if parsed.path else '/'

        # Features directly calculable from URL (using parsed and tldextract)
        features['url_length'] = len(url)
        # Use registered_domain length if more appropriate, else keep original hostname logic
        features['length_hostname'] = len(parsed.netloc.split(':')[0]) # Keep consistency if training used this
        # features['length_registered_domain'] = len(registered_domain) # Add if needed

        features['ip'] = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", parsed.netloc.split(':')[0]) else 0
        features['is_ip'] = features['ip'] # Ensure is_ip exists if required

        features['nb_dots'] = url.count('.') # Count dots in the full URL for potential consistency
        features['nb_hyphens'] = url.count('-')
        features['nb_at'] = url.count('@')
        features['nb_qm'] = url.count('?')
        features['nb_and'] = url.count('&')
        features['nb_or'] = url.count('|')
        features['nb_eq'] = url.count('=')
        features['nb_underscore'] = url.count('_')
        features['nb_tilde'] = url.count('~')
        features['nb_percent'] = url.count('%')
        features['nb_slash'] = url.count('/')
        features['nb_star'] = url.count('*')
        features['nb_colon'] = url.count(':')
        features['nb_comma'] = url.count(',')
        features['nb_semicolumn'] = url.count(';')
        features['nb_dollar'] = url.count('$')
        features['nb_space'] = url.count(' ')

        # Features refined by tldextract
        features['nb_www'] = 1 if 'www' in subdomain_name.lower().split('.') else 0
        features['nb_com'] = 1 if suffix_name.lower() == 'com' else 0 # Check specific suffix
        # Calculate nb_subdomains based on tldextract's subdomain part
        features['nb_subdomains'] = subdomain_name.count('.') + 1 if subdomain_name else 0

        features['nb_dslash'] = url.count('//')
        features['http_in_path'] = 1 if 'http' in parsed.path.lower() else 0
        features['https_token'] = 1 if parsed.scheme == 'https' else 0

        # Ratio features (handle division by zero)
        digits_url = sum(c.isdigit() for c in url)
        digits_host = sum(c.isdigit() for c in parsed.netloc) # Use netloc for consistency if trained that way
        features['ratio_digits_url'] = digits_url / features['url_length'] if features['url_length'] > 0 else 0
        features['ratio_digits_host'] = digits_host / features['length_hostname'] if features['length_hostname'] > 0 else 0

        # Path and query features
        path_segments = [seg for seg in parsed.path.split('/') if seg]
        features['num_path_segments'] = len(path_segments)
        try:
            query_params = parse_qs(parsed.query)
            features['num_query_params'] = len(query_params)
        except Exception:
             features['num_query_params'] = 0

        # --- Default values for features NOT directly derivable from URL ---
        # It's crucial these match the defaults/logic used if these features were kept
        features.setdefault('punycode', 0)
        features.setdefault('port', 0)
        features.setdefault('tld_in_path', 0)
        # Refine tld_in_subdomain using tldextract parts
        features['tld_in_subdomain'] = 1 if suffix_name and suffix_name in subdomain_name else 0
        features.setdefault('abnormal_subdomain', 0) # Keep default unless training logic known
        # features['nb_subdomains'] is now calculated above using tldextract
        features.setdefault('prefix_suffix', 0)
        features.setdefault('random_domain', 0)
        features.setdefault('shortening_service', 0)
        features.setdefault('path_extension', 0)
        features.setdefault('nb_redirection', 0)
        features.setdefault('nb_external_redirection', 0)
        features.setdefault('length_words_raw', 0)
        features.setdefault('char_repeat', 0)
        features.setdefault('shortest_words_raw', 0)
        features.setdefault('shortest_word_host', 0)
        features.setdefault('shortest_word_path', 0)
        features.setdefault('longest_words_raw', 0)
        features.setdefault('longest_word_host', 0)
        features.setdefault('longest_word_path', 0)
        features.setdefault('avg_words_raw', 0)
        features.setdefault('avg_word_host', 0)
        features.setdefault('avg_word_path', 0)
        features.setdefault('phish_hints', 0)
        features.setdefault('domain_in_brand', 0)
        features.setdefault('brand_in_subdomain', 0)
        features.setdefault('brand_in_path', 0)
        features.setdefault('suspecious_tld', 0)
        features.setdefault('statistical_report', 0)
        features.setdefault('nb_hyperlinks', 0)
        features.setdefault('ratio_intHyperlinks', 0)
        features.setdefault('ratio_extHyperlinks', 0)
        features.setdefault('ratio_nullHyperlinks', 0)
        features.setdefault('nb_extCSS', 0)
        features.setdefault('ratio_intRedirection', 0)
        features.setdefault('ratio_extRedirection', 0)
        features.setdefault('ratio_intErrors', 0)
        features.setdefault('ratio_extErrors', 0)
        features.setdefault('login_form', 0)
        features.setdefault('external_favicon', 0)
        features.setdefault('links_in_tags', 0)
        features.setdefault('submit_email', 0)
        features.setdefault('ratio_intMedia', 0)
        features.setdefault('ratio_extMedia', 0)
        features.setdefault('sfh', 0)
        features.setdefault('iframe', 0)
        features.setdefault('popup_window', 0)
        features.setdefault('safe_anchor', 0)
        features.setdefault('onmouseover', 0)
        features.setdefault('right_clic', 0)
        features.setdefault('empty_title', 0)
        features.setdefault('domain_in_title', 0)
        features.setdefault('domain_with_copyright', 0)
        features.setdefault('whois_registered_domain', 0)
        features.setdefault('domain_registration_length', -1)
        features.setdefault('domain_age', -1)
        features.setdefault('web_traffic', 0)
        features.setdefault('dns_record', 0)
        features.setdefault('google_index', 0)
        features.setdefault('page_rank', 0)


    except Exception as e:
        st.error(f"Error parsing URL or extracting features: {e}")
        return None

    return features

# --- Log Transform Function (Keep the same as before) ---
def logtrans(data):
    """Applies log transform, handling non-positive values by adding a constant."""
    data_tf = data.copy()
    # Ensure data is numeric, coerce errors quietly for now
    data_tf = data_tf.apply(pd.to_numeric, errors='coerce')

    for col in data_tf.columns:
        if pd.api.types.is_numeric_dtype(data_tf[col]): # Check if column is numeric
            min_val = data_tf[col].min() # Min of the single row input
            if np.isnan(min_val): # Handle case where coercion resulted in NaN
                continue
            if min_val <= 0:
                # Add a constant = 1 - min_val ensures the smallest value becomes 1
                constant = 1.0 - min_val
                data_tf[col] = np.log(data_tf[col] + constant)
            else:
                 data_tf[col] = np.log(data_tf[col])
        # else: # Skip non-numeric columns if any remain (shouldn't for features_kept)
            # st.warning(f"Column '{col}' is not numeric, skipping log transform.")
    return data_tf

# --- Streamlit App Layout ---
st.set_page_config(page_title="Phishing URL Detector", layout="wide")
st.title("🎣 Phishing URL Detector")
st.markdown("""
Enter a URL below to check if it's likely to be a phishing attempt using a pre-trained machine learning model. Enhanced parsing provided by `tldextract`.
""")
st.info("""
**Disclaimer:**
*   This tool provides a prediction based on URL characteristics learned during training. It may not be 100% accurate.
*   Features not directly derivable from the URL string (e.g., domain age, page rank, web traffic) are assigned default values (0 or -1), which *might* impact accuracy.
*   **Consistency Warning:** If `tldextract` was not used during model training, the feature values generated now might differ slightly, potentially affecting prediction accuracy.
*   Always exercise caution and use multiple verification methods before trusting a suspicious link or website.
""", icon="ℹ️")


# --- Load Artifacts ---
model = load_model()
scaler = load_scaler()
pca = load_pca()
encoders = load_encoders()
features_kept = load_features_kept() # Load the list of features

# --- User Input ---
url_input = st.text_input("Enter the full URL to check:", placeholder="e.g., https://www.google.com")

if st.button("Check URL", type="primary"):
    # --- Input Validation and Artifact Check ---
    if not url_input:
        st.warning("Please enter a URL.")
    elif not model or not scaler or not pca or not encoders or not features_kept:
        st.error("One or more necessary components (model, scaler, PCA, encoders, feature list) failed to load. Cannot proceed.")
    else:
        with st.spinner("Analyzing URL and making prediction..."):
            # --- Prediction Pipeline ---
            try:
                # 1. Extract Features using updated function
                features = extract_url_features(url_input)
                if not features:
                    st.error("Failed to extract features from the provided URL.")
                    st.stop() # Stop if extraction fails

                # 2. Create DataFrame (single row)
                input_df = pd.DataFrame([features])

                # 3. Apply Label Encoding (handle unseen values)
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        # Map known categories, mark new ones as -1 (or another indicator)
                        input_df[col] = input_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                    # else: # Don't warn if encoder column wasn't extracted (might not be needed)
                         # st.warning(f"Encoded column '{col}' not found in extracted features.")


                # 4. Filter and Order Columns based on 'features_kept.pkl'
                input_df_filtered = pd.DataFrame(columns=features_kept) # Create empty df with correct columns/order
                for col in features_kept:
                    if col in input_df.columns:
                        input_df_filtered[col] = input_df[col]
                    else:
                        # If a required feature wasn't extracted, fill with 0 (or a better default)
                        st.warning(f"Required feature '{col}' missing after extraction, filling with 0. Check extraction logic & features_kept.pkl.")
                        input_df_filtered[col] = 0
                input_df_filtered = input_df_filtered.astype(float) # Ensure numeric type before transforms


                # 5. Apply Log Transform
                log_input_df = logtrans(input_df_filtered) # Pass the filtered df
                if log_input_df.isnull().values.any():
                     st.warning("NaN values detected after log transform, filling with 0.")
                     log_input_df = log_input_df.fillna(0)


                # 6. Apply Scaling
                scaled_input = scaler.transform(log_input_df)


                # 7. Apply PCA
                pca_input = pca.transform(scaled_input)


                # 8. Predict
                prediction = model.predict(pca_input)
                probability = model.predict_proba(pca_input)

                is_phishing = prediction[0] == 1 # Assuming 1 = Phishing
                phishing_prob = probability[0][1]
                legit_prob = probability[0][0]

                # --- Display Results ---
                st.subheader("Prediction Result:")
                if is_phishing:
                    st.error(f"🚨 This URL is classified as **Phishing**.", icon="🚨")
                    progress_value = float(phishing_prob)
                    st.progress(progress_value)
                    st.metric(label="Confidence (Phishing)", value=f"{phishing_prob*100:.2f}%")
                else:
                    st.success(f"✅ This URL is classified as **Legitimate**.", icon="✅")
                    progress_value = float(legit_prob)
                    st.progress(progress_value)
                    st.metric(label="Confidence (Legitimate)", value=f"{legit_prob*100:.2f}%")

                # Optional: Show intermediate data
                with st.expander("Show PCA Input Vector"):
                     st.dataframe(pd.DataFrame(pca_input, columns=[f'PC{i+1}' for i in range(pca_input.shape[1])]))
                with st.expander("Show Log-Transformed Features (Input to Scaler)"):
                    st.dataframe(log_input_df)
                # with st.expander("Show Initial Extracted Features Dictionary"):
                #     st.json(features) # Display the raw extracted dict


            except Exception as e:
                st.error("An error occurred during the prediction pipeline.")
                st.exception(e) # Display the full error traceback for debugging

# Add a footer (optional)
st.markdown("---")
st.caption("Developed using Streamlit, Scikit-learn/XGBoost, and tldextract.")