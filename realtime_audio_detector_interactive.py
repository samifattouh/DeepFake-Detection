# realtime_audio_detector_interactive.py
"""
Allows testing a single audio file (uploaded via Jupyter widget or CLI path)
with a chosen trained model for deepfake detection, measuring the detection time.
"""

import os
import argparse
import time
import joblib
import pandas as pd
import numpy as np
import librosa
import logging
import traceback
import tempfile # For handling uploaded files

# Jupyter-specific imports for upload widget
try:
    from ipywidgets import FileUpload
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False
    # print("ipywidgets not found. File upload in Jupyter Notebook will not be available.")

# --- Configuration ---
TRAINED_MODELS_ROOT = "trained_models_enhanced" # Default, can be overridden by function call

# --- Setup Logging ---
logger = logging.getLogger("RealtimeInteractiveDetector")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG) # Uncomment for more detailed logs
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    try:
        fh = logging.FileHandler("realtime_interactive_detection.log", mode="w") # Overwrite log each run
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    except Exception as e:
        logger.warning(f"Could not create file logger for interactive script: {e}")

# --- Feature Extraction Function (same as before) ---
def extract_features_from_audio(audio_path, target_sr=22050, n_mfcc=40, hop_length=512, n_fft=2048):
    logger.info(f"Extracting features from: {audio_path}")
    features = {}
    try:
        y, sr = librosa.load(audio_path, sr=target_sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        for i in range(n_mfcc):
            features[f"mfcc{i+1}_mean"] = np.mean(mfccs[i,:])
            features[f"mfcc{i+1}_std"] = np.std(mfccs[i,:])
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)
        features["rms_skew"] = pd.Series(rms).skew() if len(rms) > 0 else 0
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)
        features["zcr_skew"] = pd.Series(zcr).skew() if len(zcr) > 0 else 0
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spec_cent_mean"] = np.mean(spec_cent)
        features["spec_cent_std"] = np.std(spec_cent)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spec_bw_mean"] = np.mean(spec_bw)
        features["spec_bw_std"] = np.std(spec_bw)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spec_rolloff_mean"] = np.mean(spec_rolloff)
        features["spec_rolloff_std"] = np.std(spec_rolloff)
        logger.info(f"Successfully extracted {len(features)} features.")
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {e}")
        logger.error(traceback.format_exc())
        return None

# --- Core Detection Function (same as before, processes a file path) ---
def run_audio_detection_from_path(model_set_name, algorithm, audio_file_path, models_root_dir=TRAINED_MODELS_ROOT):
    logger.info(f"Starting detection for audio path: {audio_file_path}")
    logger.info(f"Using model set: {model_set_name}, Algorithm: {algorithm}")

    model_base_path = os.path.join(models_root_dir, model_set_name, model_set_name)
    model_path = os.path.join(model_base_path, f"{algorithm.lower()}_model.joblib")
    scaler_path = os.path.join(model_base_path, "scaler.joblib")
    encoder_path = os.path.join(model_base_path, "label_encoder.joblib")
    features_names_path = os.path.join(model_base_path, "feature_names.joblib")

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        expected_feature_names = joblib.load(features_names_path)
        logger.info("All model components loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error: A required model component file was not found: {e}")
        print(f"ERROR: Could not load model components. Check paths. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading model components: {e}")
        print(f"ERROR: An unexpected error occurred. Check logs. Details: {e}")
        return None

    start_time = time.time()
    extracted_features_dict = extract_features_from_audio(audio_file_path)
    if extracted_features_dict is None: 
        print("ERROR: Feature extraction failed."); return None
    
    try:
        features_df = pd.DataFrame([extracted_features_dict])
        for col in expected_feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[expected_feature_names]
    except Exception as e:
        logger.error(f"Error aligning features: {e}")
        print(f"ERROR: Could not align features. Details: {e}"); return None

    feature_extraction_time = time.time() - start_time
    scaled_features = scaler.transform(features_df)
    scaling_time = time.time() - start_time - feature_extraction_time
    prediction_encoded = model.predict(scaled_features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]
    prediction_time = time.time() - start_time - feature_extraction_time - scaling_time
    total_time = time.time() - start_time

    results = {
        "audio_file_processed": os.path.basename(audio_file_path),
        "model_set_name": model_set_name,
        "algorithm": algorithm.upper(),
        "predicted_label": prediction_label.upper(),
        "feature_extraction_time_s": feature_extraction_time,
        "scaling_time_s": scaling_time,
        "prediction_time_s": prediction_time,
        "total_detection_time_s": total_time
    }
    
    print("\n--- Detection Result ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key.replace("_s", " Time (s)").replace("_", " ").title()}: {value:.4f}")
        else:
            print(f"{key.replace("_", " ").title()}: {value}")
    print("------------------------")
    logger.info("Detection process finished.")
    return results

# --- Jupyter Notebook Interactive Upload and Detection Function ---
def start_interactive_detection(model_set_name, algorithm, models_root_dir=TRAINED_MODELS_ROOT):
    if not IPYWIDGETS_AVAILABLE:
        print("Error: ipywidgets library is not installed or not working. Interactive upload is unavailable.")
        print("Please install it (e.g., pip install ipywidgets) and enable it for your Jupyter environment.")
        return

    print(f"Preparing interactive detection for model set: {model_set_name}, algorithm: {algorithm.upper()}")
    print("Please upload an audio file (.wav, .mp3, etc.) using the widget below:")

    uploader = FileUpload(
        accept=".wav,.mp3,.aac,.flac,.ogg",
        multiple=False  # Allow only single file upload
    )
    display(uploader) # Display the upload widget

    def on_file_upload(change):
        clear_output(wait=True) # Clear the uploader widget and previous output
        print(f"Processing uploaded file for model set: {model_set_name}, algorithm: {algorithm.upper()}")
        
        uploaded_file_info_tuple = uploader.value
        logger.debug(f"uploader.value (raw): {uploaded_file_info_tuple}")
        logger.debug(f"Type of uploader.value: {type(uploaded_file_info_tuple)}")

        if not uploaded_file_info_tuple:
            print("No file uploaded or upload was cleared.")
            logger.info("No file uploaded or upload was cleared by the user.")
            return

        if not isinstance(uploaded_file_info_tuple, tuple) or not uploaded_file_info_tuple:
            print("Error: Uploaded file data is not in the expected tuple format.")
            logger.error(f"Unexpected uploader.value format (not a non-empty tuple): {uploaded_file_info_tuple}")
            return
            
        # The FileUpload widget, when a single file is uploaded, returns a tuple containing a single dictionary.
        # This dictionary holds the metadata and content of the uploaded file.
        uploaded_file_data_dict = uploaded_file_info_tuple[0]
        logger.debug(f"uploaded_file_data_dict (from tuple[0]): {type(uploaded_file_data_dict)}")

        if not isinstance(uploaded_file_data_dict, dict):
            print("Error: Uploaded file data structure is not the expected dictionary.")
            logger.error(f"Unexpected structure for uploaded_file_data_dict: {uploaded_file_data_dict}")
            return
        
        # CORRECTED: Access filename and content from the dictionary using standard keys
        if 'name' not in uploaded_file_data_dict or 'content' not in uploaded_file_data_dict:
            logger.error(f"Uploaded file dictionary is missing 'name' or 'content' key. Keys: {list(uploaded_file_data_dict.keys())}")
            print("Error: Uploaded file data is missing essential information (name or content). Check logs.")
            return

        uploaded_filename = uploaded_file_data_dict['name']
        file_content_raw = uploaded_file_data_dict['content'] # This should be bytes or memoryview
        
        logger.info(f"File '{uploaded_filename}' uploaded. Raw content type: {type(file_content_raw)}, Length: {len(file_content_raw) if hasattr(file_content_raw, '__len__') else 'N/A'}")
        
        file_content_bytes = None
        if isinstance(file_content_raw, bytes):
            file_content_bytes = file_content_raw
            logger.debug(f"Raw content is bytes. Length: {len(file_content_bytes)}")
        elif isinstance(file_content_raw, memoryview):
            file_content_bytes = file_content_raw.tobytes()
            logger.debug(f"Raw content is memoryview, converted to bytes. Length: {len(file_content_bytes)}")
        elif isinstance(file_content_raw, str):
            # This case should ideally not happen if FileUpload widget works as expected for binary files.
            logger.warning(f"Raw content is string. This is unexpected for file content. Attempting latin-1 encoding as a fallback.")
            logger.debug(f"Raw content (string snippet for logging): {file_content_raw[:100]}")
            try:
                file_content_bytes = file_content_raw.encode('latin-1')
                logger.info(f"Encoded string content to bytes using latin-1. New length: {len(file_content_bytes)}")
            except Exception as e:
                logger.error(f"Failed to encode string content to bytes: {e}")
                print("Error: Could not process uploaded file content (string encoding failed). Check logs.")
                return
        else:
            logger.error(f"Unexpected type for file_content_raw: {type(file_content_raw)}. Cannot proceed.")
            print("Error: Uploaded file content has an unexpected type. Check logs.")
            return

        if file_content_bytes is None:
            logger.error("file_content_bytes is None after type checking. This should not happen.")
            print("Internal error processing file content. Check logs.")
            return
            
        logger.info(f"Processed file content. Final bytes length: {len(file_content_bytes)}")

        # Use a temporary file to save the uploaded content for librosa to load
        # Ensure the suffix matches the original file extension for librosa/soundfile
        original_suffix = os.path.splitext(uploaded_filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp_audio_file:
            tmp_audio_file.write(file_content_bytes)
            temp_file_path = tmp_audio_file.name
        
        logger.info(f"Uploaded file content saved temporarily to: {temp_file_path}")

        try:
            run_audio_detection_from_path(model_set_name, algorithm, temp_file_path, models_root_dir)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} deleted.")
            # Re-display the uploader for the next file
            print("\nUpload another file or call start_interactive_detection() again.")
            # display(uploader) # Re-displaying uploader here can cause issues if output is cleared again by new upload.
                              # Better to instruct user to re-run the cell that calls start_interactive_detection.

    uploader.observe(on_file_upload, names='value')

# --- Command-Line Interface Logic (for path-based detection) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Deepfake Audio Detection (CLI or Importable for Interactive Notebook Use)")
    parser.add_argument("--model_set_name", type=str, required=True, 
                        help="Name of the model set directory (e.g., for-norm_training)")
    parser.add_argument("--algorithm", type=str, required=True, 
                        help="Algorithm to use (e.g., lgbm, mlp, lr)")
    parser.add_argument("--audio_file", type=str, 
                        help="Path to the input audio file for CLI mode (.wav, .mp3, etc.)")
    parser.add_argument("--models_root", type=str, default=TRAINED_MODELS_ROOT,
                        help=f"Root directory for trained models (default: {TRAINED_MODELS_ROOT})")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive Jupyter mode (displays upload widget). Ignores --audio_file.")

    args = parser.parse_args()
    
    if args.interactive:
        if IPYWIDGETS_AVAILABLE:
            print("Interactive mode selected. If in Jupyter, call start_interactive_detection() from a cell.")
            print("Example: from realtime_audio_detector_interactive import start_interactive_detection")
            print("         start_interactive_detection(model_set_name=\"for-norm_training\", algorithm=\"lgbm\")")
        else:
            print("ERROR: --interactive mode requires ipywidgets. Please install and configure.")
    elif args.audio_file:
        run_audio_detection_from_path(args.model_set_name, 
                                    args.algorithm, 
                                    args.audio_file, 
                                    models_root_dir=args.models_root)
    else:
        print("Error: You must specify an --audio_file for CLI mode, or use --interactive in a Jupyter environment.")
        parser.print_help()
