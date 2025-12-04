# # import os
# # import tempfile
# # import joblib
# # import cv2
# # import numpy as np
# # import pandas as pd
# # from io import BytesIO
# # import base64
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt

# # from flask import Flask, request, jsonify
# # from flask_cors import CORS 

# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# # from sklearn.svm import SVR
# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.linear_model import Ridge
# # from sklearn.metrics import r2_score

# # try:
# #     from xgboost import XGBRegressor
# #     XG_AVAILABLE = True
# # except:
# #     XG_AVAILABLE = False

# # app = Flask(__name__)
# # CORS(app)

# # MODEL_PATH = "best_ensemble_model.joblib"
# # CSV_SAMPLE_PATH = "nm RGB.csv"


# # # ---------------------------
# # # TRAIN / LOAD MODEL
# # # ---------------------------
# # def train_and_select_ensemble(csv_path=CSV_SAMPLE_PATH, force_retrain=False):

# #     if os.path.exists(MODEL_PATH) and not force_retrain:
# #         print("Loading saved model...")
# #         return joblib.load(MODEL_PATH)
    
# #     if not os.path.exists(csv_path):
# #         raise FileNotFoundError(f"CSV not found: {csv_path}")

# #     print("Training ensemble model...")
# #     df = pd.read_csv(csv_path)
# #     X = df.drop(columns=['nm'])
# #     y = df['nm'].values

# #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# #     base_models = {
# #         "rf": RandomForestRegressor(n_estimators=250, random_state=42),
# #         "gbr": GradientBoostingRegressor(random_state=42),
# #         "svr": SVR(kernel="rbf"),
# #         "knn": KNeighborsRegressor(n_neighbors=5),
# #         "ridge": Ridge(alpha=1.0)
# #     }

# #     if XG_AVAILABLE:
# #         base_models["xgb"] = XGBRegressor(
# #             n_estimators=200, 
# #             learning_rate=0.08, 
# #             random_state=42
# #         )

# #     scores = {}
# #     for name, model in base_models.items():
# #         model.fit(X_train, y_train)
# #         pred = model.predict(X_val)
# #         scores[name] = r2_score(y_val, pred)
# #         print(f"{name} R2 = {scores[name]:.4f}")

# #     r2_vals = np.array([max(0, s) for s in scores.values()])
# #     weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

# #     def weighted_predict(X_):
# #         total = 0
# #         for w, m in zip(weights, base_models.values()):
# #             total += w * m.predict(X_)
# #         return total

# #     weighted_r2 = r2_score(y_val, weighted_predict(X_val))

# #     estimators = [(n, m) for n, m in base_models.items()]
# #     stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
# #     stack.fit(X_train, y_train)
# #     stack_r2 = r2_score(y_val, stack.predict(X_val))

# #     if stack_r2 >= weighted_r2:
# #         model_obj = {"type": "stacking", "model": stack}
# #         print("Using stacking model")
# #     else:
# #         model_obj = {"type": "weighted", "models": base_models, "weights": weights}
# #         print("Using weighted ensemble")

# #     joblib.dump(model_obj, MODEL_PATH)
# #     return model_obj


# # # Load model
# # model_obj = train_and_select_ensemble()



# # # ---------------------------
# # # ENSEMBLE PREDICT
# # # ---------------------------
# # def ensemble_predict(model_obj, X):
# #     if model_obj["type"] == "stacking":
# #         return model_obj["model"].predict(X)
# #     else:
# #         preds = np.zeros(len(X))
# #         for w, m in zip(model_obj["weights"], model_obj["models"].values()):
# #             preds += w * m.predict(X)
# #         return preds


# # # ---------------------------
# # # ROI CROPPING
# # # ---------------------------
# # def detect_and_crop_roi(frame, min_area=1000, brightness_thresh=30):
# #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# #     v = hsv[:, :, 2]
# #     mask = v > brightness_thresh

# #     if mask.sum() < min_area:
# #         return frame

# #     ys, xs = np.where(mask)
# #     y1, y2 = max(0, ys.min()-5), min(frame.shape[0], ys.max()+5)
# #     x1, x2 = max(0, xs.min()-5), min(frame.shape[1], xs.max()+5)

# #     cropped = frame[y1:y2, x1:x2]
# #     return cropped if cropped.size > 0 else frame



# # # ---------------------------
# # # VIDEO ANALYSIS
# # # ---------------------------
# # def analyze_video_frames(video_path, model_obj, target_fps=26, grid=8):

# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened():
# #         return None

# #     fps = cap.get(cv2.CAP_PROP_FPS) or 30
# #     step = max(1, int(round(fps / target_fps)))

# #     wavelengths = []
# #     intensities = []
# #     frame_idx = 0

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         frame_idx += 1
# #         if (frame_idx - 1) % step != 0:
# #             continue

# #         frame = detect_and_crop_roi(frame)

# #         h, w, _ = frame.shape
# #         patch_h = h // grid
# #         patch_w = w // grid

# #         patch_rgbs = []
# #         patch_ints = []

# #         for i in range(grid):
# #             for j in range(grid):
# #                 y1, y2 = i * patch_h, (i+1) * patch_h
# #                 x1, x2 = j * patch_w, (j+1) * patch_w

# #                 patch = frame[y1:y2, x1:x2]
# #                 if patch.size == 0:
# #                     continue

# #                 gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
# #                 patch_ints.append(float(np.mean(gray)))

# #                 b, g, r = cv2.resize(patch, (1,1))[0,0]
# #                 patch_rgbs.append([r, g, b])

# #         if len(patch_rgbs) == 0:
# #             continue

# #         mean_rgb = np.mean(np.array(patch_rgbs), axis=0)
# #         intensity = float(np.mean(patch_ints))

# #         nm_pred = ensemble_predict(model_obj, mean_rgb.reshape(1, -1))[0]

# #         wavelengths.append(float(nm_pred))
# #         intensities.append(intensity)

# #     cap.release()

# #     if len(wavelengths) == 0:
# #         return None

# #     return {
# #         "avg_nm": float(np.mean(wavelengths)),
# #         "peak_nm": float(np.max(wavelengths)),
# #         "min_nm": float(np.min(wavelengths)),
# #         "max_nm": float(np.max(wavelengths)),
# #         "wavelength_list": wavelengths,
# #         "intensity_list": intensities
# #     }



# # # ---------------------------
# # # PLOT
# # # ---------------------------
# # def plot_wavelength_intensity(wavelengths, intensities):
# #     plt.figure(figsize=(8,4))
# #     plt.plot(wavelengths, intensities)
# #     plt.xlabel("Wavelength (nm)")
# #     plt.ylabel("Intensity")
# #     plt.title("Wavelength vs Intensity")
# #     plt.tight_layout()

# #     buf = BytesIO()
# #     plt.savefig(buf, format="png")
# #     plt.close()
# #     buf.seek(0)
# #     return "data:image/png;base64," + base64.b64encode(buf.read()).decode()



# # @app.route("/predict", methods=["POST"])
# # def predict():
# #     if "video" not in request.files:
# #         return jsonify({"error": "No video uploaded"}), 400

# #     file = request.files["video"]

# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
# #         file.save(tmp.name)
# #         video_path = tmp.name

# #     result = analyze_video_frames(video_path, model_obj)
# #     os.remove(video_path)

# #     if result is None:
# #         return jsonify({"error": "Failed to process video"}), 500

# #     result["plot_base64"] = plot_wavelength_intensity(
# #         result["wavelength_list"], result["intensity_list"]
# #     )

# #     return jsonify(result)



# # if __name__ == "__main__":
# #     app.run(host="127.0.0.1", port=5003, debug=True)
# # train.py
# import os
# import tempfile
# import joblib
# import cv2
# import numpy as np
# import pandas as pd
# import json
# import csv
# from datetime import datetime
# from io import BytesIO
# import base64
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score

# try:
#     from xgboost import XGBRegressor
#     XG_AVAILABLE = True
# except:
#     XG_AVAILABLE = False

# app = Flask(__name__)
# CORS(app)

# MODEL_PATH = "best_ensemble_model.joblib"
# CSV_SAMPLE_PATH = "nm RGB.csv"
# SAVED_RESULTS_CSV = "saved_results.csv"


# # ---------------------------
# # Train or load ensemble model
# # ---------------------------
# def train_and_select_ensemble(csv_path=CSV_SAMPLE_PATH, force_retrain=False):
#     if os.path.exists(MODEL_PATH) and not force_retrain:
#         print("Loading saved model...")
#         return joblib.load(MODEL_PATH)

#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSV not found: {csv_path}")

#     print("Training ensemble model...")
#     df = pd.read_csv(csv_path)
#     X = df.drop(columns=["nm"])
#     y = df["nm"].values

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     base_models = {
#         "rf": RandomForestRegressor(n_estimators=250, random_state=42),
#         "gbr": GradientBoostingRegressor(random_state=42),
#         "svr": SVR(kernel="rbf"),
#         "knn": KNeighborsRegressor(n_neighbors=5),
#         "ridge": Ridge(alpha=1.0)
#     }

#     if XG_AVAILABLE:
#         base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)

#     scores = {}
#     for name, model in base_models.items():
#         model.fit(X_train, y_train)
#         pred = model.predict(X_val)
#         scores[name] = r2_score(y_val, pred)
#         print(f"{name} R2 = {scores[name]:.4f}")

#     r2_vals = np.array([max(0, s) for s in scores.values()])
#     weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

#     def weighted_predict(X_):
#         total = 0
#         for w, m in zip(weights, base_models.values()):
#             total += w * m.predict(X_)
#         return total

#     weighted_r2 = r2_score(y_val, weighted_predict(X_val))

#     estimators = [(n, m) for n, m in base_models.items()]
#     stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
#     stack.fit(X_train, y_train)
#     stack_r2 = r2_score(y_val, stack.predict(X_val))

#     if stack_r2 >= weighted_r2:
#         model_obj = {"type": "stacking", "model": stack}
#         print("Using stacking model.")
#     else:
#         model_obj = {"type": "weighted", "models": base_models, "weights": weights}
#         print("Using weighted ensemble.")

#     joblib.dump(model_obj, MODEL_PATH)
#     print(f"Saved model to {MODEL_PATH}")
#     return model_obj


# # Load model on startup (will train if MODEL_PATH missing and CSV present)
# try:
#     model_obj = train_and_select_ensemble()
# except Exception as e:
#     print("Model load/train failed:", e)
#     model_obj = None


# # ---------------------------
# # Ensemble predict helper
# # ---------------------------
# def ensemble_predict(model_obj, X):
#     if model_obj is None:
#         raise RuntimeError("Model is not available.")
#     if model_obj.get("type") == "stacking":
#         return model_obj["model"].predict(X)
#     else:
#         preds = np.zeros(len(X))
#         for w, m in zip(model_obj["weights"], model_obj["models"].values()):
#             preds += w * m.predict(X)
#         return preds


# # ---------------------------
# # ROI cropping
# # ---------------------------
# def detect_and_crop_roi(frame, min_area=1000, brightness_thresh=30):
#     """
#     Detect bright region by Value channel and crop to bounding box.
#     Returns the cropped frame or original if no sufficient bright area found.
#     """
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     v = hsv[:, :, 2]
#     mask = v > brightness_thresh

#     if mask.sum() < min_area:
#         return frame

#     ys, xs = np.where(mask)
#     y1, y2 = max(0, int(ys.min()) - 5), min(frame.shape[0], int(ys.max()) + 5)
#     x1, x2 = max(0, int(xs.min()) - 5), min(frame.shape[1], int(xs.max()) + 5)

#     cropped = frame[y1:y2, x1:x2]
#     return cropped if cropped.size > 0 else frame


# # ---------------------------
# # Analyze video frames (no weighting)
# # ---------------------------
# def analyze_video_frames(video_path, model_obj, target_fps=26, grid=8):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None

#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     step = max(1, int(round(fps / target_fps)))

#     wavelengths = []
#     intensities = []
#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1
#         if (frame_idx - 1) % step != 0:
#             continue

#         # crop ROI to remove black background
#         try:
#             frame = detect_and_crop_roi(frame)
#         except Exception:
#             # if ROI fails, continue with original frame
#             pass

#         h, w, _ = frame.shape
#         patch_h = max(1, h // grid)
#         patch_w = max(1, w // grid)

#         patch_rgbs = []
#         patch_ints = []

#         for i in range(grid):
#             for j in range(grid):
#                 y1, y2 = i * patch_h, min((i + 1) * patch_h, h)
#                 x1, x2 = j * patch_w, min((j + 1) * patch_w, w)
#                 patch = frame[y1:y2, x1:x2]
#                 if patch.size == 0:
#                     continue

#                 gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
#                 patch_ints.append(float(np.mean(gray)))

#                 small = cv2.resize(patch, (1, 1), interpolation=cv2.INTER_AREA)
#                 b, g, r = [float(x) for x in small[0, 0]]
#                 patch_rgbs.append([r, g, b])

#         if len(patch_rgbs) == 0:
#             continue

#         mean_rgb = np.mean(np.array(patch_rgbs), axis=0)
#         frame_intensity = float(np.mean(patch_ints))

#         try:
#             nm_pred = ensemble_predict(model_obj, mean_rgb.reshape(1, -1))[0]
#         except Exception:
#             nm_pred = float("nan")

#         wavelengths.append(float(nm_pred))
#         intensities.append(frame_intensity)

#     cap.release()

#     # filter out NaNs
#     arr = np.array(wavelengths)
#     mask_valid = ~np.isnan(arr)
#     valid_wavelengths = arr[mask_valid].tolist()
#     valid_intensities = np.array(intensities)[mask_valid].tolist()

#     if len(valid_wavelengths) == 0:
#         return None

#     return {
#         "avg_nm": float(np.mean(valid_wavelengths)),
#         "peak_nm": float(np.max(valid_wavelengths)),
#         "min_nm": float(np.min(valid_wavelengths)),
#         "max_nm": float(np.max(valid_wavelengths)),
#         "wavelength_list": valid_wavelengths,
#         "intensity_list": valid_intensities
#     }


# # ---------------------------
# # Plot function (Intensity on X, Wavelength on Y)
# # ---------------------------
# def plot_wavelength_intensity(wavelengths, intensities):
#     plt.figure(figsize=(8, 4))
#     plt.scatter(intensities, wavelengths, s=10)
#     plt.xlabel("Intensity")
#     plt.ylabel("Wavelength (nm)")
#     plt.title("Intensity vs Wavelength")
#     plt.tight_layout()

#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     plt.close()
#     buf.seek(0)
#     return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# # ---------------------------
# # Save results to CSV
# # ---------------------------
# def save_results_to_csv(sample_name, result):
#     """
#     Appends a row to SAVED_RESULTS_CSV containing:
#     sample_name, timestamp, avg_nm, peak_nm, min_nm, max_nm,
#     wavelength_list (json string), intensity_list (json string), plot_base64
#     """
#     file_path = SAVED_RESULTS_CSV
#     row = {
#         "timestamp": datetime.utcnow().isoformat(),
#         "sample_name": sample_name,
#         "avg_nm": result.get("avg_nm"),
#         "peak_nm": result.get("peak_nm"),
#         "min_nm": result.get("min_nm"),
#         "max_nm": result.get("max_nm"),
#         "wavelength_list": json.dumps(result.get("wavelength_list", [])),
#         "intensity_list": json.dumps(result.get("intensity_list", [])),
#         "plot_base64": result.get("plot_base64", "")
#     }

#     file_exists = os.path.exists(file_path)
#     with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=row.keys())
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)


# # ---------------------------
# # API: predict (POST video)
# # ---------------------------
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "video" not in request.files:
#         return jsonify({"error": "No video uploaded"}), 400

#     sample_name = request.form.get("sample_name", "Unknown")

#     file = request.files["video"]
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         file.save(tmp.name)
#         video_path = tmp.name

#     try:
#         result = analyze_video_frames(video_path, model_obj)
#     finally:
#         try:
#             os.remove(video_path)
#         except Exception:
#             pass

#     if result is None:
#         return jsonify({"error": "Failed to process video"}), 500

#     # attach plot
#     result["plot_base64"] = plot_wavelength_intensity(result["wavelength_list"], result["intensity_list"])

#     # save to CSV
#     try:
#         save_results_to_csv(sample_name, result)
#     except Exception as e:
#         # do not fail request if saving fails; just log
#         print("Warning: failed to save result to CSV:", e)

#     # include sample_name & timestamp for UI convenience
#     result["sample_name"] = sample_name
#     result["saved_at"] = datetime.utcnow().isoformat()

#     return jsonify(result)


# # ---------------------------
# # API: retrieve saved results
# # ---------------------------
# @app.route("/saved-results", methods=["GET"])
# def saved_results():
#     """
#     Returns list of saved rows as JSON. Each row will have
#     sample_name, timestamp, avg_nm, peak_nm, min_nm, max_nm, wavelength_list, intensity_list, plot_base64
#     """
#     file_path = SAVED_RESULTS_CSV
#     if not os.path.exists(file_path):
#         return jsonify([])

#     try:
#         df = pd.read_csv(file_path)
#         # convert the JSON-string fields back to Python lists
#         rows = []
#         for _, r in df.iterrows():
#             try:
#                 wl = json.loads(r["wavelength_list"]) if not pd.isna(r["wavelength_list"]) else []
#             except Exception:
#                 wl = []
#             try:
#                 il = json.loads(r["intensity_list"]) if not pd.isna(r["intensity_list"]) else []
#             except Exception:
#                 il = []

#             rows.append({
#                 "timestamp": r.get("timestamp"),
#                 "sample_name": r.get("sample_name"),
#                 "avg_nm": r.get("avg_nm"),
#                 "peak_nm": r.get("peak_nm"),
#                 "min_nm": r.get("min_nm"),
#                 "max_nm": r.get("max_nm"),
#                 "wavelength_list": wl,
#                 "intensity_list": il,
#                 "plot_base64": r.get("plot_base64", "")
#             })
#         return jsonify(rows)
#     except Exception as e:
#         print("Error reading saved CSV:", e)
#         return jsonify([])


# # ---------------------------
# # Run server
# # ---------------------------
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5003, debug=True)


import os
import tempfile
import joblib
import cv2
import numpy as np
import pandas as pd
import json
import csv
import logging
from datetime import datetime
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
    XG_AVAILABLE = True
except:
    XG_AVAILABLE = False

# --------------------------- 
# Configure Logging
# --------------------------- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = "best_ensemble_model.joblib"
CSV_SAMPLE_PATH = "nm RGB.csv"
SAVED_RESULTS_CSV = "saved_results.csv"

# --------------------------- 
# Train or load ensemble model
# --------------------------- 
def train_and_select_ensemble(csv_path=CSV_SAMPLE_PATH, force_retrain=False):
    logger.info("=" * 60)
    logger.info("MODEL LOADING/TRAINING PROCESS STARTED")
    logger.info("=" * 60)
    
    if os.path.exists(MODEL_PATH) and not force_retrain:
        logger.info(f"Found existing model at: {MODEL_PATH}")
        logger.info("Loading saved model...")
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("✓ Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    if not os.path.exists(csv_path):
        logger.error(f"✗ CSV file not found: {csv_path}")
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    logger.info(f"Training new ensemble model from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✓ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"  Columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"✗ Failed to read CSV: {e}")
        raise
    
    X = df.drop(columns=["nm"])
    y = df["nm"].values
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    base_models = {
        "rf": RandomForestRegressor(n_estimators=250, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42),
        "svr": SVR(kernel="rbf"),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "ridge": Ridge(alpha=1.0)
    }
    
    if XG_AVAILABLE:
        base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)
        logger.info("✓ XGBoost is available and will be used")
    else:
        logger.warning("⚠ XGBoost not available, skipping")
    
    logger.info(f"Training {len(base_models)} base models...")
    scores = {}
    
    for name, model in base_models.items():
        logger.info(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            scores[name] = r2_score(y_val, pred)
            logger.info(f"    ✓ {name} R² = {scores[name]:.4f}")
        except Exception as e:
            logger.error(f"    ✗ {name} training failed: {e}")
            raise
    
    r2_vals = np.array([max(0, s) for s in scores.values()])
    weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)
    
    logger.info("Calculating weighted ensemble performance...")
    def weighted_predict(X_):
        total = 0
        for w, m in zip(weights, base_models.values()):
            total += w * m.predict(X_)
        return total
    
    weighted_r2 = r2_score(y_val, weighted_predict(X_val))
    logger.info(f"  Weighted ensemble R² = {weighted_r2:.4f}")
    
    logger.info("Training stacking ensemble...")
    estimators = [(n, m) for n, m in base_models.items()]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stack.fit(X_train, y_train)
    stack_r2 = r2_score(y_val, stack.predict(X_val))
    logger.info(f"  Stacking ensemble R² = {stack_r2:.4f}")
    
    if stack_r2 >= weighted_r2:
        model_obj = {"type": "stacking", "model": stack}
        logger.info("✓ Selected stacking model (better performance)")
    else:
        model_obj = {"type": "weighted", "models": base_models, "weights": weights}
        logger.info("✓ Selected weighted ensemble (better performance)")
    
    try:
        joblib.dump(model_obj, MODEL_PATH)
        logger.info(f"✓ Model saved to: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Failed to save model: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("MODEL LOADING/TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return model_obj

# Load model on startup
logger.info("\n" + "="*60)
logger.info("APPLICATION STARTUP")
logger.info("="*60)
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python version: {os.sys.version}")
logger.info(f"OpenCV version: {cv2.__version__}")
logger.info(f"NumPy version: {np.__version__}")

try:
    model_obj = train_and_select_ensemble()
    logger.info("✓✓✓ Model ready for predictions ✓✓✓")
except Exception as e:
    logger.critical(f"✗✗✗ CRITICAL: Model initialization failed: {e}")
    logger.exception("Full traceback:")
    
    # If it's a pickle/numpy compatibility error, try retraining
    if "BitGenerator" in str(e) or "pickle" in str(e).lower():
        logger.warning("⚠ Detected pickle compatibility issue - deleting old model and retraining...")
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
                logger.info(f"✓ Deleted incompatible model: {MODEL_PATH}")
            
            model_obj = train_and_select_ensemble(force_retrain=True)
            logger.info("✓✓✓ Model retrained successfully ✓✓✓")
        except Exception as e2:
            logger.critical(f"✗✗✗ Failed to retrain model: {e2}")
            logger.exception("Retrain exception:")
            model_obj = None
    else:
        model_obj = None

# --------------------------- 
# Ensemble predict helper
# --------------------------- 
def ensemble_predict(model_obj, X):
    if model_obj is None:
        logger.error("Prediction failed: Model is None")
        raise RuntimeError("Model is not available.")
    
    logger.debug(f"Predicting for input shape: {X.shape}")
    
    if model_obj.get("type") == "stacking":
        result = model_obj["model"].predict(X)
    else:
        preds = np.zeros(len(X))
        for w, m in zip(model_obj["weights"], model_obj["models"].values()):
            preds += w * m.predict(X)
        result = preds
    
    logger.debug(f"Prediction result: {result}")
    return result

# --------------------------- 
# ROI cropping
# --------------------------- 
def detect_and_crop_roi(frame, min_area=1000, brightness_thresh=30):
    """Detect bright region and crop to bounding box."""
    logger.debug(f"ROI detection on frame shape: {frame.shape}")
    
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        mask = v > brightness_thresh
        
        bright_pixels = mask.sum()
        logger.debug(f"  Bright pixels found: {bright_pixels}")
        
        if bright_pixels < min_area:
            logger.debug(f"  Not enough bright pixels ({bright_pixels} < {min_area}), using original frame")
            return frame
        
        ys, xs = np.where(mask)
        y1, y2 = max(0, int(ys.min()) - 5), min(frame.shape[0], int(ys.max()) + 5)
        x1, x2 = max(0, int(xs.min()) - 5), min(frame.shape[1], int(xs.max()) + 5)
        
        cropped = frame[y1:y2, x1:x2]
        logger.debug(f"  Cropped to: {cropped.shape} from original {frame.shape}")
        
        return cropped if cropped.size > 0 else frame
        
    except Exception as e:
        logger.warning(f"ROI detection failed: {e}, using original frame")
        return frame

# --------------------------- 
# Analyze video frames
# --------------------------- 
def analyze_video_frames(video_path, model_obj, target_fps=26, grid=8):
    """
    Analyze video frames and return wavelength/intensity data.
    Returns dict with results or error dict on failure.
    """
    logger.info("=" * 60)
    logger.info(f"VIDEO ANALYSIS STARTED: {video_path}")
    logger.info("=" * 60)
    
    if model_obj is None:
        logger.error("✗ Model not available for analysis")
        return {"error": "Model not loaded", "details": "Model object is None"}
    
    if not os.path.exists(video_path):
        logger.error(f"✗ Video file not found: {video_path}")
        return {"error": "Video file not found"}
    
    file_size = os.path.getsize(video_path)
    logger.info(f"Video file size: {file_size / (1024*1024):.2f} MB")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"✗ Cannot open video with OpenCV: {video_path}")
        return {"error": "Cannot open video file", "details": "cv2.VideoCapture failed"}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties:")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {total_frames/fps:.2f} seconds")
    
    step = max(1, int(round(fps / target_fps)))
    logger.info(f"Processing every {step} frame(s) to achieve ~{target_fps} fps")
    
    wavelengths = []
    intensities = []
    frame_idx = 0
    processed_frames = 0
    skipped_frames = 0
    error_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        if (frame_idx - 1) % step != 0:
            skipped_frames += 1
            continue
        
        logger.debug(f"Processing frame {frame_idx}/{total_frames}")
        
        try:
            # Crop ROI
            frame = detect_and_crop_roi(frame)
            
            h, w, _ = frame.shape
            if h == 0 or w == 0:
                logger.warning(f"⚠ Empty frame at index {frame_idx}")
                error_frames += 1
                continue
            
            patch_h = max(1, h // grid)
            patch_w = max(1, w // grid)
            
            logger.debug(f"  Grid: {grid}x{grid}, Patch size: {patch_h}x{patch_w}")
            
            patch_rgbs = []
            patch_ints = []
            
            for i in range(grid):
                for j in range(grid):
                    y1, y2 = i * patch_h, min((i + 1) * patch_h, h)
                    x1, x2 = j * patch_w, min((j + 1) * patch_w, w)
                    patch = frame[y1:y2, x1:x2]
                    
                    if patch.size == 0:
                        continue
                    
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    patch_ints.append(float(np.mean(gray)))
                    
                    small = cv2.resize(patch, (1, 1), interpolation=cv2.INTER_AREA)
                    b, g, r = [float(x) for x in small[0, 0]]
                    patch_rgbs.append([r, g, b])
            
            if len(patch_rgbs) == 0:
                logger.warning(f"⚠ No valid patches in frame {frame_idx}")
                error_frames += 1
                continue
            
            mean_rgb = np.mean(np.array(patch_rgbs), axis=0)
            frame_intensity = float(np.mean(patch_ints))
            
            logger.debug(f"  Mean RGB: {mean_rgb}, Intensity: {frame_intensity:.2f}")
            
            nm_pred = ensemble_predict(model_obj, mean_rgb.reshape(1, -1))[0]
            
            logger.debug(f"  Predicted wavelength: {nm_pred:.2f} nm")
            
            wavelengths.append(float(nm_pred))
            intensities.append(frame_intensity)
            processed_frames += 1
            
            if processed_frames % 10 == 0:
                logger.info(f"Progress: {processed_frames} frames processed ({frame_idx}/{total_frames} total)")
            
        except Exception as e:
            logger.error(f"✗ Error processing frame {frame_idx}: {e}")
            logger.exception("Frame processing exception:")
            error_frames += 1
            continue
    
    cap.release()
    
    logger.info("Video processing completed:")
    logger.info(f"  Total frames read: {frame_idx}")
    logger.info(f"  Frames processed: {processed_frames}")
    logger.info(f"  Frames skipped (sampling): {skipped_frames}")
    logger.info(f"  Frames with errors: {error_frames}")
    
    # Filter out NaNs
    arr = np.array(wavelengths)
    mask_valid = ~np.isnan(arr)
    valid_wavelengths = arr[mask_valid].tolist()
    valid_intensities = np.array(intensities)[mask_valid].tolist()
    
    nan_count = len(wavelengths) - len(valid_wavelengths)
    if nan_count > 0:
        logger.warning(f"⚠ Filtered out {nan_count} NaN wavelength values")
    
    if len(valid_wavelengths) == 0:
        logger.error("✗ No valid wavelengths extracted from video")
        return {
            "error": "No valid data extracted",
            "details": f"Processed {processed_frames} frames but got no valid wavelengths"
        }
    
    result = {
        "avg_nm": float(np.mean(valid_wavelengths)),
        "peak_nm": float(np.max(valid_wavelengths)),
        "min_nm": float(np.min(valid_wavelengths)),
        "max_nm": float(np.max(valid_wavelengths)),
        "wavelength_list": valid_wavelengths,
        "intensity_list": valid_intensities,
        "processed_frames": processed_frames,
        "total_frames": frame_idx
    }
    
    logger.info("✓ Analysis results:")
    logger.info(f"  Average wavelength: {result['avg_nm']:.2f} nm")
    logger.info(f"  Peak wavelength: {result['peak_nm']:.2f} nm")
    logger.info(f"  Min wavelength: {result['min_nm']:.2f} nm")
    logger.info(f"  Max wavelength: {result['max_nm']:.2f} nm")
    logger.info(f"  Valid data points: {len(valid_wavelengths)}")
    logger.info("=" * 60)
    
    return result

# --------------------------- 
# Plot function
# --------------------------- 
def plot_wavelength_intensity(wavelengths, intensities):
    logger.info(f"Generating plot with {len(wavelengths)} data points")
    
    try:
        plt.figure(figsize=(8, 4))
        plt.scatter(intensities, wavelengths, s=10)
        plt.xlabel("Intensity")
        plt.ylabel("Wavelength (nm)")
        plt.title("Intensity vs Wavelength")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        
        plot_data = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
        logger.info("✓ Plot generated successfully")
        return plot_data
        
    except Exception as e:
        logger.error(f"✗ Plot generation failed: {e}")
        logger.exception("Plot exception:")
        return ""

# --------------------------- 
# Save results to CSV
# --------------------------- 
def save_results_to_csv(sample_name, result):
    logger.info(f"Saving results to CSV: {SAVED_RESULTS_CSV}")
    
    file_path = SAVED_RESULTS_CSV
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "sample_name": sample_name,
        "avg_nm": result.get("avg_nm"),
        "peak_nm": result.get("peak_nm"),
        "min_nm": result.get("min_nm"),
        "max_nm": result.get("max_nm"),
        "wavelength_list": json.dumps(result.get("wavelength_list", [])),
        "intensity_list": json.dumps(result.get("intensity_list", [])),
        "plot_base64": result.get("plot_base64", "")
    }
    
    try:
        file_exists = os.path.exists(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
                logger.info(f"  Created new CSV file with headers")
            writer.writerow(row)
        
        logger.info(f"✓ Results saved for sample: {sample_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to save results: {e}")
        logger.exception("CSV save exception:")
        raise

# --------------------------- 
# API: predict
# --------------------------- 
@app.route("/predict", methods=["POST"])
def predict():
    logger.info("\n" + "="*60)
    logger.info("API REQUEST: /predict")
    logger.info("="*60)
    
    try:
        if "video" not in request.files:
            logger.warning("⚠ No video file in request")
            return jsonify({"error": "No video uploaded"}), 400
        
        sample_name = request.form.get("sample_name", "Unknown")
        file = request.files["video"]
        
        logger.info(f"Sample name: {sample_name}")
        logger.info(f"Uploaded file: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            video_path = tmp.name
        
        file_size = os.path.getsize(video_path)
        logger.info(f"Saved to temp file: {video_path} ({file_size / (1024*1024):.2f} MB)")
        
        try:
            result = analyze_video_frames(video_path, model_obj)
        finally:
            try:
                os.remove(video_path)
                logger.info(f"✓ Cleaned up temp file: {video_path}")
            except Exception as e:
                logger.warning(f"⚠ Could not delete temp file: {e}")
        
        # Check if result contains an error
        if result is None or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "analyze_video_frames returned None"
            details = result.get("details", "") if result else ""
            logger.error(f"✗ Processing failed: {error_msg}")
            if details:
                logger.error(f"  Details: {details}")
            
            return jsonify({
                "error": error_msg,
                "details": details
            }), 500
        
        # Attach plot
        logger.info("Generating visualization plot...")
        try:
            result["plot_base64"] = plot_wavelength_intensity(
                result["wavelength_list"], 
                result["intensity_list"]
            )
        except Exception as e:
            logger.error(f"✗ Plot generation failed: {e}")
            logger.exception("Plot exception:")
            result["plot_base64"] = ""
        
        # Save to CSV
        logger.info("Saving results to CSV...")
        try:
            save_results_to_csv(sample_name, result)
        except Exception as e:
            logger.error(f"⚠ Failed to save result to CSV (non-critical): {e}")
        
        result["sample_name"] = sample_name
        result["saved_at"] = datetime.utcnow().isoformat()
        
        logger.info("✓✓✓ Request completed successfully ✓✓✓")
        logger.info("="*60 + "\n")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"✗✗✗ UNEXPECTED ERROR in /predict: {e}")
        logger.exception("Full traceback:")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# --------------------------- 
# API: retrieve saved results
# --------------------------- 
@app.route("/saved-results", methods=["GET"])
def saved_results():
    logger.info("API REQUEST: /saved-results")
    
    file_path = SAVED_RESULTS_CSV
    if not os.path.exists(file_path):
        logger.info("No saved results file found, returning empty array")
        return jsonify([])
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loading {len(df)} saved results from CSV")
        
        rows = []
        
        for idx, r in df.iterrows():
            try:
                wl = json.loads(r["wavelength_list"]) if not pd.isna(r["wavelength_list"]) else []
            except Exception as e:
                logger.warning(f"Failed to parse wavelength_list for row {idx}: {e}")
                wl = []
            
            try:
                il = json.loads(r["intensity_list"]) if not pd.isna(r["intensity_list"]) else []
            except Exception as e:
                logger.warning(f"Failed to parse intensity_list for row {idx}: {e}")
                il = []
            
            rows.append({
                "timestamp": r.get("timestamp"),
                "sample_name": r.get("sample_name"),
                "avg_nm": r.get("avg_nm"),
                "peak_nm": r.get("peak_nm"),
                "min_nm": r.get("min_nm"),
                "max_nm": r.get("max_nm"),
                "wavelength_list": wl,
                "intensity_list": il,
                "plot_base64": r.get("plot_base64", "")
            })
        
        logger.info(f"✓ Returning {len(rows)} saved results")
        return jsonify(rows)
    
    except Exception as e:
        logger.error(f"✗ Error reading saved CSV: {e}")
        logger.exception("CSV read exception:")
        return jsonify([])

# --------------------------- 
# API: health check
# --------------------------- 
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint with system status"""
    status = {
        "status": "healthy" if model_obj is not None else "unhealthy",
        "model_loaded": model_obj is not None,
        "csv_exists": os.path.exists(CSV_SAMPLE_PATH),
        "model_file_exists": os.path.exists(MODEL_PATH),
        "saved_results_count": 0
    }
    
    if os.path.exists(SAVED_RESULTS_CSV):
        try:
            df = pd.read_csv(SAVED_RESULTS_CSV)
            status["saved_results_count"] = len(df)
        except:
            pass
    
    logger.info(f"Health check: {status}")
    return jsonify(status)

# --------------------------- 
# Run server
# --------------------------- 
if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("FLASK SERVER STARTING")
    logger.info("="*60)
    logger.info(f"Host: 127.0.0.1")
    logger.info(f"Port: 5003")
    logger.info(f"Model loaded: {model_obj is not None}")
    logger.info(f"CSV path: {CSV_SAMPLE_PATH} (exists: {os.path.exists(CSV_SAMPLE_PATH)})")
    logger.info(f"Model path: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
    logger.info(f"Logs: app.log")
    logger.info("="*60)
    logger.info("Endpoints:")
    logger.info("  POST   /predict        - Process video")
    logger.info("  GET    /saved-results  - Retrieve saved results")
    logger.info("  GET    /health         - Health check")
    logger.info("="*60 + "\n")
    
    app.run(host="127.0.0.1", port=5003, debug=True)