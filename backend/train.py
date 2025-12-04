# import os
# import tempfile
# import joblib
# import cv2
# import numpy as np
# import pandas as pd
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


# # ---------------------------
# # TRAIN / LOAD MODEL
# # ---------------------------
# def train_and_select_ensemble(csv_path=CSV_SAMPLE_PATH, force_retrain=False):

#     if os.path.exists(MODEL_PATH) and not force_retrain:
#         print("Loading saved model...")
#         return joblib.load(MODEL_PATH)
    
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSV not found: {csv_path}")

#     print("Training ensemble model...")
#     df = pd.read_csv(csv_path)
#     X = df.drop(columns=['nm'])
#     y = df['nm'].values

#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     base_models = {
#         "rf": RandomForestRegressor(n_estimators=250, random_state=42),
#         "gbr": GradientBoostingRegressor(random_state=42),
#         "svr": SVR(kernel="rbf"),
#         "knn": KNeighborsRegressor(n_neighbors=5),
#         "ridge": Ridge(alpha=1.0)
#     }

#     if XG_AVAILABLE:
#         base_models["xgb"] = XGBRegressor(
#             n_estimators=200, 
#             learning_rate=0.08, 
#             random_state=42
#         )

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
#         print("Using stacking model")
#     else:
#         model_obj = {"type": "weighted", "models": base_models, "weights": weights}
#         print("Using weighted ensemble")

#     joblib.dump(model_obj, MODEL_PATH)
#     return model_obj


# # Load model
# model_obj = train_and_select_ensemble()



# # ---------------------------
# # ENSEMBLE PREDICT
# # ---------------------------
# def ensemble_predict(model_obj, X):
#     if model_obj["type"] == "stacking":
#         return model_obj["model"].predict(X)
#     else:
#         preds = np.zeros(len(X))
#         for w, m in zip(model_obj["weights"], model_obj["models"].values()):
#             preds += w * m.predict(X)
#         return preds


# # ---------------------------
# # ROI CROPPING
# # ---------------------------
# def detect_and_crop_roi(frame, min_area=1000, brightness_thresh=30):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     v = hsv[:, :, 2]
#     mask = v > brightness_thresh

#     if mask.sum() < min_area:
#         return frame

#     ys, xs = np.where(mask)
#     y1, y2 = max(0, ys.min()-5), min(frame.shape[0], ys.max()+5)
#     x1, x2 = max(0, xs.min()-5), min(frame.shape[1], xs.max()+5)

#     cropped = frame[y1:y2, x1:x2]
#     return cropped if cropped.size > 0 else frame



# # ---------------------------
# # VIDEO ANALYSIS
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

#         frame = detect_and_crop_roi(frame)

#         h, w, _ = frame.shape
#         patch_h = h // grid
#         patch_w = w // grid

#         patch_rgbs = []
#         patch_ints = []

#         for i in range(grid):
#             for j in range(grid):
#                 y1, y2 = i * patch_h, (i+1) * patch_h
#                 x1, x2 = j * patch_w, (j+1) * patch_w

#                 patch = frame[y1:y2, x1:x2]
#                 if patch.size == 0:
#                     continue

#                 gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
#                 patch_ints.append(float(np.mean(gray)))

#                 b, g, r = cv2.resize(patch, (1,1))[0,0]
#                 patch_rgbs.append([r, g, b])

#         if len(patch_rgbs) == 0:
#             continue

#         mean_rgb = np.mean(np.array(patch_rgbs), axis=0)
#         intensity = float(np.mean(patch_ints))

#         nm_pred = ensemble_predict(model_obj, mean_rgb.reshape(1, -1))[0]

#         wavelengths.append(float(nm_pred))
#         intensities.append(intensity)

#     cap.release()

#     if len(wavelengths) == 0:
#         return None

#     return {
#         "avg_nm": float(np.mean(wavelengths)),
#         "peak_nm": float(np.max(wavelengths)),
#         "min_nm": float(np.min(wavelengths)),
#         "max_nm": float(np.max(wavelengths)),
#         "wavelength_list": wavelengths,
#         "intensity_list": intensities
#     }



# # ---------------------------
# # PLOT
# # ---------------------------
# def plot_wavelength_intensity(wavelengths, intensities):
#     plt.figure(figsize=(8,4))
#     plt.plot(wavelengths, intensities)
#     plt.xlabel("Wavelength (nm)")
#     plt.ylabel("Intensity")
#     plt.title("Wavelength vs Intensity")
#     plt.tight_layout()

#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     plt.close()
#     buf.seek(0)
#     return "data:image/png;base64," + base64.b64encode(buf.read()).decode()



# @app.route("/predict", methods=["POST"])
# def predict():
#     if "video" not in request.files:
#         return jsonify({"error": "No video uploaded"}), 400

#     file = request.files["video"]

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         file.save(tmp.name)
#         video_path = tmp.name

#     result = analyze_video_frames(video_path, model_obj)
#     os.remove(video_path)

#     if result is None:
#         return jsonify({"error": "Failed to process video"}), 500

#     result["plot_base64"] = plot_wavelength_intensity(
#         result["wavelength_list"], result["intensity_list"]
#     )

#     return jsonify(result)



# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5003, debug=True)
# train.py
import os
import tempfile
import joblib
import cv2
import numpy as np
import pandas as pd
import json
import csv
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

app = Flask(__name__)
CORS(app)

MODEL_PATH = "best_ensemble_model.joblib"
CSV_SAMPLE_PATH = "nm RGB.csv"
SAVED_RESULTS_CSV = "saved_results.csv"


# ---------------------------
# Train or load ensemble model
# ---------------------------
def train_and_select_ensemble(csv_path=CSV_SAMPLE_PATH, force_retrain=False):
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print("Loading saved model...")
        return joblib.load(MODEL_PATH)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print("Training ensemble model...")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["nm"])
    y = df["nm"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models = {
        "rf": RandomForestRegressor(n_estimators=250, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42),
        "svr": SVR(kernel="rbf"),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "ridge": Ridge(alpha=1.0)
    }

    if XG_AVAILABLE:
        base_models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)

    scores = {}
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        scores[name] = r2_score(y_val, pred)
        print(f"{name} R2 = {scores[name]:.4f}")

    r2_vals = np.array([max(0, s) for s in scores.values()])
    weights = r2_vals / (r2_vals.sum() if r2_vals.sum() else 1)

    def weighted_predict(X_):
        total = 0
        for w, m in zip(weights, base_models.values()):
            total += w * m.predict(X_)
        return total

    weighted_r2 = r2_score(y_val, weighted_predict(X_val))

    estimators = [(n, m) for n, m in base_models.items()]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stack.fit(X_train, y_train)
    stack_r2 = r2_score(y_val, stack.predict(X_val))

    if stack_r2 >= weighted_r2:
        model_obj = {"type": "stacking", "model": stack}
        print("Using stacking model.")
    else:
        model_obj = {"type": "weighted", "models": base_models, "weights": weights}
        print("Using weighted ensemble.")

    joblib.dump(model_obj, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model_obj


# Load model on startup (will train if MODEL_PATH missing and CSV present)
try:
    model_obj = train_and_select_ensemble()
except Exception as e:
    print("Model load/train failed:", e)
    model_obj = None


# ---------------------------
# Ensemble predict helper
# ---------------------------
def ensemble_predict(model_obj, X):
    if model_obj is None:
        raise RuntimeError("Model is not available.")
    if model_obj.get("type") == "stacking":
        return model_obj["model"].predict(X)
    else:
        preds = np.zeros(len(X))
        for w, m in zip(model_obj["weights"], model_obj["models"].values()):
            preds += w * m.predict(X)
        return preds


# ---------------------------
# ROI cropping
# ---------------------------
def detect_and_crop_roi(frame, min_area=1000, brightness_thresh=30):
    """
    Detect bright region by Value channel and crop to bounding box.
    Returns the cropped frame or original if no sufficient bright area found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = v > brightness_thresh

    if mask.sum() < min_area:
        return frame

    ys, xs = np.where(mask)
    y1, y2 = max(0, int(ys.min()) - 5), min(frame.shape[0], int(ys.max()) + 5)
    x1, x2 = max(0, int(xs.min()) - 5), min(frame.shape[1], int(xs.max()) + 5)

    cropped = frame[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else frame


# ---------------------------
# Analyze video frames (no weighting)
# ---------------------------
def analyze_video_frames(video_path, model_obj, target_fps=26, grid=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / target_fps)))

    wavelengths = []
    intensities = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % step != 0:
            continue

        # crop ROI to remove black background
        try:
            frame = detect_and_crop_roi(frame)
        except Exception:
            # if ROI fails, continue with original frame
            pass

        h, w, _ = frame.shape
        patch_h = max(1, h // grid)
        patch_w = max(1, w // grid)

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
            continue

        mean_rgb = np.mean(np.array(patch_rgbs), axis=0)
        frame_intensity = float(np.mean(patch_ints))

        try:
            nm_pred = ensemble_predict(model_obj, mean_rgb.reshape(1, -1))[0]
        except Exception:
            nm_pred = float("nan")

        wavelengths.append(float(nm_pred))
        intensities.append(frame_intensity)

    cap.release()

    # filter out NaNs
    arr = np.array(wavelengths)
    mask_valid = ~np.isnan(arr)
    valid_wavelengths = arr[mask_valid].tolist()
    valid_intensities = np.array(intensities)[mask_valid].tolist()

    if len(valid_wavelengths) == 0:
        return None

    return {
        "avg_nm": float(np.mean(valid_wavelengths)),
        "peak_nm": float(np.max(valid_wavelengths)),
        "min_nm": float(np.min(valid_wavelengths)),
        "max_nm": float(np.max(valid_wavelengths)),
        "wavelength_list": valid_wavelengths,
        "intensity_list": valid_intensities
    }


# ---------------------------
# Plot function (Intensity on X, Wavelength on Y)
# ---------------------------
def plot_wavelength_intensity(wavelengths, intensities):
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
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ---------------------------
# Save results to CSV
# ---------------------------
def save_results_to_csv(sample_name, result):
    """
    Appends a row to SAVED_RESULTS_CSV containing:
    sample_name, timestamp, avg_nm, peak_nm, min_nm, max_nm,
    wavelength_list (json string), intensity_list (json string), plot_base64
    """
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

    file_exists = os.path.exists(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------
# API: predict (POST video)
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    sample_name = request.form.get("sample_name", "Unknown")

    file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        file.save(tmp.name)
        video_path = tmp.name

    try:
        result = analyze_video_frames(video_path, model_obj)
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass

    if result is None:
        return jsonify({"error": "Failed to process video"}), 500

    # attach plot
    result["plot_base64"] = plot_wavelength_intensity(result["wavelength_list"], result["intensity_list"])

    # save to CSV
    try:
        save_results_to_csv(sample_name, result)
    except Exception as e:
        # do not fail request if saving fails; just log
        print("Warning: failed to save result to CSV:", e)

    # include sample_name & timestamp for UI convenience
    result["sample_name"] = sample_name
    result["saved_at"] = datetime.utcnow().isoformat()

    return jsonify(result)


# ---------------------------
# API: retrieve saved results
# ---------------------------
@app.route("/saved-results", methods=["GET"])
def saved_results():
    """
    Returns list of saved rows as JSON. Each row will have
    sample_name, timestamp, avg_nm, peak_nm, min_nm, max_nm, wavelength_list, intensity_list, plot_base64
    """
    file_path = SAVED_RESULTS_CSV
    if not os.path.exists(file_path):
        return jsonify([])

    try:
        df = pd.read_csv(file_path)
        # convert the JSON-string fields back to Python lists
        rows = []
        for _, r in df.iterrows():
            try:
                wl = json.loads(r["wavelength_list"]) if not pd.isna(r["wavelength_list"]) else []
            except Exception:
                wl = []
            try:
                il = json.loads(r["intensity_list"]) if not pd.isna(r["intensity_list"]) else []
            except Exception:
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
        return jsonify(rows)
    except Exception as e:
        print("Error reading saved CSV:", e)
        return jsonify([])


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=True)
