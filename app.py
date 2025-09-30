
from flask import Flask, request, jsonify, send_from_directory
import joblib, os, numpy as np, pandas as pd
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

model_data = joblib.load("model.joblib")
model = model_data["model"]
ohe = model_data["ohe"]

CLASS_MAP = {0:"HomeWin", 1:"Draw", 2:"AwayWin"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ht, at = data.get("HomeTeam"), data.get("AwayTeam")
    hf, af = float(data.get("home_form",1.0)), float(data.get("away_form",1.0))
    X_teams = pd.DataFrame([[ht, at]], columns=["HomeTeam","AwayTeam"])
    teams_enc = ohe.transform(X_teams)
    from scipy.sparse import hstack
    X = hstack([teams_enc, np.array([[hf, af]])])
    probs = model.predict_proba(X)[0]
    pred_class = int(model.predict(X)[0])
    return jsonify({
        "prediction": CLASS_MAP[pred_class],
        "probabilities": {
            "home_win": float(probs[0]),
            "draw": float(probs[1]),
            "away_win": float(probs[2])
        }
    })

@app.route("/", defaults={"path":""})
@app.route("/<path:path>")
def static_files(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
