from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# ------------------------------
# Charger modèle et scaler
# ------------------------------
model = joblib.load("modele_logreg.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# Page HTML avec tous les features
# ------------------------------
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Détection d'attaques - Logistic Regression</title>
</head>
<body style="font-family: Arial; margin: 40px;">
    <h2>Détection d'attaques réseau (UNSW-NB15)</h2>
    <form method="post" action="/predict">
        {% for feature in features %}
            <label>{{ feature }}:</label> 
            <input type="number" step="any" name="{{ feature }}" required><br><br>
        {% endfor %}
        <button type="submit">Prédire</button>
    </form>

    {% if prediction %}
        <h3>Résultat : {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

# ------------------------------
# Liste complète des features
# ------------------------------
features = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate','sttl','dttl',
    'sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit','swin','stcpb','dtcpb',
    'dwin','tcprtt','synack','ackdat','smean','dmean','trans_depth','response_body_len',
    'ct_srv_src','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
    'ct_dst_src_ltm','is_ftp_login','ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm',
    'ct_srv_dst','is_sm_ips_ports','attack_cat'
]

# ------------------------------
# Route principale
# ------------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_form, features=features)

# ------------------------------
# Route de prédiction
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lire les données du formulaire
        data = {feature: float(request.form[feature]) for feature in features}

        # Transformer en DataFrame
        df = pd.DataFrame([data])

        # Normaliser
        df_scaled = scaler.transform(df)

        # Prédire
        prediction = model.predict(df_scaled)
        result = "Attack" if prediction[0] == 1 else "Benign"

        return render_template_string(html_form, features=features, prediction=result)
    except Exception as e:
        return jsonify({'error': str(e)})

# ------------------------------
# Lancer l'application
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
