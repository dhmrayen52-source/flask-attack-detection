import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Charger le dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Supprimer la colonne 'id' inutile pour l'entraînement
df.drop(columns=['id'], inplace=True)

# Encoder les colonnes catégorielles
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Séparer les features et la cible
X = df.drop(columns=['label'])
y = df['label']

# Division du dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nMatrice de confusion:\n", cm)
print("\nRapport de classification:\n", report)

# --------------------------
# Prédiction d’un nouvel échantillon
# --------------------------
nouveau = {
    'dur': [0.05],
    'proto': [6],
    'service': [2],
    'state': [4],
    'spkts': [20],
    'dpkts': [25],
    'sbytes': [1500],
    'dbytes': [2300],
    'rate': [0.2],
    'sttl': [255],
    'dttl': [64],
    'sload': [0.1],
    'dload': [0.15],
    'sloss': [0],
    'dloss': [0],
    'sinpkt': [0.002],
    'dinpkt': [0.003],
    'sjit': [0.1],
    'djit': [0.1],
    'swin': [2000],
    'stcpb': [12345],
    'dtcpb': [54321],
    'dwin': [2000],
    'tcprtt': [0.05],
    'synack': [0.02],
    'ackdat': [0.03],
    'smean': [300],
    'dmean': [400],
    'trans_depth': [1],
    'response_body_len': [0],
    'ct_srv_src': [2],
    'ct_state_ttl': [3],
    'ct_dst_ltm': [1],
    'ct_src_dport_ltm': [2],
    'ct_dst_sport_ltm': [1],
    'ct_dst_src_ltm': [2],
    'is_ftp_login': [0],
    'ct_ftp_cmd': [0],
    'ct_flw_http_mthd': [1],
    'ct_src_ltm': [1],
    'ct_srv_dst': [2],
    'is_sm_ips_ports': [0],
    'attack_cat': [1]
}

nouvelle_df = pd.DataFrame(nouveau)

# Normaliser avec le même scaler
nouvelle_df_scaled = scaler.transform(nouvelle_df)

# Prédiction
prediction = model.predict(nouvelle_df_scaled)
print("Résultat :", "Attack" if prediction[0] == 1 else "Benign")

import joblib

# Sauvegarde du modèle et du scaler
joblib.dump(model, "modele_logreg.pkl")
joblib.dump(scaler, "scaler.pkl")