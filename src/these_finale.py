# -*- coding: utf-8 -*-
"""
Systèmes de défense adaptatifs contre les cyberattaques évolutives
Framework IDS : Transfer Learning + Meta-Learning (MAML) + Few-Shot Learning

Auteure    : Joyce Maeva Lengue Mbangue
Encadrante : Dr Roumaissa Bekkouche
Programme  : MSc Manager de la Cybersécurité — ECE Paris, 2026

Exécution :
  - Google Colab : le Drive est monté automatiquement.
  - En local     : définir la variable d'environnement THESE_BASE
                   (ex. export THESE_BASE=./data) avant de lancer le script.
"""
# ════════════════════════════════════════════════════════════════
# NOTEBOOK : Systèmes de Défense Adaptatifs contre les Cyberattaques
# Auteur   : Joyce Maeva Lengue Mbangue
# Encadrant: Dr Bekkouche Roumaissa
# Date     : 2026
#
# STRUCTURE :
#   SECTION 0  — Configuration & Imports
#   SECTION 1  — Chargement des 3 datasets (sans ToN-IoT)
#   SECTION 2  — Prétraitement & Fusion
#   SECTION 3  — Normalisation, PCA, SMOTE
#   SECTION 4  — Backbone : Architecture & Entraînement (Phase 1)
#   SECTION 5  — Évaluation Backbone (F1, Confusion, t-SNE)
#   SECTION 6  — Gel du Backbone
#   SECTION 7  — MAML : Meta-Learning (Phase 2)
#   SECTION 8  — Few-Shot Deployment (Phase 3)
#   SECTION 9  — Test Cross-Domaine IoT (ToN-IoT — domaine inconnu)
#   SECTION 10 — Graphes & Visualisations complètes
#   SECTION 11 — Sauvegarde des résultats finaux
# ════════════════════════════════════════════════════════════════

# ── Détection de l'environnement (Colab ou local) ─────────────────
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DEFAULT_BASE = "/content/drive/MyDrive/these_cybersec"
except ImportError:
    DEFAULT_BASE = "./data"

import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix)
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Chemin de base (configurable via THESE_BASE)
BASE   = os.environ.get("THESE_BASE", DEFAULT_BASE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Créer les dossiers si absents
for d in [f"{BASE}/datasets/nslkdd", f"{BASE}/datasets/cicids2017",
          f"{BASE}/datasets/unsw_nb15", f"{BASE}/datasets/toniot",
          f"{BASE}/models"]:
    os.makedirs(d, exist_ok=True)

print(f"✅ Imports OK")
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ Device : {device}")
if device.type == "cuda":
    print(f"✅ GPU : {torch.cuda.get_device_name(0)}")

# ════════════════════════════════════════════════════════════════
# SECTION 1 — Chargement des 3 Datasets (Backbone général)
#
# IMPORTANT : ToN-IoT N'EST PAS chargé ici.
# Il sera chargé UNIQUEMENT en Section 9 pour le test
# cross-domaine — domaine jamais vu pendant l'entraînement.
# C'est le principe de l'Option A (backbone général sans IoT).
# ════════════════════════════════════════════════════════════════

# ── 1.1 NSL-KDD ─────────────────────────────────────────────────
col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty"
]
df_train = pd.read_csv(
    f"{BASE}/datasets/nslkdd/KDDTrain+.txt", names=col_names)
df_train.drop("difficulty", axis=1, inplace=True)
df_train["label"] = df_train["label"].str.replace('.', '', regex=False)

label_mapping_nsl = {
    'normal':'Normal',
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS',
    'smurf':'DoS','teardrop':'DoS','mailbomb':'DoS',
    'apache2':'DoS','processtable':'DoS','udpstorm':'DoS',
    'ipsweep':'Reconnaissance','nmap':'Reconnaissance',
    'portsweep':'Reconnaissance','satan':'Reconnaissance',
    'mscan':'Reconnaissance','saint':'Reconnaissance',
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L',
    'multihop':'R2L','phf':'R2L','spy':'R2L',
    'warezclient':'R2L','warezmaster':'R2L','sendmail':'R2L',
    'named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'worm':'R2L','xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R',
    'rootkit':'U2R','ps':'U2R','sqlattack':'U2R','xterm':'U2R',
}
df_train["super_label"] = df_train["label"].map(label_mapping_nsl)
df_train.dropna(subset=["super_label"], inplace=True)

le_proto = LabelEncoder()
for col in ["protocol_type", "service", "flag"]:
    df_train[col] = le_proto.fit_transform(df_train[col].astype(str))

print(f"✅ NSL-KDD : {df_train.shape}")
print(df_train["super_label"].value_counts())

# ── 1.2 CICIDS2017 ──────────────────────────────────────────────
df_cicids = pd.read_csv(
    f"{BASE}/datasets/cicids2017/cicids2017_cleaned.csv")
df_cicids.columns = df_cicids.columns.str.strip()
df_cicids.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cicids.dropna(inplace=True)
df_cicids.drop_duplicates(inplace=True)

label_mapping_cicids = {
    'Normal Traffic': 'Normal',
    'Port Scanning':  'Reconnaissance',
    'Web Attacks':    'Injection',
}
df_cicids["super_label"] = df_cicids["Attack Type"].map(label_mapping_cicids)
df_cicids.dropna(subset=["super_label"], inplace=True)

print(f"✅ CICIDS2017 : {df_cicids.shape}")
print(df_cicids["super_label"].value_counts())

# ── 1.3 UNSW-NB15 ───────────────────────────────────────────────
df_unsw = pd.concat([
    pd.read_parquet(
        f"{BASE}/datasets/unsw_nb15/UNSW_NB15_training-set.parquet"),
    pd.read_parquet(
        f"{BASE}/datasets/unsw_nb15/UNSW_NB15_testing-set.parquet")
], ignore_index=True)
df_unsw.replace([np.inf, -np.inf], np.nan, inplace=True)
df_unsw.dropna(inplace=True)

label_mapping_unsw = {
    'Normal':'Normal','normal':'Normal',
    'Generic':'DoS','DoS':'DoS','Fuzzers':'DoS',
    'Exploits':'Exploit','Shellcode':'Exploit',
    'Reconnaissance':'Reconnaissance','Analysis':'Reconnaissance',
    'Backdoor':'Backdoor','Worms':'Botnet',
}
df_unsw["super_label"] = (df_unsw["attack_cat"]
                          .astype(str).str.strip()
                          .map(label_mapping_unsw))
df_unsw.dropna(subset=["super_label"], inplace=True)

print(f"✅ UNSW-NB15 : {df_unsw.shape}")
print(df_unsw["super_label"].value_counts())

# ════════════════════════════════════════════════════════════════
# SECTION 2 — Prétraitement & Fusion des 3 Datasets
#
# Objectif : harmoniser les features et fusionner en un seul
# tableau avec 12 super-classes comportementales communes.
# Stratégie : UNION des features (pas intersection).
# Features absentes d'un dataset → remplissage par 0.
# ════════════════════════════════════════════════════════════════

# ── 2.1 Renommage des colonnes vers noms standards ───────────────
rename_nsl = {
    'duration':'flow_duration','src_bytes':'src_bytes',
    'dst_bytes':'dst_bytes','protocol_type':'protocol',
    'flag':'tcp_flag','count':'conn_count',
    'srv_count':'srv_count','serror_rate':'serror_rate',
    'rerror_rate':'rerror_rate','same_srv_rate':'same_srv_rate',
    'diff_srv_rate':'diff_srv_rate','land':'land',
    'wrong_fragment':'wrong_fragment','urgent':'urgent',
}
rename_cicids = {
    'Flow Duration':'flow_duration',
    'Total Fwd Packets':'fwd_packets',
    'Total Backward Packets':'bwd_packets',
    'Total Length of Fwd Packets':'src_bytes',
    'Total Length of Bwd Packets':'dst_bytes',
    'Flow Bytes/s':'flow_bytes_per_sec',
    'Flow Packets/s':'flow_pkts_per_sec',
    'Fwd Header Length':'fwd_header_len',
    'Bwd Header Length':'bwd_header_len',
    'Destination Port':'dst_port','Protocol':'protocol',
    'Fwd Packet Length Max':'fwd_pkt_len_max',
    'Fwd Packet Length Min':'fwd_pkt_len_min',
    'Fwd Packet Length Mean':'fwd_pkt_len_mean',
    'Fwd Packet Length Std':'fwd_pkt_len_std',
    'Bwd Packet Length Max':'bwd_pkt_len_max',
    'Bwd Packet Length Min':'bwd_pkt_len_min',
    'Bwd Packet Length Mean':'bwd_pkt_len_mean',
    'Bwd Packet Length Std':'bwd_pkt_len_std',
    'Flow IAT Mean':'flow_iat_mean','Flow IAT Std':'flow_iat_std',
    'Flow IAT Max':'flow_iat_max','Flow IAT Min':'flow_iat_min',
    'Fwd IAT Total':'fwd_iat_total','Fwd IAT Mean':'fwd_iat_mean',
    'Bwd IAT Total':'bwd_iat_total','Bwd IAT Mean':'bwd_iat_mean',
    'Packet Length Mean':'pkt_len_mean',
    'Packet Length Std':'pkt_len_std',
    'Packet Length Variance':'pkt_len_var',
    'Average Packet Size':'avg_pkt_size',
    'Idle Mean':'idle_mean','Idle Max':'idle_max','Idle Min':'idle_min',
}
rename_unsw = {
    'dur':'flow_duration','sbytes':'src_bytes','dbytes':'dst_bytes',
    'spkts':'fwd_packets','dpkts':'bwd_packets',
    'rate':'flow_bytes_per_sec','sload':'src_load','dload':'dst_load',
    'proto':'protocol','state':'tcp_flag',
    'sjit':'fwd_iat_mean','djit':'bwd_iat_mean',
    'smean':'fwd_pkt_len_mean','dmean':'bwd_pkt_len_mean',
    'sinpkt':'fwd_iat_total','dinpkt':'bwd_iat_total',
}

df_nsl_r    = df_train.rename(columns=rename_nsl)
df_cicids_r = df_cicids.rename(columns=rename_cicids)
df_unsw_r   = df_unsw.rename(columns=rename_unsw)

# ── 2.2 Union des features numériques ────────────────────────────
cols_exclure = {
    'label','super_label','domain_label','source_dataset',
    'Attack Type','attack_cat','difficulty','id','Label'
}

def get_num_features(df):
    return set(df.select_dtypes(
        include=[np.number]).columns) - cols_exclure

f_nsl    = get_num_features(df_nsl_r)
f_cicids = get_num_features(df_cicids_r)
f_unsw   = get_num_features(df_unsw_r)

toutes_features = sorted(f_nsl | f_cicids | f_unsw)
cols_finales    = toutes_features + [
    "super_label", "domain_label", "source_dataset"]

print(f"✅ Features NSL-KDD    : {len(f_nsl)}")
print(f"✅ Features CICIDS2017 : {len(f_cicids)}")
print(f"✅ Features UNSW-NB15  : {len(f_unsw)}")
print(f"✅ Total features union : {len(toutes_features)}")

# ── 2.3 Ajout des labels et domaines ─────────────────────────────
df_nsl_r["super_label"]    = df_train["super_label"].values
df_nsl_r["domain_label"]   = "general"
df_nsl_r["source_dataset"] = "nslkdd"

df_cicids_r["super_label"]    = df_cicids["super_label"].values
df_cicids_r["domain_label"]   = "general"
df_cicids_r["source_dataset"] = "cicids2017"

df_unsw_r["super_label"]    = df_unsw["super_label"].values
df_unsw_r["domain_label"]   = "general"
df_unsw_r["source_dataset"] = "unsw_nb15"

# ── 2.4 Fusion avec remplissage par 0 ────────────────────────────
def preparer(df):
    """
    Réindexe le df sur les colonnes finales.
    Convertit les colonnes catégorielles en numérique.
    Remplit les valeurs manquantes par 0.
    """
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col], 'cat'):
            df[col] = df[col].astype(object)
    df2 = df.reindex(columns=cols_finales)
    for col in toutes_features:
        df2[col] = pd.to_numeric(df2[col], errors='coerce').fillna(0)
    for col in ["super_label", "domain_label", "source_dataset"]:
        if col in df.columns:
            df2[col] = df[col].values
    return df2

df_fusion = pd.concat([
    preparer(df_nsl_r),
    preparer(df_cicids_r),
    preparer(df_unsw_r),
], ignore_index=True)

df_fusion.replace([np.inf, -np.inf], np.nan, inplace=True)
df_fusion.dropna(subset=["super_label"], inplace=True)

print(f"✅ Fusion 3 datasets : {df_fusion.shape}")
print(f"\nPar source :")
print(df_fusion["source_dataset"].value_counts())
print(f"\nPar super-classe :")
print(df_fusion["super_label"].value_counts())

# ════════════════════════════════════════════════════════════════
# SECTION 3 — Normalisation, Réduction Dimensionnelle (PCA), SMOTE
#
# Entrées  : df_fusion (N lignes × 111 features)
# Sorties  : X_tr_pca, X_vl_pca, X_ts_pca (64 dimensions)
#            y_train_smote (après rééquilibrage)
#            tr_loader, vl_loader, ts_loader (DataLoaders PyTorch)
# ════════════════════════════════════════════════════════════════

X     = df_fusion[toutes_features].values.astype(np.float32)
y     = df_fusion["super_label"].values
le    = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"Classes encodées : {dict(zip(le.classes_, range(len(le.classes_))))}")
print(f"Nombre de classes : {len(le.classes_)}")

# ── 3.1 Split 70/15/15 ───────────────────────────────────────────
X_tr_r, X_tmp, y_tr_r, y_tmp = train_test_split(
    X, y_enc, test_size=0.30, random_state=42, stratify=y_enc)
X_vl_r, X_ts_r, y_vl_r, y_ts_r = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"\nTrain : {X_tr_r.shape[0]:,}")
print(f"Val   : {X_vl_r.shape[0]:,}")
print(f"Test  : {X_ts_r.shape[0]:,}")

# ── 3.2 Normalisation ────────────────────────────────────────────
# RÈGLE : fit UNIQUEMENT sur le train pour éviter le data leakage
scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr_r)
X_vl_sc  = scaler.transform(X_vl_r)
X_ts_sc  = scaler.transform(X_ts_r)

print(f"\n✅ Normalisation OK")
print(f"Moyenne train (≈0) : {X_tr_sc.mean():.6f}")
print(f"Std train    (≈1)  : {X_tr_sc.std():.6f}")

# ── 3.3 PCA — 111 → 64 dimensions ───────────────────────────────
pca      = PCA(n_components=64, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_sc)
X_vl_pca = pca.transform(X_vl_sc)
X_ts_pca = pca.transform(X_ts_sc)

variance   = pca.explained_variance_ratio_.sum()
input_dim  = X_tr_pca.shape[1]
n_classes  = len(le.classes_)

print(f"\n✅ PCA : {input_dim} dimensions")
print(f"Variance expliquée : {variance:.2%}")

# ── 3.4 SMOTE — Rééquilibrage des classes minoritaires ───────────
# Classes portées à 2000 exemples minimum
comptage     = Counter(y_tr_r)
cible_smote  = {k: max(v, 2000) for k, v in comptage.items()}
smote        = SMOTE(sampling_strategy=cible_smote,
                     random_state=42, k_neighbors=3)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_pca, y_tr_r)

print(f"\n✅ SMOTE : {X_tr_sm.shape[0]:,} exemples train")
print("Distribution après SMOTE :")
for cls, cnt in sorted(Counter(y_tr_sm).items()):
    print(f"  {le.classes_[cls]:<15} : {cnt:,}")

# ── 3.5 Tenseurs & DataLoaders ───────────────────────────────────
X_tr_t = torch.FloatTensor(X_tr_sm).to(device)
y_tr_t  = torch.LongTensor(y_tr_sm).to(device)
X_vl_t  = torch.FloatTensor(X_vl_pca).to(device)
y_vl_t  = torch.LongTensor(y_vl_r).to(device)
X_ts_t  = torch.FloatTensor(X_ts_pca).to(device)
y_ts_t  = torch.LongTensor(y_ts_r).to(device)

tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                       batch_size=512, shuffle=True)
vl_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t),
                       batch_size=512, shuffle=False)
ts_loader = DataLoader(TensorDataset(X_ts_t, y_ts_t),
                       batch_size=512, shuffle=False)

print(f"\n✅ DataLoaders prêts")
print(f"Train : {X_tr_t.shape[0]:,} | Val : {X_vl_t.shape[0]:,} | Test : {X_ts_t.shape[0]:,}")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — Backbone 1D-CNN + BiLSTM (Phase 1 — Transfer Learning)
#
# Architecture :
#   Entrée  : vecteur 64 dim (après PCA)
#   1D-CNN  : détecte les patterns locaux entre features
#   BiLSTM  : capture les dépendances temporelles bidirectionnelles
#   Dense   : produit l'embedding de 128 dim
#   Sortie  : embedding 128 dim + logits n_classes
#
# L'embedding est la représentation latente utilisée en Phase 3
# par les Prototypical Networks pour calculer les prototypes.
# ════════════════════════════════════════════════════════════════

class Backbone(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, n_classes=9):
        super().__init__()

        # 1D-CNN — patterns locaux entre features voisines
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(input_dim)
        )
        # BiLSTM — dépendances entre features distantes
        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=64, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        # Dense — embedding final 128 dim
        self.embedding_layer = nn.Sequential(
            nn.Linear(128, embedding_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        # Tête de classification (Phase 1 uniquement)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, 64)
        x = self.cnn(x)             # (B, 128, 64)
        x = x.permute(0, 2, 1)     # (B, 64, 128)
        x, _ = self.bilstm(x)      # (B, 64, 128)
        x = x[:, -1, :]            # (B, 128)
        emb = self.embedding_layer(x)   # (B, 128)
        logits = self.classifier(emb)   # (B, n_classes)
        return emb, logits

print(f"✅ Classe Backbone définie")
print(f"   Entrée  : {input_dim} dim (PCA)")
print(f"   Embedding : 128 dim")
print(f"   Classes : {n_classes}")

# ── 4.2 Entraînement du Backbone ─────────────────────────────────
model     = Backbone(input_dim, 128, n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

history      = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
best_val     = float('inf')
patience_ctr = 0
EPOCHS       = 30

print(f"Backbone — 3 datasets | {n_classes} classes | {input_dim} dim | {device}")
print("─" * 60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    tr_loss, tr_ok, tr_n = 0, 0, 0
    for xb, yb in tr_loader:
        optimizer.zero_grad()
        _, logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        tr_ok   += (logits.argmax(1)==yb).sum().item()
        tr_n    += len(yb)

    # Validation
    model.eval()
    vl_loss, vl_ok, vl_n = 0, 0, 0
    with torch.no_grad():
        for xb, yb in vl_loader:
            _, logits = model(xb)
            loss = criterion(logits, yb)
            vl_loss += loss.item()
            vl_ok   += (logits.argmax(1)==yb).sum().item()
            vl_n    += len(yb)

    avg_tr = tr_loss/len(tr_loader)
    avg_vl = vl_loss/len(vl_loader)
    history['train_loss'].append(avg_tr)
    history['val_loss'].append(avg_vl)
    history['train_acc'].append(tr_ok/tr_n*100)
    history['val_acc'].append(vl_ok/vl_n*100)
    scheduler.step(avg_vl)

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss:{avg_tr:.4f} Acc:{tr_ok/tr_n*100:.1f}% | "
              f"Val Loss:{avg_vl:.4f} Acc:{vl_ok/vl_n*100:.1f}%")

    if avg_vl < best_val:
        best_val = avg_vl
        torch.save(model.state_dict(),
                   f"{BASE}/models/backbone_3datasets.pth")
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= 10:
            print(f"\n⏹ Early stopping epoch {epoch+1}")
            break

model.load_state_dict(
    torch.load(f"{BASE}/models/backbone_3datasets.pth"))
print(f"\n✅ Backbone terminé | Val Loss : {best_val:.4f}")
print(f"✅ Sauvegardé → backbone_3datasets.pth")

# ════════════════════════════════════════════════════════════════
# SECTION 5 — Évaluation du Backbone
# F1-score par classe + Matrice de confusion + t-SNE
# ════════════════════════════════════════════════════════════════

model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for xb, yb in ts_loader:
        _, logits = model(xb)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_true.extend(yb.cpu().numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

print("=== F1-SCORE PAR CLASSE — BACKBONE 3 DATASETS ===\n")
print(classification_report(all_true, all_preds,
      target_names=le.classes_, digits=4))

f1_macro = f1_score(all_true, all_preds, average='macro')
f1_weighted = f1_score(all_true, all_preds, average='weighted')
print(f"F1 Macro    : {f1_macro:.4f}")
print(f"F1 Weighted : {f1_weighted:.4f}")

# ════════════════════════════════════════════════════════════════
# SECTION 6 — Gel du Backbone
#
# On gèle CNN + BiLSTM + embedding_layer.
# Seule la tête de classification reste entraînable.
# Le backbone gelé sera utilisé tel quel en Phase 2 (MAML)
# et Phase 3 (Few-Shot).
# ════════════════════════════════════════════════════════════════

for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

params_geles  = sum(p.numel() for p in model.parameters()
                    if not p.requires_grad)
params_libres = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

print(f" Backbone gelé")
print(f"  Paramètres gelés  : {params_geles:,}")
print(f"  Paramètres libres : {params_libres:,}")

torch.save(model.state_dict(),
           f"{BASE}/models/backbone_3datasets_GELE.pth")
print(f" Sauvegardé → backbone_3datasets_GELE.pth")

# ════════════════════════════════════════════════════════════════
# SECTION 7 — Fonctions Few-Shot + Indices par classe
# ════════════════════════════════════════════════════════════════

def generer_episode(X_data, indices_par_classe, n_way, k_shot, n_query, device):
    classes_choisies = random.sample(list(indices_par_classe.keys()), n_way)
    sx,sy,qx,qy = [],[],[],[]
    for nl,vl in enumerate(classes_choisies):
        idx = indices_par_classe[vl]
        choisis = (random.choices(idx,k=k_shot+n_query)
                   if len(idx)<k_shot+n_query
                   else random.sample(idx,k_shot+n_query))
        for i in choisis[:k_shot]: sx.append(X_data[i]); sy.append(nl)
        for i in choisis[k_shot:]: qx.append(X_data[i]); qy.append(nl)
    return (torch.FloatTensor(np.array(sx)).to(device),
            torch.LongTensor(np.array(sy)).to(device),
            torch.FloatTensor(np.array(qx)).to(device),
            torch.LongTensor(np.array(qy)).to(device),
            classes_choisies)

def calculer_prototypes(embeddings, labels, n_way):
    return torch.stack([embeddings[labels==c].mean(dim=0)
                        for c in range(n_way)])

def distance_cosinus(query_emb, prototypes):
    q = query_emb  / (query_emb.norm(dim=1,keepdim=True)  + 1e-8)
    p = prototypes / (prototypes.norm(dim=1,keepdim=True) + 1e-8)
    return 1 - torch.mm(q, p.t())

def few_shot_adaptation(model, X_data, indices_par_classe,
                        n_way, k_shot, n_query, inner_steps, alpha, device):
    model.train()
    classes_dispo = [c for c,idx in indices_par_classe.items()
                     if len(idx) >= k_shot+n_query]
    if len(classes_dispo) < n_way: n_way = len(classes_dispo)
    classes_choisies = random.sample(classes_dispo, n_way)
    sx,sy,qx,qy = [],[],[],[]
    for nl,vl in enumerate(classes_choisies):
        idx = indices_par_classe[vl]
        choisis = random.sample(idx, k_shot+n_query)
        for i in choisis[:k_shot]: sx.append(X_data[i]); sy.append(nl)
        for i in choisis[k_shot:]: qx.append(X_data[i]); qy.append(nl)
    sx = torch.FloatTensor(np.array(sx)).to(device)
    sy = torch.LongTensor(np.array(sy)).to(device)
    qx = torch.FloatTensor(np.array(qx)).to(device)
    qy = torch.LongTensor(np.array(qy)).to(device)
    poids_orig = {n:p.data.clone() for n,p in model.named_parameters()}
    inner_opt = optim.SGD(model.parameters(), lr=alpha)
    for _ in range(inner_steps):
        model.train()
        emb_s,_ = model(sx)
        proto    = calculer_prototypes(emb_s, sy, n_way)
        loss_in  = nn.CrossEntropyLoss()(-distance_cosinus(emb_s,proto), sy)
        inner_opt.zero_grad(); loss_in.backward(); inner_opt.step()
    model.eval()
    with torch.no_grad():
        emb_q,_  = model(qx)
        emb_s2,_ = model(sx)
        proto2    = calculer_prototypes(emb_s2, sy, n_way)
        logits_q  = -distance_cosinus(emb_q, proto2)
        acc = (logits_q.argmax(1)==qy).float().mean().item()*100
    for n,p in model.named_parameters(): p.data = poids_orig[n]
    return acc

# Indices par classe pour les épisodes MAML
indices_par_classe = defaultdict(list)
for idx, label in enumerate(y_tr_sm):
    indices_par_classe[int(label)].append(idx)

# Indices test
indices_test = defaultdict(list)
for idx, label in enumerate(y_ts_r):
    indices_test[int(label)].append(idx)

print("✅ Fonctions few-shot définies")
print(f"Classes : {list(le.classes_)}")

# ── 7.2 Entraînement MAML ────────────────────────────────────────

# Dégeler pour MAML (besoin de gradients sur tout le backbone)
model_maml = model
for param in model_maml.parameters():
    param.requires_grad = True
model_maml.train()

# Configuration finale (MAML v2)
ALPHA=0.005; BETA=0.001; INNER_STEPS=10
N_EPISODES=3000; PRINT_EVERY=300; N_WAY=5; K_SHOT=10; N_QUERY=15

meta_optimizer = optim.Adam(model_maml.parameters(), lr=BETA)
criterion_maml = nn.CrossEntropyLoss()
history_maml   = {'loss':[], 'acc':[]}
best_acc_maml  = 0

print(f"=== MAML v2 — Alpha:{ALPHA} | Beta:{BETA} | Steps:{INNER_STEPS} ===")
print(f"N-way:{N_WAY} | K-shot:{K_SHOT} | Épisodes:{N_EPISODES}")
print("─" * 55)

for episode in range(N_EPISODES):
    model_maml.train()
    sx, sy, qx, qy, _ = generer_episode(
        X_tr_sm, indices_par_classe, N_WAY, K_SHOT, N_QUERY, device)

    poids_orig = {n: p.data.clone()
                  for n, p in model_maml.named_parameters()}

    # Inner loop
    inner_opt = optim.SGD(model_maml.parameters(), lr=ALPHA)
    for _ in range(INNER_STEPS):
        model_maml.train()
        emb_s, _ = model_maml(sx)
        proto     = calculer_prototypes(emb_s, sy, N_WAY)
        loss_in   = criterion_maml(
            -distance_cosinus(emb_s, proto), sy)
        inner_opt.zero_grad(); loss_in.backward(); inner_opt.step()

    # Outer loop
    model_maml.train()
    emb_q, _  = model_maml(qx)
    emb_s2, _ = model_maml(sx)
    proto2     = calculer_prototypes(emb_s2, sy, N_WAY)
    logits_q   = -distance_cosinus(emb_q, proto2)
    loss_out   = criterion_maml(logits_q, qy)
    acc_out    = (logits_q.argmax(1)==qy).float().mean().item()*100

    meta_optimizer.zero_grad(); loss_out.backward()
    meta_optimizer.step()

    for n, p in model_maml.named_parameters():
        p.data = poids_orig[n]

    history_maml['loss'].append(loss_out.item())
    history_maml['acc'].append(acc_out)

    if (episode+1) % PRINT_EVERY == 0:
        avg_l = np.mean(history_maml['loss'][-PRINT_EVERY:])
        avg_a = np.mean(history_maml['acc'][-PRINT_EVERY:])
        print(f"Épisode {episode+1:4d}/{N_EPISODES} | "
              f"Loss:{avg_l:.4f} | Acc:{avg_a:.1f}%")
        if avg_a > best_acc_maml:
            best_acc_maml = avg_a
            torch.save(model_maml.state_dict(),
                       f"{BASE}/models/maml_3datasets_v2.pth")


# ════════════════════════════════════════════════════════════════
# SECTION 8 — Few-Shot Deployment (Phase 3) — domaine global
# ════════════════════════════════════════════════════════════════

N_TEST_EPISODES = 50
resultats_fewshot_v2 = {}
for k in [10, 20]:
    accs = [few_shot_adaptation(
        model_maml, X_ts_pca, indices_test,
        n_way=5, k_shot=k, n_query=15,
        inner_steps=10, alpha=0.005, device=device)
        for _ in range(N_TEST_EPISODES)]
    resultats_fewshot_v2[k] = (np.mean(accs), np.std(accs))
    print(f"{k}-shot global : {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")

# ════════════════════════════════════════════════════════════════
# SECTION 9 — Tests cross-domaine
#   9.1 UNSW-NB15 isolé
#   9.2 ToN-IoT (domaine jamais vu à l'entraînement)
# ════════════════════════════════════════════════════════════════

# ── 9.1 UNSW-NB15 isolé ───────────────────────────────────────────
mask_unsw    = df_fusion["source_dataset"].values == "unsw_nb15"
X_unsw_raw   = df_fusion[toutes_features].values.astype(np.float32)[mask_unsw]
y_unsw_raw   = le.transform(df_fusion["super_label"].values[mask_unsw])
X_unsw_pca   = pca.transform(scaler.transform(X_unsw_raw))
indices_unsw = defaultdict(list)
for idx, label in enumerate(y_unsw_raw):
    indices_unsw[int(label)].append(idx)
classes_ok_unsw = {c: idx for c, idx in indices_unsw.items() if len(idx) >= 25}

resultats_unsw = {}
for k in [10, 20]:
    accs = [few_shot_adaptation(
        model_maml, X_unsw_pca, classes_ok_unsw,
        n_way=min(5, len(classes_ok_unsw)), k_shot=k, n_query=15,
        inner_steps=10, alpha=0.005, device=device)
        for _ in range(N_TEST_EPISODES)]
    resultats_unsw[k] = (np.mean(accs), np.std(accs))
    print(f"UNSW {k}-shot : {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")

# ── 9.2 ToN-IoT — chargement (uniquement ici) ─────────────────────

# Le scaler et la PCA sont ceux ajustés sur les 3 datasets d'entraînement
df_toniot = pd.read_csv(f"{BASE}/datasets/toniot/train_test_network.csv")
df_toniot.columns = df_toniot.columns.str.strip()
df_toniot.replace([np.inf,-np.inf], np.nan, inplace=True)
df_toniot.dropna(inplace=True)
df_toniot.drop_duplicates(inplace=True)
df_toniot["super_label"] = df_toniot["type"].map({
    'normal':'Normal','backdoor':'Backdoor','ddos':'DoS','dos':'DoS',
    'injection':'Injection','password':'Brute_Force','scanning':'Reconnaissance',
    'ransomware':'Ransomware','xss':'Injection','mitm':'MITM'})
df_toniot.dropna(subset=["super_label"], inplace=True)
df_toniot_r = df_toniot.rename(columns={
    'duration':'flow_duration','src_bytes':'src_bytes','dst_bytes':'dst_bytes',
    'src_pkts':'fwd_packets','dst_pkts':'bwd_packets','src_ip_bytes':'src_load',
    'dst_ip_bytes':'dst_load','dst_port':'dst_port',
    'http_response_body_len':'response_body_len',
    'http_request_body_len':'trans_depth'})
df_toniot_r = df_toniot_r[
    df_toniot_r["super_label"].isin(set(le.classes_))].copy()
X_iot_df = df_toniot_r.reindex(columns=toutes_features)
for col in X_iot_df.columns:
    X_iot_df[col] = pd.to_numeric(X_iot_df[col], errors='coerce')
X_iot     = X_iot_df.fillna(0).values.astype(np.float32)
y_iot     = le.transform(df_toniot_r["super_label"].values)
X_iot_pca = pca.transform(scaler.transform(X_iot))
indices_iot = defaultdict(list)
for idx, label in enumerate(y_iot):
    indices_iot[int(label)].append(idx)
print(f"✅ ToN-IoT : {X_iot_pca.shape}")


print(f"NaN résiduels : {np.isnan(X_iot).sum()} | Inf : {np.isinf(X_iot).sum()}")

# ── 9.3 Test few-shot cross-domaine IoT ───────────────────────────
resultats_iot_v2 = {}
for k in [10, 20]:
    accs = [few_shot_adaptation(
        model_maml, X_iot_pca, indices_iot,
        n_way=5, k_shot=k, n_query=15,
        inner_steps=10, alpha=0.005, device=device)
        for _ in range(N_TEST_EPISODES)]
    resultats_iot_v2[k] = (np.mean(accs), np.std(accs))
    print(f"IoT {k}-shot : {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")

# ── 9.4 Récapitulatif tous domaines ───────────────────────────────
print(f"\n{'Domaine':<25} {'10-shot':>12} {'20-shot':>12}")
print("─" * 55)
for nom, r in [("Global (test set)", resultats_fewshot_v2),
               ("UNSW-NB15 isolé", resultats_unsw),
               ("ToN-IoT cross-domaine", resultats_iot_v2)]:
    print(f"{nom:<25} {r[10][0]:>10.1f}% {r[20][0]:>10.1f}%")
print("─" * 55)
print(f"{'Baseline hasard':<25} {'20.0%':>12} {'20.0%':>12}")

# ════════════════════════════════════════════════════════════════
# SECTION 10 — Graphes & Visualisations
# Tous les graphes sont générés ici et sauvegardés sur Drive.
# ════════════════════════════════════════════════════════════════

# ── 10.1 Courbes d'apprentissage ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train Loss',
         color='#2E86C1', linewidth=2)
ax1.plot(history['val_loss'],   label='Val Loss',
         color='#E74C3C', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Courbe de Loss — Backbone 3 Datasets (sans IoT)')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history['train_acc'], label='Train Acc',
         color='#2E86C1', linewidth=2)
ax2.plot(history['val_acc'],   label='Val Acc',
         color='#E74C3C', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
ax2.set_title("Courbe d'Accuracy — Backbone 3 Datasets (sans IoT)")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{BASE}/models/courbes_backbone.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ courbes_backbone.png sauvegardé")

# ── 10.2 Matrice de confusion ─────────────────────────────────────
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues', linewidths=0.5,
            annot_kws={"size":8})
plt.title('Matrice de Confusion — Backbone 3 Datasets', fontsize=13)
plt.ylabel('Vraie classe'); plt.xlabel('Classe prédite')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f"{BASE}/models/confusion_matrix_backbone.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ confusion_matrix_backbone.png sauvegardé")

# ── 10.3 Barplot F1 par classe ────────────────────────────────────
from sklearn.metrics import f1_score
f1_par_classe = f1_score(all_true, all_preds,
                          average=None, labels=range(n_classes))
colors = ['#2ECC71' if f>=0.8 else '#F39C12' if f>=0.5
          else '#E74C3C' for f in f1_par_classe]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(le.classes_, f1_par_classe, color=colors)
ax.set_xlabel('F1-Score', fontsize=12)
ax.set_title('F1-Score par Classe — Backbone 3 Datasets', fontsize=13)
ax.axvline(x=0.8, color='gray', linestyle='--',
           alpha=0.5, label='Seuil 0.80')
ax.set_xlim(0, 1.1)
for bar, val in zip(bars, f1_par_classe):
    ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)
ax.legend(); plt.tight_layout()
plt.savefig(f"{BASE}/models/f1_par_classe.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ f1_par_classe.png sauvegardé")

# ── 10.4 t-SNE des embeddings ─────────────────────────────────────
model.eval()
all_emb, all_lbl = [], []
with torch.no_grad():
    for xb, yb in ts_loader:
        emb, _ = model(xb)
        all_emb.append(emb.cpu().numpy())
        all_lbl.extend(yb.cpu().numpy())

all_emb = np.vstack(all_emb)
all_lbl = np.array(all_lbl)

idx     = np.random.choice(len(all_emb), 8000, replace=False)
emb_2d  = TSNE(n_components=2, random_state=42,
               perplexity=30, n_iter=1000).fit_transform(all_emb[idx])

colors_tsne = ['#2E86C1','#E74C3C','#27AE60','#F39C12',
               '#8E44AD','#1ABC9C','#E67E22','#D35400',
               '#95A5A6','#2ECC71','#C0392B','#3498DB']

plt.figure(figsize=(14, 9))
for i, (cls, color) in enumerate(zip(le.classes_, colors_tsne)):
    mask = all_lbl[idx] == i
    if mask.sum() > 0:
        plt.scatter(emb_2d[mask,0], emb_2d[mask,1],
                   c=color, label=f"{cls} (n={mask.sum()})",
                   alpha=0.6, s=8)
plt.title('t-SNE des Embeddings — Backbone 3 Datasets\n'
          '8 000 points du test set · 128 dim → 2D', fontsize=13)
plt.legend(markerscale=3, fontsize=9, loc='best')
plt.axis('off'); plt.tight_layout()
plt.savefig(f"{BASE}/models/tsne_backbone.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ tsne_backbone.png sauvegardé")

# ── 10.5 Comparaison Few-Shot — Global vs Cross-domaine ──────────
fig, ax = plt.subplots(figsize=(10, 6))

labels   = ['10-shot\nGlobal', '20-shot\nGlobal',
            '10-shot\nIoT', '20-shot\nIoT']
accs     = [resultats_fewshot_v2[10][0], resultats_fewshot_v2[20][0],
            resultats_iot_v2[10][0],     resultats_iot_v2[20][0]]
stds     = [resultats_fewshot_v2[10][1], resultats_fewshot_v2[20][1],
            resultats_iot_v2[10][1],     resultats_iot_v2[20][1]]
colors   = ['#2E86C1','#27AE60','#E67E22','#27AE60']

bars = ax.bar(labels, accs, color=colors, yerr=stds,
              capsize=5, error_kw={'linewidth':1.5})
ax.axhline(y=20, color='red', linestyle='--',
           linewidth=1.5, label='Baseline hasard (20%)')

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 1.5,
            f'{acc:.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=11)

ax.set_ylim(0, 105)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Résultats Few-Shot — Global vs Cross-Domaine IoT', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{BASE}/models/comparaison_fewshot.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ comparaison_fewshot.png sauvegardé")

# ════════════════════════════════════════════════════════════════
# SECTION 11 — Sauvegarde des Résultats Finaux
# ════════════════════════════════════════════════════════════════

resultats_finaux = {
    "configuration": "Option A — Backbone général sans IoT",
    "phase1_backbone": {
        "datasets_entraînement": ["NSL-KDD","CICIDS2017","UNSW-NB15"],
        "dataset_test_cross_domaine": "ToN-IoT (jamais vu)",
        "volume_train": int(X_tr_sm.shape[0]),
        "features": int(input_dim),
        "classes": int(n_classes),
        "variance_pca": float(round(variance*100, 2)),
        "val_loss_final": float(round(best_val, 4)),
    },
    "phase2_maml_v2": {
        "episodes": 3000,
        "n_way": 5,
        "k_shot": 10,
        "inner_steps": 10,
        "alpha": 0.005,
        "beta": 0.001,
        "meilleure_acc": float(round(best_acc_maml, 2)),
    },
    "phase3_fewshot_global": {
        "10_shot_acc": float(round(resultats_fewshot_v2[10][0], 2)),
        "10_shot_std": float(round(resultats_fewshot_v2[10][1], 2)),
        "20_shot_acc": float(round(resultats_fewshot_v2[20][0], 2)),
        "20_shot_std": float(round(resultats_fewshot_v2[20][1], 2)),
    },
    "phase3_cross_domaine_iot": {
        "10_shot_acc": float(round(resultats_iot_v2[10][0], 2)),
        "10_shot_std": float(round(resultats_iot_v2[10][1], 2)),
        "20_shot_acc": float(round(resultats_iot_v2[20][0], 2)),
        "20_shot_std": float(round(resultats_iot_v2[20][1], 2)),
    },
    "phase3_cross_domaine_unsw": {
        "10_shot_acc": float(round(resultats_unsw[10][0], 2)),
        "10_shot_std": float(round(resultats_unsw[10][1], 2)),
        "20_shot_acc": float(round(resultats_unsw[20][0], 2)),
        "20_shot_std": float(round(resultats_unsw[20][1], 2)),
    }
}

with open(f"{BASE}/models/resultats_option_A_final.json", "w") as f:
    json.dump(resultats_finaux, f, indent=2, ensure_ascii=False)

torch.save(model_maml.state_dict(),
           f"{BASE}/models/modele_FINAL_option_A.pth")

print("🎉 FRAMEWORK COMPLET — OPTION A")
print("─" * 55)
print(f"Backbone (3 datasets)     : Val Loss {best_val:.4f}")
print(f"MAML v2                   : {best_acc_maml:.1f}%")
print(f"10-shot global            : {resultats_fewshot_v2[10][0]:.1f}% ± {resultats_fewshot_v2[10][1]:.1f}%")
print(f"20-shot global            : {resultats_fewshot_v2[20][0]:.1f}% ± {resultats_fewshot_v2[20][1]:.1f}%")
print(f"UNSW 10-shot (isolé)      : {resultats_unsw[10][0]:.1f}% ± {resultats_unsw[10][1]:.1f}%")
print(f"UNSW 20-shot (isolé)      : {resultats_unsw[20][0]:.1f}% ± {resultats_unsw[20][1]:.1f}%")
print(f"IoT 10-shot (cross-dom.)  : {resultats_iot_v2[10][0]:.1f}% ± {resultats_iot_v2[10][1]:.1f}%")
print(f"IoT 20-shot (cross-dom.)  : {resultats_iot_v2[20][0]:.1f}% ± {resultats_iot_v2[20][1]:.1f}%")
print("─" * 55)
print(f"✅ Résultats → resultats_option_A_final.json")
print(f"✅ Modèle   → modele_FINAL_option_A.pth")
