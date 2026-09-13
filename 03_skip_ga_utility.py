# ── LEWATI GA (Cell 5), LANGSUNG SIAP UNTUK CELL 6 ──────────────────
# Tempel SEMUA isi sel ini sekaligus di 1 sel Colab, lalu jalankan.
# Cell 1-4 tetap dijalankan (ringan, ~1 menit, BUKAN training),
# tapi Cell 5 (GA) di-skip total - fitur diambil dari checkpoint Drive.

# ---------- Mount Drive & load checkpoint GA ----------
from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = '/content/drive/MyDrive/model_ga_bilstm_attention'

import pickle
with open(f'{SAVE_DIR}/ga_history.pkl', 'rb') as f:
    ga_history = pickle.load(f)

selected_indices  = ga_history['selected_indices']
selected_features = ga_history['selected_features']
best_fit_g        = ga_history['best_fitness']
print("Fitur GA dimuat dari checkpoint:", selected_features)

# ---------- Cell 1: Import ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc, time, warnings
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- Cell 2: Load 3 File (upload dulu ke Colab kalau belum ada) ----------
import xlrd
df_data  = pd.read_excel('Kelulusan_Data_Um_Ad_3.xls', engine='xlrd')
df_train = pd.read_excel('Kelulusan_Train.xls', engine='xlrd')
df_test  = pd.read_excel('Kelulusan_Test.xls',  engine='xlrd')

df = pd.concat([df_data, df_train, df_test], ignore_index=True)
df.columns = [c.strip() for c in df.columns]
print("Total baris:", len(df))

# ---------- Cell 3: Preprocessing & Feature Engineering ----------
df = df.drop(columns=['NAMA', 'STATUS NIKAH'])

num_cols = ['IPS 1','IPS 2','IPS 3','IPS 4',
            'IPS 5','IPS 6','IPS 7','IPS 8','IPK']
imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])

df['JENIS KELAMIN']    = (df['JENIS KELAMIN'] == 'PEREMPUAN').astype(int)
df['STATUS MAHASISWA'] = (df['STATUS MAHASISWA'] == 'BEKERJA').astype(int)
df['LABEL'] = (df['STATUS KELULUSAN'] == 'TEPAT').astype(int)

ips = ['IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8']
df['Rata2_IPS']    = df[ips].mean(axis=1).round(3)
df['Std_IPS']      = df[ips].std(axis=1).round(3)
df['IPS_Terbaik']  = df[ips].max(axis=1)
df['IPS_Terburuk'] = df[ips].min(axis=1)
df['Tren_IPS']     = (df['IPS 8'] - df['IPS 1']).round(3)
df['Delta_IPS_12'] = (df['IPS 2'] - df['IPS 1']).round(3)
df['Delta_IPS_23'] = (df['IPS 3'] - df['IPS 2']).round(3)
df['Delta_IPS_34'] = (df['IPS 4'] - df['IPS 3']).round(3)
df['Delta_IPS_45'] = (df['IPS 5'] - df['IPS 4']).round(3)
df['Delta_IPS_56'] = (df['IPS 6'] - df['IPS 5']).round(3)
df['Delta_IPS_67'] = (df['IPS 7'] - df['IPS 6']).round(3)
df['Delta_IPS_78'] = (df['IPS 8'] - df['IPS 7']).round(3)
df['Gap_IPK_IPS_Akhir'] = (df['IPK'] - df['IPS 8']).round(3)
df['Gap_IPK_Rata2']     = (df['IPK'] - df['Rata2_IPS']).round(3)

FEATURE_COLS = [
    'JENIS KELAMIN', 'STATUS MAHASISWA', 'UMUR',
    'IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK',
    'Rata2_IPS','Std_IPS','IPS_Terbaik','IPS_Terburuk','Tren_IPS',
    'Delta_IPS_12','Delta_IPS_23','Delta_IPS_34',
    'Delta_IPS_45','Delta_IPS_56','Delta_IPS_67','Delta_IPS_78',
    'Gap_IPK_IPS_Akhir','Gap_IPK_Rata2',
]
N_FEATURES = len(FEATURE_COLS)
N_CLASSES  = 2
print("Total fitur:", N_FEATURES, "(harus 26)")

# ---------- Cell 4: Normalisasi ----------
scaler = MinMaxScaler()
X_all  = scaler.fit_transform(df[FEATURE_COLS].values)
y_all  = df['LABEL'].values.astype(int)

class_counts  = Counter(y_all)
total         = len(y_all)
class_weights = {k: total / (N_CLASSES * v) for k, v in class_counts.items()}

print("Shape X_all:", X_all.shape)
print("Class weights:", class_weights)

# ---------- LEWATI Cell 5 (GA) - langsung pakai checkpoint ----------
X_selected = X_all[:, selected_indices]
print()
print("=" * 55)
print("GA DILEWATI - pakai fitur checkpoint dari Drive")
print("Fitness tersimpan :", round(best_fit_g, 4))
print("X_selected shape  :", X_selected.shape, "(harus 18 kolom)")
print("=" * 55)
print()
print("✅ SEMUA VARIABEL SIAP: X_all, y_all, X_selected, class_weights,")
print("   FEATURE_COLS, selected_features, selected_indices, N_CLASSES, SEED")
print("   Lanjutkan ke Cell 6 (perbandingan 7 model) seperti biasa.")
