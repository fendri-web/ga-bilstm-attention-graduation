# %% [markdown]
# # Reproduksi Kode Paper Konferensi ICERA (GA-BiLSTM) — Versi Diperbaiki
#
# Perbaikan dari notebook asli (`GA_BiLSTM.ipynb`):
# 1. **Cell 2**: load 3 file (`Kelulusan_Data_Um_Ad_3.xls` + `Train` + `Test`),
#    notebook asli yang diterima sempat kehilangan baris load file pertama.
# 2. **Nama kolom**: `'IPK '` (ada spasi) dinormalisasi jadi `'IPK'` supaya
#    konsisten dengan nama kolom asli di file .xls.
# 3. **Uji signifikansi statistik**: sebelumnya pakai data fold hardcoded
#    yang TIDAK LENGKAP untuk BiLSTM & BiLSTM+IG (cuma 4 dari 5 fold — bakal
#    error). Sekarang seluruh model menyimpan `fold_acc` langsung dari hasil
#    K-Fold CV yang sungguhan dijalankan di Cell 6/6B, bukan angka manual.
# 4. **Cell 6B duplikat** (ada 2 versi identik di notebook asli) — dihapus,
#    disatukan jadi 1 versi saja.
#
# **Cara pakai di Colab:**
# 1. Runtime > Change runtime type > pilih **GPU (T4)**
# 2. Upload 3 file: `Kelulusan_Data_Um_Ad_3.xls`, `Kelulusan_Train.xls`, `Kelulusan_Test.xls`
# 3. Jalankan sel berurutan dari atas (Runtime > Run all)
#
# **Catatan waktu**: dengan GPU, estimasi total ~30-60 menit (GA jadi bagian
# paling lama). Tanpa GPU bisa 5-7 jam.

# %% [markdown]
# ## Cell 1: Import Library

# %%
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

plt.rcParams.update({
    'figure.dpi': 120, 'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

print("TF version :", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU        :", gpus[0].name, "| Status: GPU aktif!")
else:
    print("GPU        : Tidak tersedia — Runtime > Change runtime type > GPU T4")

# %% [markdown]
# ## Cell 2 (DIPERBAIKI): Load 3 File & Eksplorasi Dataset

# %%
import xlrd

df_data  = pd.read_excel('Kelulusan_Data_Um_Ad_3.xls', engine='xlrd')
df_train = pd.read_excel('Kelulusan_Train.xls', engine='xlrd')
df_test  = pd.read_excel('Kelulusan_Test.xls',  engine='xlrd')

df = pd.concat([df_data, df_train, df_test], ignore_index=True)
df.columns = [c.strip() for c in df.columns]  # normalisasi nama kolom

print("=" * 55)
print("  INFORMASI DATASET")
print("=" * 55)
print("  Data_Um_Ad_3 :", len(df_data), "baris")
print("  Train        :", len(df_train), "baris")
print("  Test         :", len(df_test),  "baris")
print("  Total        :", len(df),       "baris")
print("  Kolom        :", len(df.columns))
print()

print("  Distribusi Label (STATUS KELULUSAN):")
dist = df['STATUS KELULUSAN'].value_counts()
for label, count in dist.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 3)
    print("   ", label, bar, count, "(" + str(round(pct,1)) + "%)")

print()
print("  Missing Values:")
miss = df.isnull().sum()
miss = miss[miss > 0]
if len(miss) > 0:
    for col, n in miss.items():
        print("   ", col, ":", n, "missing")
else:
    print("   Tidak ada")

print()
ips_cols = ['IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8']
print("  Statistik IPS per Semester:")
print(df[ips_cols].describe().round(2).to_string())

# %% [markdown]
# ## Cell 3 (DIPERBAIKI): Preprocessing & Feature Engineering
# (kolom 'IPK ' -> 'IPK', sudah dinormalisasi di Cell 2)

# %%
df = df.drop(columns=['NAMA', 'STATUS NIKAH'])
print("✅ Hapus NAMA & STATUS NIKAH (tidak informatif)")

num_cols = ['IPS 1','IPS 2','IPS 3','IPS 4',
            'IPS 5','IPS 6','IPS 7','IPS 8','IPK']
imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])
print("✅ Imputasi missing values dengan median")
print("   Missing sekarang:", df.isnull().sum().sum())

df['JENIS KELAMIN']    = (df['JENIS KELAMIN'] == 'PEREMPUAN').astype(int)
df['STATUS MAHASISWA'] = (df['STATUS MAHASISWA'] == 'BEKERJA').astype(int)
df['LABEL'] = (df['STATUS KELULUSAN'] == 'TEPAT').astype(int)
print("✅ Encoding: PEREMPUAN=1, BEKERJA=1, TEPAT=1")

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
print("✅ Feature Engineering: 14 fitur baru dibuat")

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

print()
print("=" * 50)
print("  RINGKASAN FITUR")
print("=" * 50)
print("  Fitur asli        : 12")
print("  Fitur FE          : 14")
print("  Total fitur       :", N_FEATURES)

y = df['LABEL'].values
print()
print("  Distribusi label:")
print("   TEPAT     :", sum(y==1), "(" + str(round(sum(y==1)/len(y)*100,1)) + "%)")
print("   TERLAMBAT :", sum(y==0), "(" + str(round(sum(y==0)/len(y)*100,1)) + "%)")

# %% [markdown]
# ## Cell 4: Normalisasi & Persiapan Data

# %%
scaler = MinMaxScaler()
X_all  = scaler.fit_transform(df[FEATURE_COLS].values)
y_all  = df['LABEL'].values.astype(int)

print("Shape X_all :", X_all.shape)
print("Shape y_all :", y_all.shape)

class_counts  = Counter(y_all)
total         = len(y_all)
class_weights = {k: total / (N_CLASSES * v) for k, v in class_counts.items()}

print()
print("Class weights:")
label_map = {1: 'TEPAT', 0: 'TERLAMBAT'}
for k, v in sorted(class_weights.items()):
    print("  ", label_map[k], ":", round(v, 4))

print()
print("✅ X_all dan y_all siap!")

# %% [markdown]
# ## Cell 5: Genetic Algorithm (Parameter Asli: pop=30, gen=50)
# **Ini bagian paling lama** — dengan GPU estimasi 20-40 menit.

# %%
POP_SIZE = 30
N_GEN    = 50
PC       = 0.8
PM       = 0.01
N_BITS   = N_FEATURES

print("Parameter GA:")
print("  Populasi  :", POP_SIZE)
print("  Generasi  :", N_GEN)
print("  Kromosom  :", N_BITS, "bit")

def evaluate_fitness(chromosome, X, y, cw):
    sel = np.where(chromosome == 1)[0]
    if len(sel) == 0:
        return 0.0
    X_sel = X[:, sel]
    try:
        tf.keras.backend.clear_session()
        Xtr, Xv, ytr, yv = train_test_split(
            X_sel, y, test_size=0.2, random_state=42, stratify=y)
        Xtr3 = Xtr.reshape(-1, 1, len(sel))
        Xv3  = Xv.reshape(-1, 1, len(sel))
        m = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(16), input_shape=(1, len(sel))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(N_CLASSES, activation='softmax')
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        m.fit(Xtr3, ytr, epochs=15, batch_size=64, class_weight=cw, verbose=0)
        _, acc = m.evaluate(Xv3, yv, verbose=0)
        del m; gc.collect(); tf.keras.backend.clear_session()
        return float(acc)
    except Exception:
        tf.keras.backend.clear_session()
        return 0.0

def tournament_select(pop, fit, k=3):
    sel = []
    for _ in range(len(pop)):
        c = np.random.choice(len(pop), k, replace=False)
        sel.append(pop[c[np.argmax(fit[c])]].copy())
    return np.array(sel)

def do_crossover(p1, p2):
    if np.random.rand() < PC:
        pt = np.random.randint(1, len(p1))
        return (np.concatenate([p1[:pt], p2[pt:]]),
                np.concatenate([p2[:pt], p1[pt:]]))
    return p1.copy(), p2.copy()

def mutate(ch):
    m = ch.copy()
    m[np.random.rand(len(m)) < PM] ^= 1
    return m

np.random.seed(SEED)
pop = np.random.randint(0, 2, size=(POP_SIZE, N_BITS))
for k in range(POP_SIZE):
    if pop[k].sum() < 2:
        pop[k, np.random.choice(N_BITS, 2, replace=False)] = 1

best_hist, avg_hist = [], []
best_fit_g = 0.0
best_chrom = pop[0].copy()

print()
print("=" * 52)
print("  EVOLUSI GA -", N_GEN, "gen x", POP_SIZE, "kromosom")
print("=" * 52)
print("Gen  |   Best  |   Avg   | Fitur |  Sisa")
print("-" * 42)

t0 = time.time()
for gen in range(N_GEN):
    t_gen = time.time()
    fit = np.array([evaluate_fitness(c, X_all, y_all, class_weights) for c in pop])

    bi = np.argmax(fit)
    if fit[bi] > best_fit_g:
        best_fit_g = fit[bi]
        best_chrom = pop[bi].copy()

    best_hist.append(best_fit_g)
    avg_hist.append(fit.mean())

    # Auto-stop DIPERKETAT: butuh 15 generasi stabil (bukan 5) dan toleransi
    # lebih kecil, supaya GA punya cukup ruang eksplorasi via mutasi/crossover
    # sebelum dianggap benar-benar konvergen (menghindari berhenti prematur
    # di local optimum seperti yang terjadi sebelumnya - 18 fitur/fitness
    # 0.938 padahal solusi lebih baik, 11 fitur/fitness 0.964, mungkin masih
    # bisa ditemukan kalau diberi generasi lebih banyak).
    if len(best_hist) >= 15 and all(
            abs(best_hist[-1]-best_hist[-i]) < 0.0001 for i in range(1, 15)):
        print(str(gen+1).rjust(4), "| KONVERGEN — auto stop")
        break

    t_sisa = (time.time()-t_gen) * (N_GEN-gen-1)
    if (gen+1) % 5 == 0 or gen == 0:
        print(str(gen+1).rjust(4), "|",
              str(round(best_fit_g, 4)).rjust(7), "|",
              str(round(fit.mean(), 4)).rjust(7), "|",
              str(int(best_chrom.sum())).rjust(5), "|",
              str(round(t_sisa/60, 1)).rjust(5) + "m")

    new_pop  = tournament_select(pop, fit)
    children = []
    for j in range(0, POP_SIZE-1, 2):
        c1, c2 = do_crossover(new_pop[j], new_pop[j+1])
        children.extend([mutate(c1), mutate(c2)])
    children[0] = best_chrom.copy()
    pop = np.array(children[:POP_SIZE])
    gc.collect()

total_menit = (time.time()-t0) / 60
print("-" * 42)
print("Selesai:", round(total_menit, 1), "menit")

selected_indices  = np.where(best_chrom == 1)[0]
selected_features = [FEATURE_COLS[i] for i in selected_indices]
X_selected        = X_all[:, selected_indices]

print()
print("HASIL GA:")
print("  Fitness   :", round(best_fit_g, 4), "(" + str(round(best_fit_g*100, 2)) + "%)")
print("  Fitur     :", len(selected_features), "dari", N_BITS)
for k, f in enumerate(selected_features, 1):
    print("   ", k, ".", f)

np.save('ga_result.npy', {
    'best_chrom': best_chrom, 'selected_indices': selected_indices,
    'selected_features': selected_features, 'best_fitness': best_fit_g,
    'best_hist': best_hist, 'avg_hist': avg_hist,
    'X_all': X_all, 'X_selected': X_selected, 'y_all': y_all,
    'class_weights': class_weights, 'FEATURE_COLS': FEATURE_COLS,
    'N_FEATURES': N_FEATURES, 'N_CLASSES': N_CLASSES, 'SEED': SEED,
}, allow_pickle=True)
print("Checkpoint -> ga_result.npy")

fig, ax = plt.subplots(figsize=(10, 5))
g = range(1, len(best_hist)+1)
ax.plot(g, best_hist, color='#534AB7', linewidth=2.5, label='Fitness Terbaik')
ax.plot(g, avg_hist, color='#1D9E75', linewidth=1.8, linestyle='--', alpha=0.8, label='Fitness Rata-rata')
ax.fill_between(g, avg_hist, best_hist, alpha=0.1, color='#534AB7')
ax.set_xlabel('Generasi'); ax.set_ylabel('Fitness (Akurasi BiLSTM)')
ax.set_title('Konvergensi Genetic Algorithm', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig('fig6_ga_convergence_actual.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Cell 6: Perbandingan 6 Model (DIPERBAIKI: fold_acc disimpan utk semua model)

# %%
ck = np.load('ga_result.npy', allow_pickle=True).item()
X_all             = ck['X_all']
y_all             = ck['y_all']
X_selected        = ck['X_selected']
class_weights     = ck['class_weights']
selected_features = ck['selected_features']
N_CLASSES         = int(ck['N_CLASSES'])
FEATURE_COLS      = ck['FEATURE_COLS']
SEED              = int(ck['SEED'])
n_sel             = X_selected.shape[1]

print("Checkpoint loaded! Fitur GA:", n_sel)

selector = SelectKBest(mutual_info_classif, k=n_sel)
selector.fit(X_all, y_all)
X_ig = selector.transform(X_all)

def build_model(n_feat, arch='bilstm'):
    tf.keras.backend.clear_session()
    inp = tf.keras.Input(shape=(1, n_feat))
    if arch == 'lstm':
        x = tf.keras.layers.LSTM(64)(inp)
    elif arch == 'bilstm_attn':
        # ── Feature-level Attention ──────────────────────────────
        # Input cuma 1 timestep (bukan sequence waktu sungguhan),
        # jadi Attention di sini menimbang PENTINGNYA TIAP FITUR
        # (bukan antar-waktu) sebelum masuk BiLSTM - melengkapi
        # seleksi statis GA dengan bobot adaptif yang dipelajari.
        flat = tf.keras.layers.Flatten()(inp)  # (batch, n_feat)
        attn_scores  = tf.keras.layers.Dense(n_feat, activation='tanh')(flat)
        attn_weights = tf.keras.layers.Dense(n_feat, activation='softmax',
                                              name='feature_attention')(attn_scores)
        weighted = tf.keras.layers.Multiply()([flat, attn_weights])
        weighted = tf.keras.layers.Reshape((1, n_feat))(weighted)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(weighted)
    else:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(inp)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def run_kfold(X, y, arch, cw, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    al, pl, rl, fl = [], [], [], []
    trues, preds = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if arch == 'dt':
            cw2 = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
            m = DecisionTreeClassifier(max_depth=10,
                class_weight=dict(zip(np.unique(ytr), cw2)), random_state=SEED)
            m.fit(Xtr, ytr)
            yp = m.predict(Xte)
        else:
            nf = X.shape[1]
            Xtr3 = Xtr.reshape(-1, 1, nf)
            Xte3 = Xte.reshape(-1, 1, nf)
            m = build_model(nf, arch)
            es = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)
            m.fit(Xtr3, ytr, epochs=50, batch_size=64, class_weight=cw,
                  validation_split=0.1, callbacks=[es], verbose=0)
            yp = np.argmax(m.predict(Xte3, verbose=0), axis=1)
            del m; gc.collect(); tf.keras.backend.clear_session()
        al.append(accuracy_score(yte, yp))
        pl.append(precision_score(yte, yp, average='macro', zero_division=0))
        rl.append(recall_score(yte, yp, average='macro', zero_division=0))
        fl.append(f1_score(yte, yp, average='macro', zero_division=0))
        trues.extend(yte); preds.extend(yp)
        print("  Fold", fold+1, "Acc=", round(al[-1], 4), "F1=", round(fl[-1], 4))
    return {'accuracy': np.mean(al), 'precision': np.mean(pl),
            'recall': np.mean(rl), 'f1': np.mean(fl),
            'fold_acc': al, 'fold_f1': fl,  # <-- PERBAIKAN: selalu simpan per-fold
            'y_true': trues, 'y_pred': preds}

results = {}
models = [
    ('Decision Tree', X_all,      'dt'),
    ('LSTM',          X_all,      'lstm'),
    ('BiLSTM',        X_all,      'bilstm'),
    ('BiLSTM+IG',     X_ig,       'bilstm'),
    ('BiLSTM-Attention',   X_all,      'bilstm_attn'),  # <-- MODEL BARU: Attention tanpa GA (desain 2x2)
    ('GA-BiLSTM',     X_selected, 'bilstm'),
    ('GA-BiLSTM-Attention', X_selected, 'bilstm_attn'),
]
for name, X_in, arch in models:
    print("\n--- Model:", name, "---")
    results[name] = run_kfold(X_in, y_all, arch, class_weights)
    r = results[name]
    print("  -> Acc={:.4f} Prec={:.4f} Rec={:.4f} F1={:.4f}".format(
        r['accuracy'], r['precision'], r['recall'], r['f1']))

print("\n" + "="*65)
print("  TABEL PERBANDINGAN PERFORMA MODEL (K-Fold k=5)")
print("="*65)
print("{:<18}{:>10}{:>10}{:>10}{:>10}".format(
    "Model","Akurasi","Precision","Recall","F1-Score"))
print("-"*60)
best_acc = max(r['accuracy'] for r in results.values())
for name, r in results.items():
    mk = " <-TERBAIK" if r['accuracy'] == best_acc else ""
    print("{:<18}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{}".format(
        name, r['accuracy'], r['precision'], r['recall'], r['f1'], mk))

np.save('results.npy', results, allow_pickle=True)
print("\nTersimpan -> results.npy")

# %% [markdown]
# ## Cell 6B: Baseline Tambahan — Random Forest

# %%
def run_kfold_rf(X, y, k=5, seed=SEED):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    al, pl, rl, fl = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
        class_weight_dict = dict(zip(np.unique(ytr), cw))
        m = RandomForestClassifier(n_estimators=95, max_depth=5,
            class_weight=class_weight_dict, random_state=seed, n_jobs=-1)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        al.append(accuracy_score(yte, yp))
        pl.append(precision_score(yte, yp, average='macro', zero_division=0))
        rl.append(recall_score(yte, yp, average='macro', zero_division=0))
        fl.append(f1_score(yte, yp, average='macro', zero_division=0))
        print("  Fold", fold+1, "Acc=", round(al[-1], 4), "F1=", round(fl[-1], 4))
    return {'accuracy': np.mean(al), 'precision': np.mean(pl),
            'recall': np.mean(rl), 'f1': np.mean(fl),
            'fold_acc': al, 'fold_f1': fl}

print("\n--- Model: Random Forest ---")
results['Random Forest'] = run_kfold_rf(X_all, y_all)
r = results['Random Forest']
print("  -> Acc={:.4f} Prec={:.4f} Rec={:.4f} F1={:.4f}".format(
    r['accuracy'], r['precision'], r['recall'], r['f1']))

print("\n" + "="*65)
print("  TABEL PERBANDINGAN PERFORMA MODEL (6 model, K-Fold k=5)")
print("="*65)
print("{:<18}{:>10}{:>10}{:>10}{:>10}".format(
    "Model","Akurasi","Precision","Recall","F1-Score"))
print("-"*60)
best_acc = max(r['accuracy'] for r in results.values())
for name, r in results.items():
    mk = " <-TERBAIK" if r['accuracy'] == best_acc else ""
    print("{:<18}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{}".format(
        name, r['accuracy'], r['precision'], r['recall'], r['f1'], mk))

np.save('results_extended.npy', results, allow_pickle=True)
print("\nTersimpan -> results_extended.npy")

# %% [markdown]
# ## Cell 7: Confusion Matrix + Early Prediction

# %%
y_true = results['GA-BiLSTM']['y_true']
y_pred = results['GA-BiLSTM']['y_pred']
label_names = ['TERLAMBAT', 'TEPAT']
cm = confusion_matrix(y_true, y_pred)

print("\nClassification Report GA-BiLSTM:")
print(classification_report(y_true, y_pred, target_names=label_names))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, ax=axes[0], annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_xlabel('Prediksi'); axes[0].set_ylabel('Aktual')
axes[0].set_title('Confusion Matrix (Jumlah)', fontweight='bold')

cm_pct = (cm.astype(float).T / cm.sum(axis=1)).T * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, ax=axes[1], annot_kws={'size': 11}, vmin=0, vmax=100)
axes[1].set_xlabel('Prediksi'); axes[1].set_ylabel('Aktual')
axes[1].set_title('Confusion Matrix (%)', fontweight='bold')
plt.suptitle('Confusion Matrix — Model GA-BiLSTM', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig7_confusion_matrix_actual.png', dpi=200, bbox_inches='tight')
plt.show()

def build_bilstm_final(nf):
    tf.keras.backend.clear_session()
    inp = tf.keras.Input(shape=(1, nf))
    x   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(inp)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def run_early(X, y, cw, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    al, pl, rl, fl = [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr3 = X[tr].reshape(-1, 1, X.shape[1])
        Xte3 = X[te].reshape(-1, 1, X.shape[1])
        m = build_bilstm_final(X.shape[1])
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        m.fit(Xtr3, y[tr], epochs=50, batch_size=64, class_weight=cw,
              validation_split=0.1, callbacks=[es], verbose=0)
        yp = np.argmax(m.predict(Xte3, verbose=0), axis=1)
        al.append(accuracy_score(y[te], yp))
        pl.append(precision_score(y[te], yp, average='macro', zero_division=0))
        rl.append(recall_score(y[te], yp, average='macro', zero_division=0))
        fl.append(f1_score(y[te], yp, average='macro', zero_division=0))
        del m; gc.collect(); tf.keras.backend.clear_session()
        print("  Fold", fold+1, "Acc=", round(al[-1], 4), "F1=", round(fl[-1], 4))
    return {'accuracy': np.mean(al), 'precision': np.mean(pl),
            'recall': np.mean(rl), 'f1': np.mean(fl)}

feat_s12 = [f for f in ['JENIS KELAMIN','STATUS MAHASISWA','UMUR','IPS 1','IPS 2'] if f in FEATURE_COLS]
feat_s14 = [f for f in ['JENIS KELAMIN','STATUS MAHASISWA','UMUR','IPS 1','IPS 2','IPS 3','IPS 4',
                          'Delta_IPS_12','Delta_IPS_23','Delta_IPS_34'] if f in FEATURE_COLS]
feat_s18 = selected_features

scenarios = {
    'Semester 1-2 (Sangat Dini)': feat_s12,
    'Semester 1-4 (Dini)'       : feat_s14,
    'Semester 1-8 (Lengkap)'    : feat_s18,
}

early_results = {}
for nama, feats in scenarios.items():
    idx = [FEATURE_COLS.index(f) for f in feats if f in FEATURE_COLS]
    if len(idx) < 2:
        idx = list(range(min(5, X_all.shape[1])))
    X_ep = X_all[:, idx]
    print("\n--- Early:", nama, "(" + str(len(idx)) + " fitur) ---")
    early_results[nama] = run_early(X_ep, y_all, class_weights)
    r = early_results[nama]
    print("  -> Acc={:.4f} F1={:.4f}".format(r['accuracy'], r['f1']))

print("\n" + "="*65)
print("  TABEL VII. HASIL EARLY PREDICTION")
print("="*65)
print("{:<28}{:>10}{:>10}{:>10}{:>10}".format(
    "Skenario","Akurasi","Precision","Recall","F1-Score"))
print("-"*62)
for nama, r in early_results.items():
    print("{:<28}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format(
        nama, r['accuracy'], r['precision'], r['recall'], r['f1']))

# %% [markdown]
# ## Cell 8 (DIPERBAIKI): Uji Signifikansi Statistik
# Sekarang pakai `fold_acc` LIVE dari `results` (Cell 6/6B), bukan angka
# hardcoded yang sebelumnya tidak lengkap untuk BiLSTM & BiLSTM+IG.

# %%
from scipy import stats

MODEL_UTAMA = 'GA-BiLSTM-Attention'
ALPHA = 0.05

fold_acc = {name: r['fold_acc'] for name, r in results.items()}

ga = np.array(fold_acc[MODEL_UTAMA])
print(f"{MODEL_UTAMA} — mean akurasi: {ga.mean():.4f} | fold: {ga.tolist()}")
print()
print(f"{'Model Pembanding':<16}{'Mean Acc':>10}{'t-stat':>10}{'p (t-test)':>12}"
      f"{'p (Wilcoxon)':>14}{'Signifikan?':>13}")
print("-" * 75)

hasil = []
for name, scores in fold_acc.items():
    if name == MODEL_UTAMA:
        continue
    other = np.array(scores)
    t_stat, t_p = stats.ttest_rel(ga, other)
    try:
        w_stat, w_p = stats.wilcoxon(ga, other)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    signif = "Ya (t-test)" if t_p < ALPHA else "Tidak"
    hasil.append((name, other.mean(), t_stat, t_p, w_p, signif))
    print(f"{name:<16}{other.mean():>10.4f}{t_stat:>10.3f}{t_p:>12.4f}"
          f"{w_p:>14.4f}{signif:>13}")

print("-" * 75)
print("\nCatatan: dengan n=5 pasangan fold, Wilcoxon dua-sisi punya batas")
print("p minimum 0.0625 -> secara struktural tidak pernah bisa mencapai")
print("p<0.05 berapapun konsistennya perbedaan. Gunakan t-test sebagai acuan utama.")

print("\n✅ Semua eksperimen selesai!")

# %% [markdown]
# ## Cell 8B: Analisis Ablation 2×2 — Kontribusi GA vs Attention
#
# Desain 2×2 untuk mengisolasi kontribusi masing-masing komponen secara
# terpisah dan gabungan, sebagai kontribusi utama yang membedakan penelitian
# ini dari paper konferensi (yang tidak melakukan ablation study semacam ini).
#
# |                | Tanpa Attention | Dengan Attention      |
# |----------------|------------------|------------------------|
# | Tanpa GA (26 fitur) | BiLSTM       | BiLSTM-Attention       |
# | Dengan GA (fitur GA)| GA-BiLSTM    | GA-BiLSTM-Attention    |

# %%
print("=" * 65)
print("  TABEL ABLATION 2×2")
print("=" * 65)
print(f"{'Varian':<22}{'Akurasi':>10}{'Precision':>11}{'Recall':>10}{'F1-Score':>10}")
print("-" * 63)
ablation_order = ['BiLSTM', 'GA-BiLSTM', 'BiLSTM-Attention', 'GA-BiLSTM-Attention']
for name in ablation_order:
    r = results[name]
    print(f"{name:<22}{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>10.4f}{r['f1']:>10.4f}")

print()
print("=" * 65)
print("  KONTRIBUSI TIAP KOMPONEN (selisih akurasi, paired t-test)")
print("=" * 65)

perbandingan_ablation = [
    ("Efek GA saja",          'GA-BiLSTM',           'BiLSTM'),
    ("Efek Attention saja",   'BiLSTM-Attention',    'BiLSTM'),
    ("Efek Attention | GA sudah ada", 'GA-BiLSTM-Attention', 'GA-BiLSTM'),
    ("Efek GA | Attention sudah ada", 'GA-BiLSTM-Attention', 'BiLSTM-Attention'),
]

print(f"{'Perbandingan':<32}{'Selisih Acc':>13}{'t-stat':>9}{'p-value':>10}{'Signifikan?':>13}")
print("-" * 77)

hasil_ablation = []
for label, model_a, model_b in perbandingan_ablation:
    a = np.array(results[model_a]['fold_acc'])
    b = np.array(results[model_b]['fold_acc'])
    selisih = a.mean() - b.mean()
    t_stat, t_p = stats.ttest_rel(a, b)
    signif = "Ya" if t_p < ALPHA else "Tidak"
    hasil_ablation.append((label, model_a, model_b, selisih, t_stat, t_p, signif))
    print(f"{label:<32}{selisih:>+13.4f}{t_stat:>9.3f}{t_p:>10.4f}{signif:>13}")

print("-" * 77)
print()
print("Interpretasi:")
print("  - 'Efek GA saja'  : apakah seleksi fitur GA membantu, tanpa Attention")
print("  - 'Efek Attention saja' : apakah Attention membantu, tanpa seleksi GA")
print("  - Baris 3 & 4     : apakah menambah komponen kedua (setelah satu")
print("    komponen sudah ada) masih memberi kontribusi tambahan yang berarti")

# Simpan hasil ablation 2x2 (pakai SAVE_DIR kalau sudah pernah didefinisikan
# di Cell 9, kalau belum simpan ke folder lokal biasa)
# Simpan hasil ablation 2x2 (pakai SAVE_DIR kalau sudah pernah didefinisikan
# di Cell 9, kalau belum simpan ke folder lokal biasa)
import pickle
import os
_dir = SAVE_DIR if 'SAVE_DIR' in dir() else '.'
with open(os.path.join(_dir, 'ablation_2x2.pkl'), 'wb') as f:
    pickle.dump(hasil_ablation, f)
print("\n✅ Tersimpan ->", os.path.join(_dir, 'ablation_2x2.pkl'))



# %% [markdown]
# ## Cell 9: Latih & Simpan Model Final untuk Dipakai Ulang
#
# **Penting**: model dari K-Fold CV di Cell 6/6B cuma untuk EVALUASI (5 model
# terpisah per fold, tidak ada satupun yang "model akhir"). Di sel ini, model
# **GA-BiLSTM-Attention** dilatih ULANG dari nol memakai **seluruh data**
# (dengan sedikit disisihkan untuk validation/early stopping), lalu disimpan
# bersama SEMUA komponen preprocessing (scaler, imputer, daftar fitur GA)
# supaya bisa dipakai langsung untuk prediksi data mahasiswa baru nantinya
# tanpa perlu mengulang seluruh pipeline dari awal.

# %%
import joblib
import json
import os

# --- Mount Google Drive supaya model TERSIMPAN PERMANEN ---
# (tidak hilang meski runtime Colab restart/putus koneksi)
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    DRIVE_BASE = '/content/drive/MyDrive/model_ga_bilstm_attention'
except ImportError:
    # bukan di Colab (mis. dijalankan lokal) -> simpan di folder biasa
    DRIVE_BASE = 'model_final_ga_bilstm_attention'

SAVE_DIR = DRIVE_BASE
print("Model akan disimpan permanen di:", SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. Split train/val untuk early stopping (bukan untuk evaluasi akhir) ---
Xtr, Xval, ytr, yval = train_test_split(
    X_selected, y_all, test_size=0.15, random_state=SEED, stratify=y_all)

Xtr3  = Xtr.reshape(-1, 1, X_selected.shape[1])
Xval3 = Xval.reshape(-1, 1, X_selected.shape[1])

print("Melatih model final GA-BiLSTM-Attention pada seluruh data...")
print("  Data latih     :", Xtr.shape[0])
print("  Data validasi  :", Xval.shape[0], "(untuk early stopping saja)")

final_model = build_model(X_selected.shape[1], arch='bilstm_attn')
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

history = final_model.fit(
    Xtr3, ytr,
    validation_data=(Xval3, yval),
    epochs=100, batch_size=64,
    class_weight=class_weights,
    callbacks=[es], verbose=1,
)

final_acc = final_model.evaluate(Xval3, yval, verbose=0)[1]
print(f"\nAkurasi model final pada validation set: {final_acc:.4f}")

# --- 2. Simpan model (format .keras native, direkomendasikan TF terbaru) ---
model_path = os.path.join(SAVE_DIR, 'ga_bilstm_attention_model.keras')
final_model.save(model_path)
print("✅ Model tersimpan   ->", model_path)

# --- 3. Simpan scaler (WAJIB dipakai ulang persis sama untuk data baru) ---
scaler_path = os.path.join(SAVE_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print("✅ Scaler tersimpan  ->", scaler_path)

# --- 4. Simpan imputer (untuk isi missing value data baru dengan cara sama) ---
imputer_path = os.path.join(SAVE_DIR, 'imputer.joblib')
joblib.dump(imp, imputer_path)
print("✅ Imputer tersimpan ->", imputer_path)

# --- 5. Simpan metadata (fitur GA, urutan kolom, dll) sebagai JSON ---
metadata = {
    'selected_features': selected_features,
    'selected_indices': [int(i) for i in selected_indices],
    'all_feature_cols': FEATURE_COLS,
    'num_cols_for_imputer': num_cols,
    'n_features_selected': len(selected_features),
    'ga_fitness': float(best_fit_g),
    'final_val_accuracy': float(final_acc),
    'architecture': 'GA-BiLSTM-Attention (feature-level attention, single-timestep)',
    'label_mapping': {'TEPAT': 1, 'TERLAMBAT': 0},
    'seed': SEED,
}
metadata_path = os.path.join(SAVE_DIR, 'metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("✅ Metadata tersimpan ->", metadata_path)

print()
print("=" * 60)
print("  SEMUA KOMPONEN TERSIMPAN DI FOLDER:", SAVE_DIR)
print("=" * 60)
print("  - ga_bilstm_attention_model.keras  (model terlatih)")
print("  - scaler.joblib                    (MinMaxScaler)")
print("  - imputer.joblib                   (SimpleImputer)")
print("  - metadata.json                    (fitur GA & info lain)")
print()
print("✅ Tersimpan PERMANEN di Google Drive - tidak hilang meski runtime")
print("   di-restart atau sesi Colab ditutup.")
print()
print("   Lokasi lengkap di Drive-mu:")
print("  ", SAVE_DIR)
print()
print("   Sesi Colab BERIKUTNYA cukup jalankan ulang blok mount Drive +")
print("   Cell 10 di bawah - TIDAK PERLU upload file atau training ulang.")

# %% [markdown]
# ## Cell 10: Contoh Cara Memuat Ulang & Pakai untuk Prediksi Data Baru
# Jalankan sel ini di sesi/notebook BARU (tidak perlu ulang training) untuk
# memprediksi data mahasiswa baru menggunakan model yang sudah disimpan.

# %%
def load_model_and_predict(new_student_df, save_dir=SAVE_DIR):
    """
    Muat model + preprocessing tersimpan, lalu prediksi data mahasiswa baru.

    Parameter
    ---------
    new_student_df : pd.DataFrame
        Harus punya kolom yang sama dengan dataset asli SEBELUM feature
        engineering (JENIS KELAMIN, STATUS MAHASISWA, UMUR, IPS 1-8, IPK).
        Feature engineering (Rata2_IPS, Delta_IPS, dst) akan dihitung ulang
        otomatis di dalam fungsi ini, memakai definisi yang sama dengan
        training.

    Return
    ------
    dict berisi 'prediksi' (label TEPAT/TERLAMBAT) dan 'probabilitas'
    """
    model = tf.keras.models.load_model(os.path.join(save_dir, 'ga_bilstm_attention_model.keras'))
    scaler_loaded  = joblib.load(os.path.join(save_dir, 'scaler.joblib'))
    imputer_loaded = joblib.load(os.path.join(save_dir, 'imputer.joblib'))
    with open(os.path.join(save_dir, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)

    d = new_student_df.copy()
    d.columns = [c.strip() for c in d.columns]

    # imputasi & feature engineering PERSIS sama seperti saat training
    d[meta['num_cols_for_imputer']] = imputer_loaded.transform(d[meta['num_cols_for_imputer']])
    d['JENIS KELAMIN']    = (d['JENIS KELAMIN'] == 'PEREMPUAN').astype(int)
    d['STATUS MAHASISWA'] = (d['STATUS MAHASISWA'] == 'BEKERJA').astype(int)

    ips = ['IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8']
    d['Rata2_IPS']    = d[ips].mean(axis=1).round(3)
    d['Std_IPS']      = d[ips].std(axis=1).round(3)
    d['IPS_Terbaik']  = d[ips].max(axis=1)
    d['IPS_Terburuk'] = d[ips].min(axis=1)
    d['Tren_IPS']     = (d['IPS 8'] - d['IPS 1']).round(3)
    d['Delta_IPS_12'] = (d['IPS 2'] - d['IPS 1']).round(3)
    d['Delta_IPS_23'] = (d['IPS 3'] - d['IPS 2']).round(3)
    d['Delta_IPS_34'] = (d['IPS 4'] - d['IPS 3']).round(3)
    d['Delta_IPS_45'] = (d['IPS 5'] - d['IPS 4']).round(3)
    d['Delta_IPS_56'] = (d['IPS 6'] - d['IPS 5']).round(3)
    d['Delta_IPS_67'] = (d['IPS 7'] - d['IPS 6']).round(3)
    d['Delta_IPS_78'] = (d['IPS 8'] - d['IPS 7']).round(3)
    d['Gap_IPK_IPS_Akhir'] = (d['IPK'] - d['IPS 8']).round(3)
    d['Gap_IPK_Rata2']     = (d['IPK'] - d['Rata2_IPS']).round(3)

    X_new_all = scaler_loaded.transform(d[meta['all_feature_cols']].values)
    X_new_sel = X_new_all[:, meta['selected_indices']]
    X_new_3d  = X_new_sel.reshape(-1, 1, len(meta['selected_indices']))

    proba = model.predict(X_new_3d, verbose=0)
    pred_class = np.argmax(proba, axis=1)
    label_rev = {v: k for k, v in meta['label_mapping'].items()}

    return {
        'prediksi': [label_rev[p] for p in pred_class],
        'probabilitas_tepat': proba[:, meta['label_mapping']['TEPAT']].tolist(),
    }

print("Fungsi load_model_and_predict() siap dipakai.")
print("Contoh: hasil = load_model_and_predict(df_mahasiswa_baru)")

