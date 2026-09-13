import pandas as pd
import numpy as np
import tensorflow as tf
import gc, time, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats

# ---------- Load & preprocessing (identik dengan pipeline utama) ----------
df_data  = pd.read_excel('Kelulusan_Data_Um_Ad_3.xls', engine='xlrd')
df_train = pd.read_excel('Kelulusan_Train.xls', engine='xlrd')
df_test  = pd.read_excel('Kelulusan_Test.xls', engine='xlrd')
df = pd.concat([df_data, df_train, df_test], ignore_index=True)
df.columns = [c.strip() for c in df.columns]
df = df.drop(columns=['NAMA', 'STATUS NIKAH'])

num_cols = ['IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK']
imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])
df['JENIS KELAMIN'] = (df['JENIS KELAMIN'] == 'PEREMPUAN').astype(int)
df['STATUS MAHASISWA'] = (df['STATUS MAHASISWA'] == 'BEKERJA').astype(int)
df['LABEL'] = (df['STATUS KELULUSAN'] == 'TEPAT').astype(int)

ips = ['IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8']
df['Rata2_IPS']=df[ips].mean(axis=1).round(3); df['Std_IPS']=df[ips].std(axis=1).round(3)
df['IPS_Terbaik']=df[ips].max(axis=1); df['IPS_Terburuk']=df[ips].min(axis=1)
df['Tren_IPS']=(df['IPS 8']-df['IPS 1']).round(3)
df['Delta_IPS_12']=(df['IPS 2']-df['IPS 1']).round(3); df['Delta_IPS_23']=(df['IPS 3']-df['IPS 2']).round(3)
df['Delta_IPS_34']=(df['IPS 4']-df['IPS 3']).round(3); df['Delta_IPS_45']=(df['IPS 5']-df['IPS 4']).round(3)
df['Delta_IPS_56']=(df['IPS 6']-df['IPS 5']).round(3); df['Delta_IPS_67']=(df['IPS 7']-df['IPS 6']).round(3)
df['Delta_IPS_78']=(df['IPS 8']-df['IPS 7']).round(3)
df['Gap_IPK_IPS_Akhir']=(df['IPK']-df['IPS 8']).round(3); df['Gap_IPK_Rata2']=(df['IPK']-df['Rata2_IPS']).round(3)

FEATURE_COLS = ['JENIS KELAMIN','STATUS MAHASISWA','UMUR','IPS 1','IPS 2','IPS 3','IPS 4','IPS 5','IPS 6','IPS 7','IPS 8','IPK',
    'Rata2_IPS','Std_IPS','IPS_Terbaik','IPS_Terburuk','Tren_IPS','Delta_IPS_12','Delta_IPS_23','Delta_IPS_34',
    'Delta_IPS_45','Delta_IPS_56','Delta_IPS_67','Delta_IPS_78','Gap_IPK_IPS_Akhir','Gap_IPK_Rata2']

scaler = MinMaxScaler()
X_all = scaler.fit_transform(df[FEATURE_COLS].values)
y_all = df['LABEL'].values.astype(int)

selected_features = ['STATUS MAHASISWA', 'UMUR', 'IPS 1', 'IPS 2', 'IPS 3', 'IPS 5', 'IPS 7', 'IPS 8',
                      'Std_IPS', 'IPS_Terbaik', 'IPS_Terburuk', 'Delta_IPS_12', 'Delta_IPS_23',
                      'Delta_IPS_34', 'Delta_IPS_56', 'Delta_IPS_67', 'Gap_IPK_IPS_Akhir', 'Gap_IPK_Rata2']
idx = [FEATURE_COLS.index(f) for f in selected_features]
X_selected = X_all[:, idx]

N_CLASSES = 2

def build_model(n_feat, has_attn):
    tf.keras.backend.clear_session()
    inp = tf.keras.Input(shape=(1, n_feat))
    if has_attn:
        flat = tf.keras.layers.Flatten()(inp)
        scores = tf.keras.layers.Dense(n_feat, activation='tanh')(flat)
        attn = tf.keras.layers.Dense(n_feat, activation='softmax')(scores)
        weighted = tf.keras.layers.Multiply()([flat, attn])
        weighted = tf.keras.layers.Reshape((1, n_feat))(weighted)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(weighted)
    else:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

VARIANTS = {
    'BiLSTM':               (X_all, False),
    'GA-BiLSTM':            (X_selected, False),
    'BiLSTM-Attention':     (X_all, True),
    'GA-BiLSTM-Attention':  (X_selected, True),
}

N_REPEATS = 3
N_FOLDS = 5
results = {name: [] for name in VARIANTS}

t0 = time.time()
for repeat in range(N_REPEATS):
    seed = [42, 123, 456][repeat]
    np.random.seed(seed); tf.random.set_seed(seed)
    for name, (X, has_attn) in VARIANTS.items():
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y_all)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y_all[tr], y_all[te]
            nf = X.shape[1]
            Xtr3 = Xtr.reshape(-1, 1, nf); Xte3 = Xte.reshape(-1, 1, nf)
            cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
            class_weight = dict(zip(np.unique(ytr), cw))
            m = build_model(nf, has_attn)
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            m.fit(Xtr3, ytr, epochs=50, batch_size=64, class_weight=class_weight,
                  validation_split=0.1, callbacks=[es], verbose=0)
            yp = np.argmax(m.predict(Xte3, verbose=0), axis=1)
            acc = accuracy_score(yte, yp)
            results[name].append(acc)
            del m; gc.collect(); tf.keras.backend.clear_session()
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s] repeat={repeat+1}/3 fold={fold+1}/5 {name:<22} acc={acc:.4f}", flush=True)

print("\n=== RINGKASAN (n=15 per varian) ===")
for name, accs in results.items():
    print(f"{name:<22} mean={np.mean(accs):.4f} std={np.std(accs):.4f} n={len(accs)}")

print("\n=== UJI SIGNIFIKANSI (paired t-test, n=15) ===")
ga = np.array(results['GA-BiLSTM']); bilstm = np.array(results['BiLSTM'])
attn = np.array(results['BiLSTM-Attention']); ga_attn = np.array(results['GA-BiLSTM-Attention'])

comparisons = [
    ("Efek GA saja", ga, bilstm),
    ("Efek Attention saja", attn, bilstm),
    ("Efek Attention | GA sudah ada", ga_attn, ga),
    ("Efek GA | Attention sudah ada", ga_attn, attn),
]
for label, a, b in comparisons:
    t, p = stats.ttest_rel(a, b)
    print(f"{label:<32} selisih={a.mean()-b.mean():+.4f} t={t:.3f} p={p:.4f} {'Ya' if p<0.05 else 'Tidak'}")

with open('multirun_ablation_results.json', 'w') as f:
    json.dump({k: v for k, v in results.items()}, f, indent=2)
print("\nSelesai. Tersimpan -> multirun_ablation_results.json")
