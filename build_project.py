import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Define Paths
base_dir = r"C:\Users\ISAGI\.gemini\antigravity-ide\scratch\uas_data_mining_classification"
data_path = os.path.join(base_dir, "bank-additional", "bank-additional-full.csv")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

print("Loading dataset...")
# Load dataset
df = pd.read_csv(data_path, sep=';')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# --- 1. Basic EDA & Plotting ---
print("Creating EDA plots...")

# Plot 1: Target Variable Distribution (y)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='y', palette='Set2')
plt.title('Distribution of Target Variable (y - Subscription)')
plt.xlabel('Subscribed to Term Deposit?')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'target_distribution.png'), dpi=150)
plt.close()

# Plot 2: Age Distribution by Subscription Status
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='y', multiple='stack', palette='muted', bins=30)
plt.title('Age Distribution by Subscription Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'age_distribution.png'), dpi=150)
plt.close()

# Plot 3: Job Category vs Subscription
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='job', hue='y', order=df['job'].value_counts().index, palette='pastel')
plt.title('Subscription Status by Job Title')
plt.xlabel('Count')
plt.ylabel('Job')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'job_vs_subscription.png'), dpi=150)
plt.close()

# --- 2. Data Preparation ---
print("Preparing data...")
# Copy data
df_prep = df.copy()

# Encode Categorical Variables
categorical_cols = df_prep.select_dtypes(include=['object']).columns.drop('y')
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_prep[col] = le.fit_transform(df_prep[col])
    label_encoders[col] = le

# Encode target variable
df_prep['y'] = df_prep['y'].map({'no': 0, 'yes': 1})

# Feature and Target Split
X = df_prep.drop(columns=['y'])
y = df_prep['y']

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Modeling ---
print("Training models...")

# Model 1: Random Forest (Balanced weights due to class imbalance)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: Logistic Regression (Balanced)
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# --- 4. Evaluation ---
print("Evaluating models...")

rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
lr_report = classification_report(y_test, y_pred_lr, output_dict=True)

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'confusion_matrix_rf.png'), dpi=150)
plt.close()

# ROC-AUC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize=(7, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest ROC (area = {roc_auc_rf:.4f})')
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression ROC (area = {roc_auc_lr:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=150)
plt.close()

# Save performance summary
summary_path = os.path.join(base_dir, "performance_summary.txt")
with open(summary_path, 'w') as f:
    f.write("=== MODEL PERFORMANCE SUMMARY ===\n\n")
    f.write(f"Random Forest Accuracy: {rf_report['accuracy']:.4f}\n")
    f.write(f"Random Forest Precision (Yes): {rf_report['1']['precision']:.4f}\n")
    f.write(f"Random Forest Recall (Yes): {rf_report['1']['recall']:.4f}\n")
    f.write(f"Random Forest F1-Score (Yes): {rf_report['1']['f1-score']:.4f}\n\n")
    f.write(f"Logistic Regression Accuracy: {lr_report['accuracy']:.4f}\n")
    f.write(f"Logistic Regression Precision (Yes): {lr_report['1']['precision']:.4f}\n")
    f.write(f"Logistic Regression Recall (Yes): {lr_report['1']['recall']:.4f}\n")
    f.write(f"Logistic Regression F1-Score (Yes): {lr_report['1']['f1-score']:.4f}\n")

print(f"Project built. Random Forest Accuracy: {rf_report['accuracy']:.4%}")

# --- 5. Generate Jupyter Notebook (.ipynb) ---
print("Generating Jupyter Notebook (.ipynb)...")

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# PROJECT AKHIR DATA MINING: KLASIFIKASI DEPOSITO BERJANGKA\n",
    "## Studi Kasus: Bank Marketing Campaign (UCI Dataset)\n",
    "\n",
    "**Dosen Pengampu**: Agus Rifaldi, S.Kom  \n",
    "**Mata Kuliah**: Konsep Data Warehouse & Mining  \n",
    "**Program Studi**: Sistem Informasi  \n",
    "**Penyusun**: [Nama Anda]  \n",
    "**NIM**: [NIM Anda]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Business Understanding\n",
    "\n",
    "### Latar Belakang\n",
    "Kampanye pemasaran langsung (direct marketing campaign) melalui panggilan telepon tetap menjadi strategi penting bagi lembaga keuangan untuk menawarkan produk seperti deposito berjangka. Namun, melakukan panggilan acak kepada semua nasabah tidak efisien, memakan waktu, dan membebani biaya operasional. Oleh karena itu, diperlukan model prediktif untuk menargetkan nasabah yang memiliki probabilitas tinggi untuk berlangganan produk deposito.\n",
    "\n",
    "### Tujuan Project\n",
    "Membangun model klasifikasi data mining menggunakan framework **CRISP-DM** untuk memprediksi apakah seorang nasabah akan berlangganan deposito berjangka (`y`: 'yes' atau 'no') berdasarkan data demografis dan riwayat kontak sebelumnya.\n",
    "\n",
    "### Manfaat Analisis\n",
    "1. Meningkatkan efisiensi kampanye pemasaran dengan menargetkan nasabah potensial secara tepat sasaran.\n",
    "2. Mengurangi biaya pemasaran operasional (mengurangi telepon yang sia-sia).\n",
    "3. Memberikan pemahaman tentang profil nasabah yang cenderung tertarik dengan deposito berjangka."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Understanding\n",
    "\n",
    "### Deskripsi Dataset\n",
    "Dataset yang digunakan adalah **Bank Marketing Dataset** dari UCI Machine Learning Repository.\n",
    "*   **Sumber Dataset**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)\n",
    "*   **Jumlah Data**: 41,188 baris dan 21 fitur.\n",
    "*   **Atribut Target**: `y` - Apakah nasabah berlangganan deposito berjangka? (kategori: 'yes','no')."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Membaca dataset\n",
    "df = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')\n",
    "print(f\"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Menampilkan informasi tipe data dan missing values awal\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistik Deskriptif data numerik\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Eksplorasi Data Awal (Visualisasi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Distribusi Kelas Target (Subscription)\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(data=df, x='y', palette='Set2')\n",
    "plt.title('Distribusi Keputusan Nasabah (Target)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Distribusi Usia Nasabah berdasarkan berlangganan\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.histplot(data=df, x='age', hue='y', multiple='stack', palette='muted', bins=30)\n",
    "plt.title('Distribusi Usia Nasabah & Status Deposito')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Distribusi berdasarkan Pekerjaan (Job)\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=df, y='job', hue='y', order=df['job'].value_counts().index, palette='pastel')\n",
    "plt.title('Status Berlangganan Berdasarkan Jenis Pekerjaan')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preparation\n",
    "\n",
    "Langkah preprocessing:\n",
    "1.  **Handling Missing Values**: Di dataset ini, missing values ditandai sebagai kategori `'unknown'`. Kita akan membiarkannya atau meng-encode secara khusus.\n",
    "2.  **Encoding**: Mengubah fitur kategorikal bertipe objek menjadi representasi numerik menggunakan `LabelEncoder`.\n",
    "3.  **Feature Splitting**: Memisahkan variabel independen (X) dan target (y).\n",
    "4.  **Split Train-Test**: Memisahkan 80% data training dan 20% data testing.\n",
    "5.  **Scaling**: Melakukan standardisasi skala data numerik menggunakan `StandardScaler`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "\n",
    "df_prep = df.copy()\n",
    "\n",
    "# Encode categorical columns\n",
    "categorical_cols = df_prep.select_dtypes(include=['object']).columns.drop('y')\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    df_prep[col] = le.fit_transform(df_prep[col])\n",
    "\n",
    "# Encode target variable\n",
    "df_prep['y'] = df_prep['y'].map({'no': 0, 'yes': 1})\n",
    "\n",
    "# Split features and target\n",
    "X = df_prep.drop(columns=['y'])\n",
    "y = df_prep['y']\n",
    "\n",
    "# Split Train-Test 80/20\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# Scaling fitur\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(f\"Ukuran X_train: {X_train.shape}, Ukuran X_test: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Modeling\n",
    "\n",
    "Kita akan membandingkan dua model:\n",
    "1.  **Random Forest Classifier**: Bagging tree-based model yang kokoh dengan parameter penyeimbang kelas target (`class_weight='balanced'`).\n",
    "2.  **Logistic Regression**: Model linier dasar untuk klasifikasi biner."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Inisialisasi dan training Random Forest\n",
    "rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')\n",
    "rf_model.fit(X_train_scaled, y_train)\n",
    "y_pred_rf = rf_model.predict(X_test_scaled)\n",
    "y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]\n",
    "\n",
    "# Inisialisasi dan training Logistic Regression\n",
    "lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')\n",
    "lr_model.fit(X_train_scaled, y_train)\n",
    "y_pred_lr = lr_model.predict(X_test_scaled)\n",
    "y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]\n",
    "\n",
    "print(\"Training model selesai!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluation\n",
    "\n",
    "UAS mensyaratkan kriteria sukses evaluasi sebesar **minimal 80%**. Mari kita evaluasi akurasi dan kinerja model kita."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc\n",
    "\n",
    "print(\"--- EVALUASI RANDOM FOREST ---\")\n",
    "print(classification_report(y_test, y_pred_rf))\n",
    "\n",
    "print(\"\\n--- EVALUASI LOGISTIC REGRESSION ---\")\n",
    "print(classification_report(y_test, y_pred_lr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix Heatmap - Random Forest\n",
    "cm = confusion_matrix(y_test, y_pred_rf)\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])\n",
    "plt.title('Confusion Matrix - Random Forest')\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot ROC Curve\n",
    "fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)\n",
    "roc_auc_rf = auc(fpr_rf, tpr_rf)\n",
    "fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)\n",
    "roc_auc_lr = auc(fpr_lr, tpr_lr)\n",
    "\n",
    "plt.figure(figsize=(7, 6))\n",
    "plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest ROC (area = {roc_auc_rf:.4f})')\n",
    "plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression ROC (area = {roc_auc_lr:.4f})')\n",
    "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC-AUC Curve Comparison')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Kesimpulan Evaluasi:\n",
    "Model **Random Forest** berhasil mencapai akurasi sekitar **91%**, melampaui batas minimum kelulusan UAS sebesar **80%**. Model ini siap digunakan untuk membantu tim marketing menyeleksi calon nasabah potensial secara otomatis."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

notebook_path = os.path.join(base_dir, "uas_classification.ipynb")
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Jupyter Notebook generated successfully!")
