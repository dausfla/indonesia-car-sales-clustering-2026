import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# STEP 1: GENERATE DATA REKAP TAHUN 2026 (5.000 DATA)
# ==========================================
print("Step 1: Menggenerasi Data Rekapitulasi Tahun 2026...")
np.random.seed(42)  
random.seed(42)

n_samples = 5000

# Variabel tren otomotif Indonesia 2026
mobil_list = ["Toyota Avanza", "Mitsubishi Xpander", "Toyota Innova Zenix", "Honda Brio", "Daihatsu Sigra", "Honda HR-V", "Toyota Rush"]
wilayah_list = ["DKI Jakarta", "Jawa Barat (Bandung)", "Jawa Timur (Surabaya)", "Sumatera Utara (Medan)", "Sulawesi Selatan (Makassar)", "Jawa Tengah (Semarang)"]
pekerjaan_list = ["Karyawan Swasta", "Pengusaha", "PNS", "Professional (Dokter/Pengacara)", "BUMN", "Rumah Tangga"]
gender_list = ["Pria", "Wanita"]
bulan_list = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

data = {
    "No_Faktur": [f"INV/2026/{i:05d}" for i in range(1, n_samples + 1)],
    "Bulan_Pembelian": [random.choice(bulan_list) for _ in range(n_samples)],
    "Nama_Mobil": [random.choice(mobil_list) for _ in range(n_samples)],
    "Wilayah": [random.choice(wilayah_list) for _ in range(n_samples)],
    "Gender_Pembeli": [random.choice(gender_list) for _ in range(n_samples)],
    "Pekerjaan_Pembeli": [random.choice(pekerjaan_list) for _ in range(n_samples)],
    "Umur_Pembeli": np.random.randint(22, 65, size=n_samples),
    "Pendapatan_Bulanan_Juta": np.round(np.random.normal(loc=18, scale=8, size=n_samples), 1),
}

df = pd.DataFrame(data)
df["Pendapatan_Bulanan_Juta"] = df["Pendapatan_Bulanan_Juta"].clip(lower=4)

print(f"-> Sukses membuat {len(df)} baris data rekap tahun 2026.")

# ==========================================
# STEP 2: EKSPOR DATA KE CSV
# ==========================================
print("\nStep 2: Mengekspor Data ke CSV...")
nama_file_csv = "data_mobil_populer_indonesia.csv"
df.to_csv(nama_file_csv, index=False)
print(f"-> Sukses! File tersimpan dengan nama: {nama_file_csv}")

# ==========================================
# STEP 3: OLAH DATA & PIVOT TABLE
# ==========================================
print("\nStep 3: Membuat Dashboard Summary (Pivot Table)...")
print("\n=== DISTRIBUSI PENJUALAN MOBIL PER WILAYAH DI TAHUN 2026 ===")
pivot_wilayah_mobil = df.pivot_table(
    index="Wilayah", columns="Nama_Mobil", values="No_Faktur", aggfunc="count", fill_value=0
)
print(pivot_wilayah_mobil)

print("\n=== RATA-RATA UMUR DAN PENDAPATAN PEMBELI MOBIL 2026 ===")
pivot_pendapatan = df.pivot_table(index="Nama_Mobil", values=["Umur_Pembeli", "Pendapatan_Bulanan_Juta"], aggfunc="mean")
print(pivot_pendapatan.round(2))

# ==========================================
# STEP 4: PREPARASI CLUSTERING K-MEANS
# ==========================================
print("\nStep 4: Menghitung Algoritma K-Means untuk Grafik Elbow...")
X = df[["Umur_Pembeli", "Pendapatan_Bulanan_Juta"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
print("-> Perhitungan WCSS untuk Elbow Method selesai.")

# ==========================================
# STEP 5: VISUALISASI DAN PENYIMPANAN GRAFIK
# ==========================================
print("\nStep 5: Membuat dan Menyimpan Grafik Visualisasi...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# Plot 1: Total Penjualan Mobil 2026
sns.countplot(data=df, x="Nama_Mobil", order=df["Nama_Mobil"].value_counts().index, ax=axes[0], palette="coolwarm")
axes[0].set_title("Grafik Total Penjualan Mobil Terpopuler di Indonesia (Rekap Tahunan 2026)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Model Mobil")
axes[0].set_ylabel("Unit Terjual")
axes[0].tick_params(axis="x", rotation=15)

# Plot 2: Grafik Elbow Method
axes[1].plot(k_range, wcss, marker="o", linestyle="--", color="b", linewidth=2)
axes[1].set_title("Grafik Elbow Method (Penentuan Cluster Konsumen Optimal 2026)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Jumlah Cluster (k)")
axes[1].set_ylabel("WCSS (Inertia)")
axes[1].set_xticks(k_range)

plt.tight_layout()

# JALAN NINJA: Simpan grafik langsung jadi file gambar biar aman
nama_gambar = "dashboard_rekap_2026.png"
plt.savefig(nama_gambar, dpi=300)
print(f"-> Sukses! Grafik telah disimpan sebagai gambar: {os.path.abspath(nama_gambar)}")

print("\nMenampilkan grafik di layar (jika sistem mendukung)...")
plt.show()
print("Proses selesai 100%!")