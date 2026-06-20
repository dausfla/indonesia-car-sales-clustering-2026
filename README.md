# Segmentasi Pasar Konsumen Otomotif Indonesia (Rekap 2026) Menggunakan Algoritma K-Means Clustering

Proyek ini merupakan sebuah implementasi *Data Science* dan *Machine Learning* (*Unsupervised Learning*) untuk menganalisis karakteristik konsumen serta melakukan segmentasi pasar berdasarkan data rekapitulasi penjualan 7 model mobil terpopuler di Indonesia sepanjang tahun 2026. Proyek ini dirancang untuk memenuhi kebutuhan analisis bisnis (analisis deskriptif) dan pemodelan prediktif/pengelompokan ilmiah (analisis preskriptif).

## 🚀 Fitur Utama
1. **Otomatisasi Generasi Data Rekap 2026**: Menghasilkan 5.000 data transaksi fiktif terstruktur yang mensimulasikan kondisi riil pasar otomotif Indonesia berdasarkan distribusi statistik normal.
2. **Dashboard Summary (Pivot Tables)**: Menyajikan data agregasi distribusi kendaraan per wilayah dan profil finansial rata-rata (umur & pendapatan) untuk setiap lini mobil secara real-time.
3. **Penerapan Algoritma K-Means**: Mengelompokkan konsumen berdasarkan tingkat pendapatan bulanan dan rentang usia untuk memetakan target pasar yang potensial.
4. **Optimasi Cluster via Elbow Method**: Menggunakan perhitungan *Within-Cluster Sum of Squares* (WCSS) secara matematis untuk menemukan jumlah cluster paling efisien.
5. **Auto-Export Artefak**: Otomatis menyimpan hasil olahan data ke dalam format `.csv` dan visualisasi dashboard grafik ke format `.png` beresolusi tinggi (300 DPI).

---

## 📊 Struktur Variabel Data
Dataset rekapitulasi penjualan ini memiliki 5.000 baris dengan kolom sebagai berikut:
- `No_Faktur`: Kode invoice unik dengan format formal tahun berjalan (`INV/2026/XXXXX`).
- `Bulan_Pembelian`: Representasi waktu transaksi (Januari - Desember 2026).
- `Nama_Mobil`: Model kendaraan (Toyota Avanza, Mitsubishi Xpander, Toyota Innova Zenix, Honda Brio, Daihatsu Sigra, Honda HR-V, Toyota Rush).
- `Wilayah`: Lokasi pemasaran utama (DKI Jakarta, Jawa Barat, Jawa Timur, Sumatera Utara, Sulawesi Selatan, Jawa Tengah).
- `Gender_Pembeli`: Demografi jenis kelamin pembeli (Pria / Wanita).
- `Pekerjaan_Pembeli`: Latar belakang profesi pembeli (Karyawan Swasta, Pengusaha, PNS, Professional, BUMN, Rumah Tangga).
- `Umur_Pembeli`: Rentang usia produktif hingga senior (22 – 65 tahun).
- `Pendapatan_Bulanan_Juta`: Finansial bulanan konsumen dalam satuan Juta Rupiah (Distribusi normal terikat nilai minimum regional).

---

## 🛠️ Prasyarat & Pemasangan (*Tech Stack*)
Aplikasi ini dikembangkan menggunakan bahasa pemrograman **Python 3.13** dengan pustaka-pustaka data science berikut:
- **Pandas**: Pengolahan data tabular dan pembuatan Pivot Table.
- **NumPy**: Komputasi numerik dan pembuatan data berbasis distribusi acak terstruktur.
- **Matplotlib & Seaborn**: Pembuatan grafik dan dashboard visualisasi data statistik.
- **Scikit-Learn**: Penerapan penskalaan fitur (`StandardScaler`) dan algoritma `KMeans`.

### Langkah Instalasi Library:
Pastikan Python sudah terdaftar dalam sistem environment variables (PATH) Anda, lalu jalankan perintah berikut di terminal:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
