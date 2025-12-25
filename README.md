# 📊 Customer Churn Prediction System

<img width="435" height="270" alt="image" src="https://github.com/user-attachments/assets/f738bf29-100a-4954-b4a0-efaae185f060" />


---

## 📚 **Deskripsi Project** 📚

Proyek ini bertujuan untuk **mengembangkan sistem prediksi churn pelanggan** pada layanan subscription. Sistem ini menggunakan beberapa **Multiple AI Models** seperti **TabNet**, **FT-Transformer**, dan **MLP** untuk memprediksi apakah pelanggan akan berhenti menggunakan layanan (churn) atau tidak.

### **Latar Belakang**

Churn pelanggan adalah salah satu tantangan terbesar bagi perusahaan layanan subscription. Mengidentifikasi pelanggan yang berisiko churn memungkinkan perusahaan untuk mengambil langkah preventif, meningkatkan retensi, dan mengurangi kerugian finansial. Sistem ini mengandalkan beberapa model pembelajaran mesin untuk melakukan prediksi churn berdasarkan berbagai faktor seperti demografi, pola penggunaan, data transaksi, dan lainnya.

### **Tujuan Pengembangan**

* **Membangun Model Prediksi Churn**: Menggunakan beberapa model AI untuk memprediksi apakah pelanggan akan churn atau tidak.
* **Evaluasi Performa Model**: Menguji beberapa model AI (TabNet, FT-Transformer, MLP) untuk mendapatkan hasil prediksi terbaik.
* **Membangun Sistem Web**: Menggunakan Streamlit untuk memudahkan pengguna dalam memprediksi churn pelanggan dengan antarmuka yang mudah digunakan.

---

## 📊 **Sumber Dataset** 📊

Dataset yang digunakan dalam proyek ini berisi informasi tentang pelanggan dan status churn mereka. Sumber dataset ini berasal dari file CSV yang diunggah dan memuat berbagai fitur pelanggan yang relevan untuk prediksi churn.

---

## 🧾 **Deskripsi Dataset** 🧾

**Customer Churn Prediction Business Dataset (https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)** adalah dataset sintetik yang dirancang untuk pemodelan churn pelanggan, dengan data yang mencakup demografi pelanggan, pola penggunaan produk, riwayat penagihan dan pembayaran, interaksi dukungan pelanggan, serta metrik keterlibatan pelanggan.

### **Karakteristik Dataset**

* **Jumlah rekaman**: 10.000 pelanggan
* **Variabel target**: churn (0 = Tidak, 1 = Ya)
* **Tipe data**: Numerik & Kategorikal
* **Domain**: Bisnis Subscription / SaaS / Telekomunikasi / Layanan
* **Sumber data**: Sintetik (berbasis logika bisnis)

### **Kategori Fitur**

1. **Profil Pelanggan**: usia, jenis kelamin, lokasi, masa berlangganan, tipe kontrak
2. **Penggunaan Produk**: login, durasi sesi, penggunaan fitur, tren aktivitas
3. **Penagihan & Pembayaran**: biaya langganan, pendapatan, kegagalan pembayaran, diskon
4. **Dukungan Pelanggan**: tiket, waktu penyelesaian, CSAT, keluhan
5. **Keterlibatan & Umpan Balik**: aktivitas email, skor NPS, respon survei

### **Kasus Penggunaan**

* Memprediksi pelanggan dengan risiko churn tinggi
* Mengidentifikasi faktor-faktor utama churn
* Mengestimasi pendapatan yang berisiko
* Membangun strategi retensi pelanggan
* Melatih dan mengevaluasi model Machine Learning / Deep Learning
* Membuat dashboard bisnis tingkat eksekutif

### **Disclaimer**

Dataset ini dibuat secara sintetik untuk tujuan pendidikan, penelitian, dan portofolio. Meskipun mencerminkan pola bisnis yang realistis, dataset ini tidak mewakili data pelanggan nyata.

---

## 🧑‍💻 **Preprocessing dan Pemodelan** 🧑‍💻

### **Pemilihan Kolom/Atribut**

Proses preprocessing meliputi pengolahan data yang dibagi menjadi fitur numerik dan kategorikal. Fitur yang digunakan untuk memprediksi churn pelanggan diambil dari kolom-kolom yang relevan seperti usia, jenis kelamin, segmen pelanggan, dan lainnya.

### **Preprocessing Data**

1. **Transformasi Data**: Kolom-kolom kategorikal diubah menjadi numerik menggunakan **Label Encoding**. Fitur numerik lainnya dinormalisasi menggunakan **Standard Scaler**.
2. **Pembagian Data**: Data dibagi menjadi **80%** untuk pelatihan dan **20%** untuk pengujian.

### **Pemodelan**

Tiga model yang digunakan untuk klasifikasi churn pelanggan adalah:

1. **TabNet**: Model berbasis attention yang sangat efektif untuk data tabular dan menyediakan interpretasi fitur.
2. **FT-Transformer**: Model Transformer yang dapat menangkap hubungan fitur yang lebih kompleks.
3. **MLP (Multilayer Perceptron)**: Model neural network standar yang cocok untuk banyak tugas klasifikasi.

---

## 🔧 **Langkah Instalasi** 🔧

### **Software Utama**

Proyek ini dapat dijalankan menggunakan Google Colab atau IDE lain seperti VSCode. Pastikan Python 3.x terinstal di sistem Anda.

### **Dependensi**

Dependensi yang dibutuhkan tercantum di dalam file `requirements.txt`. Anda dapat menginstal seluruh dependensi dengan perintah berikut:

```bash
pip install -r requirements.txt
```

### **Menjalankan Sistem Prediksi**

Untuk menjalankan aplikasi Streamlit, gunakan perintah berikut:

```bash
streamlit run app.py
```

---

## 💡 **Pelatihan Model** 💡

Model yang telah dilatih tersedia di direktori Model. Namun, model **TabNet** dan **FT-Transformer** yang sudah dilatih dapat diunduh melalui Google Drive. Anda dapat mengunduhnya untuk digunakan dalam aplikasi ini.

Jika Anda ingin melatih model dari awal, jalankan file Notebook yang tersedia di direktori ini menggunakan Google Colab.

---

## 🔍 **Hasil dan Analisis** 🔍

### **Evaluasi Model**

Berikut adalah tabel evaluasi model untuk **MLP**, **TabNet**, dan **FT-Transformer** berdasarkan hasil **Classification Report** yang Anda berikan:

| **Model**                | **Accuracy** | **Hasil Analysis** |
| ------------------------ | ------------ | ------------------ |
| **MLP (Neural Network)** | **90%**      | Meskipun MLP berhasil mencapai accuracy yang tinggi, namun performanya dalam mendeteksi churn sangat buruk. Precision dan Recall untuk kategori "Churn" sangat rendah (0.00), yang menunjukkan bahwa model ini tidak mampu mendeteksi pelanggan yang churn dengan baik. Performa model ini tidak seimbang antara kedua kategori, yaitu churn dan no churn. |
| **TabNet**               | **90%**      | TabNet menunjukkan hasil yang lebih baik daripada MLP pada kategori "No Churn", dengan Precision 0.91 dan Recall 0.99. Namun, model ini masih mengalami kesulitan dalam mengklasifikasikan pelanggan yang churn, dengan Precision dan Recall yang lebih rendah pada kategori "Churn" (0.42 dan 0.09). Meskipun memiliki accuracy yang tinggi, performanya dalam mendeteksi churn masih terbatas. |
| **FT-Transformer**       | **90%**      | FT-Transformer menunjukkan hasil yang sangat baik pada kategori "No Churn" dengan Precision 0.90 dan Recall 1.00. Namun, model ini memiliki kinerja yang sangat buruk dalam mendeteksi churn (0.00), meskipun mencapai accuracy yang tinggi. Model ini sangat terbatas dalam mendeteksi churn secara efektif. |


**Keterangan:**

* **Precision**: Mengukur akurasi prediksi positif (contoh: pelanggan churn yang diprediksi churn).
* **Recall**: Mengukur seberapa baik model menemukan prediksi positif yang sebenarnya.
* **F1-Score**: Rata-rata harmonis antara precision dan recall.
* **Accuracy**: Proporsi dari total prediksi yang benar (jumlah benar / total).

Berdasarkan hasil tersebut, meskipun model **MLP** dan **FT-Transformer** menunjukkan hasil yang kurang memadai untuk kategori churn (dengan precision dan recall 0), model **TabNet** berhasil menunjukkan kinerja yang lebih baik pada kategori "No Churn", meskipun recall-nya untuk kategori "Churn" masih rendah.

### **Confusion Matrix**

Confusion Matrix untuk masing-masing model memberikan gambaran mengenai bagaimana model mengklasifikasikan churn dan non-churn pelanggan.
<img width="658" height="547" alt="image" src="https://github.com/user-attachments/assets/eca5a428-b1cb-4ea4-b02c-94467f983eec" />
<img width="658" height="547" alt="image" src="https://github.com/user-attachments/assets/02081f2f-a611-4b14-8b84-2f4a2c03b200" />
<img width="658" height="547" alt="image" src="https://github.com/user-attachments/assets/83684948-7e1a-4f06-a600-a5a623ee86a5" />

---

## 🎓 **Sistem Sederhana Streamlit** 🎓

Aplikasi berbasis **Streamlit** memungkinkan pengguna untuk memprediksi churn pelanggan dengan mudah melalui antarmuka web. Berikut adalah tampilan antarmuka aplikasi:

* **Form Input**: Pengguna dapat memasukkan informasi pelanggan untuk memprediksi kemungkinan churn.
* **Prediksi**: Setelah mengisi form, pengguna dapat menekan tombol untuk mendapatkan prediksi churn pelanggan.
* **Hasil**: Prediksi churn ditampilkan dengan tingkat probabilitas dan rekomendasi untuk tindakan lebih lanjut.

---
### Link Live Demo

Coba aplikasi prediksi kesuksesan akademik mahasiswa secara langsung dengan mengunjungi tautan di bawah ini:

[🔗 **Demo Aplikasi Sederhana Streamlit**](https://iftitahyr-uap-ml-258-srcapp-gnyoyt.streamlit.app/)

---

## 👤 **Biodata**

👤 **Nama**: Iftitah Yanuar Rahmawati

🎓 **Program Studi**: Teknik Informatika

🏛️ **Universitas**: Universitas Muhammadiyah Malang

📧 **Email**: [iftitahyanuar@webmail.umm.ac.id](mailto:iftitahyanuar@webmail.umm.ac.id)

---

README ini dirancang untuk memberikan pemahaman yang lengkap tentang proyek **Customer Churn Prediction System**. Jika ada penyesuaian lebih lanjut atau tambahan yang diinginkan, Anda bisa memberi tahu saya!
