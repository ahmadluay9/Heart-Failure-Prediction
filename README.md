[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/bErXH9jA)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10487164&assignment_repo_type=AssignmentRepo)
# Graded Challenge 3

_Graded Challenge ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Fulltime Program khususnya pada konsep Ensemble Learning._

---

## Assignment Objectives

*Graded Challenge 3* ini dibuat guna mengevaluasi konsep Ensemble Learning sebagai berikut:

- Mampu memperoleh data menggunakan BigQuery

- Mampu memahami konsep Classification dengan Ensemble Learning

- Mampu mengimplementasikan Ensemble Learning untuk membuat prediksi

- Mampu memahami konsep salah satu algoritma boosting dan mengimplementasikannya dalam kasus Supervised Learning

---

## Dataset

```{attention}
Perhatikan petunjuk penggunaan dataset!
```

1. Pada tugas kali ini, dataset yang digunakan **tidak akan menggunakan `bigquery-public-data`**. 

2. Masuk ke dalam Google BigQuery. Gunakan informasi dibawah ini sebagai tempat untuk mengambil data (gunakan sebagai informasi untuk klausa `FROM`).
   * Project ID : `ftds-hacktiv8-project`
   
   * Dataset Name : 
     + Batch offline : `phase1_ftds_<nomor-batch>_hck` contoh `phase1_ftds_001_hck`
     
     + Batch online : `phase1_ftds_<nomor-batch>_rmt` contoh `phase1_ftds_001_rmt`
   
   * Table Name : `heart-failure`

3. Ambil semua data dari table nomor 2.

4. Berikut ini adalah informasi dari setiap column. 
   <img src='https://i.ibb.co/YBGwMXm/P1-G3-Dataset-Information.png'>

5. Simpan dataset dalam bentuk `.csv` dengan format `h8dsft_P1G3_<nama-students>.csv`. Misal : `h8dsft_P1G3_raka_ardhi.csv`.

6. Salin query yang telah dibuat di Google Cloud Platform. Tulislah pada bagian atas notebook !

7. Tampilkan `10 data pertama` dan `10 data terakhir` dari dataset pada notebook !

---

## Problems

Buatlah model Classification menggunakan Random Forest dan salah satu algoritma boosting untuk memprediksi apakah seorang pasien akan meninggal atau tidak menggunakan dataset yang sudah Anda dapatkan.

---

## Conceptual Problems

*Jawab pertanyaan berikut:*

1. Jelaskan latar belakang adanya bagging dan cara kerja bagging !

2. Jelaskan perbedaan cara kerja algoritma Random Forest dengan algoritma boosting yang Anda pilih !

---

## Assignment Instructions

*Graded Challenge 3* dikerjakan dalam format ***notebook*** dengen beberapa **kriteria wajib** di bawah ini:

1. Machine learning framework yang digunakan adalah *Scikit-Learn*.

2. Ada penggunaan library visualisasi, seperti *matplotlib*, *seaborn*, atau yang lain.

3. Isi *notebook* harus mengikuti *outline* di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, gambaran besar dataset yang digunakan, dan *objective* yang ingin dicapai.

   2. Query SQL
      > Tulis query yang telah dibuat untuk mengambil data dari Google Cloud Platform di bagian ini.

   3. Import Libraries
      > *Cell* pertama pada *notebook* **harus berisi dan hanya berisi** semua *library* yang digunakan dalam *project*.
   
   4. Data Loading
      > Bagian ini berisi proses penyiapan data sebelum dilakukan eksplorasi data lebih lanjut. Proses Data Loading dapat berupa memberi nama baru untuk setiap kolom, mengecek ukuran dataset, dll.
   
   5. Exploratory Data Analysis (EDA)
      > Bagian ini berisi eksplorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.
   
   6. Feature Engineering
      > Bagian ini berisi proses penyiapan data untuk proses pelatihan model, seperti pembagian data menjadi train-test, transformasi data (normalisasi, encoding, dll.), dan proses-proses lain yang dibutuhkan.
   
   7. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

   8. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.
   
   9. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis terkait dengan hasil pada model dan tuliskan hasil analisisnya**.
   
   10. Model Saving
       > Pada bagian ini, dilakukan proses penyimpanan model dan file-file lain yang terkait dengan hasil proses pembuatan model.

   11. Model Inference
       > Model yang sudah dilatih akan dicoba pada data yang bukan termasuk ke dalam train-set ataupun test-set. Data ini harus dalam format yang asli, bukan data yang sudah di-scaled.
   
   12. Pengambilan Kesimpulan
       > Pada bagian terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan *objective* yang sudah ditulis di bagian pengenalan.
    
4. *Notebook* harus diupload dalam akun GitHub masing-masing student untuk selanjutnya dinilai.

---

## Assignment Submission

- Simpan assignment pada sesi ini dengan nama `h8dsft_P1G3_<nama-student>.ipynb` misal `h8dsft_P1G3_raka_ardhi.ipynb`.

- Push assignment yang telah Anda buat ke akun Github Classroom Anda masing-masing.

- Untuk Model Deployment :
  * Buat sebuah folder bernama `deployment` dan masukkan semua file yang berkaitan dengan deployment ke folder ini.
  * Buat sebuah file bernama `url.txt` yang berisi URL deployment.
  * Contoh bentuk isi repository dengan Model Deployment.
    ```
    ├── deployment/
    │   ├── app.py
    │   └── eda.py
    │   └── prediction.py
    │   └── model.pkl
    ├── h8dsft_P1M2_raka_ardhi.ipynb
    ├── url.txt
    └── README.md
    ```

---

## Assignment Rubrics

### Code Review

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| SQL | Mampu melakukan query data dengan kriteria yang telah diberikan | 10 pts |
| Feature Engineering | Mampu melakukan preprocessing dataset sebelum melakukan proses modeling (split data, normalisasi, encoding, dll) | 35 pts |
| Random Forest | Mengimplementasikan Random Forest dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 10 pts |
| Boosting Algorithm | Mengimplementasikan Boosting dan menentukan hyperparameter yang tepat dengan Scikit-Learn | 10 pts |
| Model Inference | Mencoba model yang telah dibuat dengan data baru | 10 pts |
| Runs Perfectly | Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar | 10 pts |

### Concepts

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Classifications | Mampu menjawab pertanyaan dengan singkat, jelas, dan padat serta sesuai dengan konsep dan logika yang ada mengenai Conceptual Problems (10 pts each) | 20 pts |

### Readability

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Tertata Dengan Baik | Semua baris kode terdokumentasi dengan baik dengan Markdown untuk penjelasan kode | 15 pts |

```
Kriteria tertata dengan baik diantaranya adalah: 

1. Terdapat section Perkenalan yang jelas dan lengkap terkait masalah dan latar belakang masalah yang akan diselesaikan.
2. Tidak menyalin markdown dari tugas lain.
3. Import library rapih (terdapat dalam 1 cell dan tidak ada unused libs).
4. Pemakaian fungsi markdown yang optimal (Heading, text formating, dll).
5. Terdapat komentar pada setiap baris kode.
6. Adanya pemisah yang jelas antar section, dll.
7. Tidak adanya typo.
```

### Analysis

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Model Analysis | Menganalisa informasi dari model yang telah dibuat | 35 pts |
| Overall Analysis | Menarik informasi/kesimpulan dari keseluruhan kegiatan yang dilakukan | 20 pts |

```
Contoh kriteria analisa yang baik diantaranya adalah: 

1. Terdapat penjelasan macam-macam hasil metric evaluasi dan interpretasinya terhadap kasus yang diselesaikan.
2. Dapat menjelaskan KELEBIHAN dan KELEMAHAN dari model yang dibuat DENGAN KAITANNYA DENGAN DOMAIN BUSINESS YANG DIHADAPI yang dibuktikan dengan eksplorasi sederhana (grafik, plot, teori, dll).
3. Dapat memberikan statement untuk improvement selanjutnya dari model yang dibuat. 
4. Dapat menyebutkan insight yang dapat diambil setelah proses EDA, dll.
```

### Model Deployment

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Model Deployment | Membuat webapps terhadap project yang telah dibuat | 15 pts |

```
Catatan mengenai Model Deployment : 

1. Ketiadaan URL deployment ataupun source code deployment di repository, akan tetap diperhitungkan untuk menilai bagian Model Deployment. 
2. Tidak diperkenankan adanya informasi tambahan/informasi susulan seperti lupa memberikan URL deployment atau lupa mengupload source code via apapun (DM buddy, email, atau yang lain).
3. Student akan dianggap tidak melakukan Model Deployment jika tidak ada URL deployment dan source code deployment di repository.
```

---

```
Total Points : 190
```

---

## Notes

* **Deadline : P1W3D5 pukul 23:59 WIB.**

* **Keterlambatan pengumpulan tugas mengakibatkan skor GC 3 menjadi 0.**
