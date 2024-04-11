# Laporan Proyek Machine Learning - Jordan Marcelino

## Domain Proyek

_Mental Health_ adalah istilah yang sudah tidak asing lagi untuk didengar di era modern ini. Istilah ini sering digunakan oleh kaum muda untuk membahas kondisi psikologis & emosional seseorang. Banyak hal yang dapat memengaruhi kondisi _Mental Health_ seseorang, media sosial adalah salah satunya. Maraknya penggunaan media sosial saat ini menyebabkan banyaknya orang menaruh jati dirinya pada hal tersebut, sehingga kondisi _Mental Health_ seseorang akan sangat bergantung pada media sosial. Pesan atau komen yang diberikan oleh seseorang pada media sosial dapat secara tidak langsung mencerminkan kondisi _Mental Health_ orang tersebut. Maka dari itu proyek ini dilakukan untuk mendeteksi stres dari artikel media sosial berbasis teks dari Reddit menggunakan pendekatan _machine learning_.

> [Valkenburg, P. M., Meier, A., & Beyens, I. (2022). Social media use and its impact on adolescent mental health: An umbrella review of the evidence. Current opinion in psychology, 44, 58-68.](https://www.sciencedirect.com/science/article/pii/S2352250X21001500)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:

-   Bagaimana efektivitas pendekatan _machine learning_ dalam mengklasifikan stres berdasarkan teks artikel media sosial

### Goals

-   Mengevaluasi efektivitas pendekatan _machine learning_ dalam mengklasifikan stres berdasarkan teks artikel media sosial

    ### Solution statements

    -   Melakukan improvement pada _Random Forest_ sebagai _baseline model_ dengan _hyperparameter tuning_ menggunakan _grid search_ dan melakukan cross validation dengan split sebanyak 3.

## Data Understanding

Data yang digunakan dalam proyek ini diambil dari [Kaggle Datasets](https://www.kaggle.com/datasets/mexwell/stress-detection-from-social-media-articles/data) yang berjudul "**Stress Detection from Social Media Articles**". Terdapat 4 dataset yang disediakan yaitu:

1.  Reddit_Combi.csv
2.  Reddit_Title.csv
3.  Twitter_Non-Advert-Tabelle 1.csv
4.  Twitter_Full.csv

Namun, proyek ini hanya akan menggunakan dataset pertama (Reddit_Combi.csv)

### Variabel-variabel pada Reddit_Combi dataset adalah sebagai berikut:

-   title : merupakan judul artikel.
-   body : merupakan isi artikel.
-   Body_Title : merupakan judul artikel dan isi artikel yang digabung menjadi satu kumpulan teks.
-   label : merupakan indikator stres, dimana 1 mengindikasikan stres, 0 tidak stres

## Data Preparation

### EDA (Exploratory Data Analysis)

-   Menvisualisasikan distribusi label
-   Menvisualisasikan sebaran panjang teks artikel berdasarkan label
-   Membuang outlier berdasarkan panjang teks dengan metode IQR

### Data preprocessing

Karena terdapat ketidakseimbangan label pada data, maka akan digunakan teknik _over sampling_ dengan menduplikasi data yang tidak seimbang sampai jumlahnya sama besar. Selanjutnya, setelah melalui proses trial & error, data akan diambil sebanyak 2000 sampel saja, karena lebih dari itu memakan waktu yang sangat lama ketika melakukan preprocessing. Data yang disampel sudah seimbang.

Data akan displit menjadi train dan test dengan test size sebesar 10%.

Untuk memastikan bahwa teks dapat memberikan informasi yang berharga maka dilakukan tahapan preprocessing, sebagai berikut:

1.  Mengubah semua teks menjadi huruf kecil (menjaga konsistensi teks).
2.  Membuang noise seperti; special character, angka, tautan, dll (membuang informasi yang tidak berharga untuk model).
3.  Memperbaiki kesalahan ejaan (menjaga konsistensi penulisan kata).
4.  Tokenisasi teks (memecah teks menjadi token atau kata secara individu).
5.  Membuang stop words (membuang kata-kata yang sering muncul dan tidak terlalu bermakna).
6.  Membatasi jumlah kata sebanyak 512 kata (menghindari teks yang terlalu panjang)
7.  Melakukan stemming (mengembalikan kata menjadi kata dasar dengan membuang imbuhan).
8.  Vektorisasi teks menggunakan TF-IDF (mengubah teks menjadi angka agar dapat dimengerti model).

### Modeling

RandomForestClassifier akan digunakan sebagai baseline model dan dilakukan hyperparameter tuning, dengan hyperparameter yang dituning sebagai berikut:

-   n_estimators: [50, 100, 200]
-   max_depth: [16, 32]
-   max_features: [sqrt, log2]
-   min_samples_split: [2, 5]

Untuk memastikan performa model yang robust, maka dilakukan juga cross validation dengan jumlah split sebanyak 3. Metrik utama yang digunakan dalam tuning adalah akurasi, model dengan akurasi tertinggi akan dianggap sebagai model terbaik.

### Evaluation

Hasil hyperparameter tuning pada RandomForestClassifier menghasilkan model terbaik dengan hyperparameter sebagai berikut:

-   n_estimators: 100
-   max_depth: 32
-   max_features: sqrt
-   min_samples_split: 2

Model akan dievaluasi pada test set untuk memastikan bahwa model tidak overfit, dan dapat memprediksi data yang belum pernah dilihat secara akurat. Terdapat 4 metrik evaluasi yang digunakan:

-   $ Accuracy = \frac{(TP+TN)}{(TP+TN+FN+FP)} $
-   $ Precision = \frac{TP}{(TP+FP)} $
-   $ Recall = \frac{TP}{(TP+FN)} $
-   $ F1 = \frac{(2*precision*recall)}{(precision+recall)} $

Keterangan:

-   TP (True Positive) = Prediksi 1, Ground Truth 1
-   TN (True Negative) = Prediksi 0, Ground Truth 0
-   FP (False Positive) = Prediksi 1, Ground Truth 0
-   FN (False Negative) = Prediksi 0, Ground Truth 1

Model terbaik menghasilkan hasil sebagai berikut:

-   Accuracy: 96.5%
-   Recall: 94%
-   Precision: 98.9%
-   F1: 96.5%
