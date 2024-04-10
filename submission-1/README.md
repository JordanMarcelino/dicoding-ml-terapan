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

    -   Melakukan improvement pada _naive bayes_ sebagai _baseline model_ dengan _hyperparameter tuning_ menggunakan _grid search_, menggunakan F1 Score sebagai metrik evaluasi.

## Data Understanding

Data yang digunakan dalam proyek ini diambil dari [Kaggle Datasets](https://www.kaggle.com/datasets/mexwell/stress-detection-from-social-media-articles/data) yang berjudul "**Stress Detection from Social Media Articles**". Terdapat 4 dataset yang disediakan yaitu:

1.  Reddit_Combi.csv
2.  Reddit_Title.csv
3.  Twitter\_ Non-Advert-Tabelle 1.csv
4.  Twitter_Full.csv

Namun, proyek ini hanya akan menggunakan dataset pertama (Reddit_Combi.csv)

### Variabel-variabel pada Reddit_Combi dataset adalah sebagai berikut:

-   title : merupakan judul artikel.
-   body : merupakan isi artikel.
-   Body_Title : merupakan judul artikel dan isi artikel yang digabung menjadi satu kumpulan teks.
-   label : merupakan indikator stres, dimana 1 mengindikasikan stres, 0 tidak stres

**Rubrik/Kriteria Tambahan (Opsional)**:

-   Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

-   Menjelaskan proses data preparation yang dilakukan
-   Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

-   Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
-   Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
-   Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

-   Penjelasan mengenai metrik yang digunakan
-   Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

-   Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

-   _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
-   Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
