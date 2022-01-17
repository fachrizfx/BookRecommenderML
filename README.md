# Laporan Proyek Machine Learning
---
## Project Overview

Kita tentu sering mendengar kalimat "Buku adalah jendela dunia", mungkin seberapa dari kita sudah mengetahui arti dari kalimat tersebut. Tetapi bagi yang belum, kalimat tersebut menggambarkan betapa pentingnya buku karena buku adalah sumber ilmu yang tentu tidak ada habisnya. Menurut [11] membaca buku memiliki banyak manfaat, salah satunya adalah dapat mencegah penurunan kognitif yang diakibatkanusia. Oleh karena itu kita harus membudidayakan budaya membaca. Kita dapat membaca melewati buku fisik ataupun buku digital/E-Book. Di bidang Machine Learning kita dapat berkontribusi dalam membudidayakan budaya membaca dengan banyak cara, salah satunya adalah dengan membuat recommendation system agar kita semua dapat membaca sesuai dengan apa yang kita suka.

## Business Understanding

Dari penjelasan pada bagian Project Overview kita mengetahui bahwa sistem rekomendasi dapat membantu membudidayakan budaya membaca buku. Tetapi bagi pelaku bisnis apa keuntungannya? Jawabannya bisa bermacam-macam, salah satunya adalah jika kita memiliki bisnis toko buku, kita dapat mengembangkan sistem rekomendasi yang dapat merekomendasikan user/pelanggan buku berdasarkan kategori buku apa yang mereka suka, sehingga bisnis toko buku-pun dapat meraih keuntungan berkat terjualnya buku-buku yang direkomendasikan oleh sistem.

### Problem Statements

Tentu kita harus memiliki tujuan/goal. Untuk membuat goal tersebut kita harus memiliki permasalahan, atau pada project ini adalah sebagai berikut:
- Bagaimana membuat sistem rekomendasi dengan teknik **content base filtering**?
- Bagaimana kecocokan rekomendasi buku yang diberikan dengan kesesuaian user?
- Bagaimana sistem rekomendasi dapat membantu dibidang bisnis?

### Goals

Kita dapat membuat goal dari permasalah diatas. Goal pada project ini adalah sebagai berikut:
- Membuat recommendation system berdasarkan kesukaan user
- Mengetahui kecocokan rekomendasi terhadap user
- Melihat apakah model/sistem cocok untuk dipakai dibidang bisnis

### Solution statements

Agar bisa menjawab permasalahan yang telah diuraikan diatas kita bisa membuat Solution sebagai berikut:
- Menerapkan teknik personalized recommendation system
- Melihat apakah kategori buku yang kita pernah baca, terdapat pada hasil rekomendasi
- Mengevaluasi hasil dari pernyataan solution statement kedua untuk menentukan apakah cocok untuk digunakan dibidang bisnis

## Data Understanding

![insightdata](https://drive.google.com/uc?export=view&id=1YYGyqpuzy-I7fKDSJnktaX3U6162Mi1y)

Dataset yang akan saya pakai pada project ini memiliki 2  folder pada directory yaitu: 'Book reviews' dan 'Books Data with Category Language and Summary', disini saya akan pakai folder kedua yaitu 'Books Data with Category Language and Summary'. Di dalam folder ini kita akan menjumpai file yang bernama 'Preprocessed_data.csv', file inilah yang berisi rangkuman dari folder pertama. Jumlah data yang terdapat pada file ini juga cukup banyak yaitu 1031175. Untuk melihat ataupun mendownload dataset bisa melewati link berikut: [Kaggle]

Variabel-variabel pada Book-Crossing: User review ratings dataset adalah sebagai berikut:
- user_id : user id
- location : lokasi user
- age : umur user
- isbn : kode ISBN buku tersebut
- rating : rating buku tersebut
- book_title : judul buku tersebut
- book_author : pengarang/penulis dari buku tersebut
- year_of_publication : tahun publikasi
- publisher	: penerbit buku
- Summary : rangkuman buku
- Language : bahasa dari buku
- Category : kategori buku
- city : kota dari lokasi user
- state : negara bagian dari lokasi user
- country : negara dari lokasi user

Sebelum kita masuk ke tahap Univariate Analysis, kita akan drop row 'Unamed: 0', 'img_s', 'img_m', 'img_l', 'Summary', 'location' karena column tersebut tidak terlalu berguna. Alasa column 'location' juga di drop adalah dikarenakan column tersebut sudah direpresentasikan oleh column 'city', 'state', 'country'.

## Univariate Analysis
Sebelum kita masuk ke dalam tahap Data Preparation, kita bisa memasuki tahap Data Analysis terlebih dahulu untuk mengetahui data pada dataset ini. Pada tahap ini saya akan menggunakan teknik Univariate Data Analysis. Teknik ini adalah teknik paling dasar untuk menganalysis data. Secara singkat Uni berarti satu, yang berarti menganalysis data secara terpisah (satu per satu). Tujuannya adalah untuk melihat dan memberikan insight mengenai data kita.

![hasilanalysis](https://drive.google.com/uc?export=view&id=181cwBBJIxJ1KR0wxl5xlGAeTXNpNRj7g)

Kita dapat simpulkan bahwa column 'city', 'state', dan 'country' terdapat nilai yang bernilai NaN, ini berarti pada column tersebut terdapat missing value. Jika kita perhatikan lagi kita bisa lihat bahwa pada column 'Language', dan 'Category' juga terdapat missing value yang direpresentasikan dengan nilai '9', nilai tersebut dapat dibilang missing value dikarenakan tidak mungkin column 'Language', dan 'Category' memiliki nilai yang sama yaitu '9'. Jumlah dari colum yang memiliki nilai '9' ini juga cukup banyak yaitu sebesar 176176. Kita akan tangani semua missing value ini pada tahap **Data Cleaning**.

## Data Preprocessing
Pada tahap Data Preprocessing disini saya tidak akan melakukan teknik-teknik untuk Data Preprocessing karena dataset yang saya gunakan ini sudah tergabung semua, atau bisa dibilang siap digunakan dan hanya perlu melakukan Data Cleaning dari missing value. Oleh karena itu disini saya tidak akan melakukan apa-apa selain dari melakukan teknik Sorting dengan fungsi .sort_values [01] dari library Pandas.

## Data Preparation
Pada tahap ini kita akan mempersiapkan data untuk dilatih. Persiapan data disini mencangkupi Data Cleaning dan Feature Selection. Mari kita mulai dengan langkah pertama yaitu Data Cleaning.

### Data Size Reduction
Paragraph ini saya tulis pada tahap Modelling. Jadi pada saat saya ingin menghitung cosine similarity dari project ini saya menemukan masalah, yaitu runtime saya terus-menerus mengalami crash dikarenakan RAM usagenya yang melebihi batas maksimal, jika saya mengaktifkan Hardware accelerator GPU, maka akan muncul error bahwa runtime tidak bisa connect dengan GPU pada Back-end. Hal ini bisa saja dikarenakan saya sering menggunakan GPU sehingga menurut [10] "As a result, users who use Colab for long-running computations, or users who have recently used more resources in Colab, are more likely to run into usage limits and have their access to GPUs and TPUs temporarily restricted" atau dapat disimpulkan bahwa jika kita menggunakan resource komputasi lebih banyak akan cenderung mengalami batas penggunaan, ataupun ada faktor-faktor lain yang menyebabkan hal ini. Oleh karena itu saya disini memutuskan untuk mereduksi dataset size dengan cara drop > 50%. Saya sudah mencoba mereduksi < 40% tetapi hasilnya tetap sama yaitu runtime crash.

### Data Cleaning
Seperti yang sudah dibilang pada bagian Univariate Analysis, data kita memiliki missing value yang cukup banyak. Oleh karena itu pada tahap ini kita akan mengani missing value.

#### Menangani Missing Value
Kita sudah tahu bahwa missing value terletak pada column 'city', 'state', 'country', 'Language', dan 'Category'. Kedua column 'Language' dan 'Category ini memiliki missing value yang sudah di fill/direpresentasikan dengan nilai '9'. Missing value ini tidak terdeteksi oleh fungsi .isnull() dari library Pandas [02]. Hal ini dikarenakan fungsi .isnull() hanya mendeteksi null value. Nilai null merupakan nilai yang tidak ada nilainya sementara itu missing value pada kedua column ini direpresentasikan dengan '9' yang bertipe Object/String oleh karena itu missing value sudah direpresentasikan dengan nilai apapun yang ada di dalam string tersebut. Teknik tersebut menurut [03] disebut 'Imputation method for categorical columns' atau jika di translasi Metode imputasi untuk kolom categorical. Menurut [03] Salah satu kelebihan teknik ini adalah mencegah hilangnya data dan kekurangannya adalah dapat membuat kinerja menurun saat proses encoding.

Menurut [03] juga, salah satu cara untuk menangani hal ini adalah menggantinya dengan category yang paling sering muncul. Namun menurut saya hal ini tidak cocok dikarenakan kita merekomendasikan buku berdasarkan categorynya. Contoh bayangkan kita suka membaca buku berkategori Technology dan di dalam aplikasi/website E-book, kita direkomendasikan mengenai buku Action. Hal tersebut lebih cocok di masukan pada bagian explore bukan bagian rekomendasi untuk anda, sehingga buku yang direkomendasikan oleh sistem tersebut tidak cocok dengan apa yang kita suka. Oleh karena itu kita akan menggunakan teknik Delete Rows with Missing Value menggunakan fungsi .dropna() [04] dari library Pandas.

Sebelum kita menghapus row dengan missing value kita perlu ingat bahwa missing value pada column 'Language', dan 'Category' direpresentasikan oleh string yang bernilai '9' oleh karena itu kita harus menjadikan nilai '9' menjadi NaN agar bisa didrop oleh fungsi .dropna(). Kita bisa mengganti nilai tersebut dengan fungsi .replace() dari library Pandas [05].

Output:
```
user_id                0
age                    0
isbn                   0
rating                 0
book_title             0
book_author            0
year_of_publication    0
publisher              0
Language               0
Category               0
city                   0
state                  0
country                0
dtype: int64
```

```
after data cleaning dataset size: 303428
```

Setelah kita bersihkan kita bisa lihat bahwa data kita sudah bersih, tetapi kita mengalami data loss lebih dari 40%, yang dapat dibilang cukup signifikan. Selanjutnya kita akan drop semua ISBN yang duplikat, hal ini dilakukan agar hanya ada buku yang unique saja. Kita akan drop dengan fungsi .drop_duplicates() [06]. 

### Feature Selection
Pada tahap ini kita akan memilih feature yang relevan dengan variable target kita.

#### Memisahkan Feature dari Dataset
Pertama-tama kita harus konversi data menjadi list terlebih dahulu. Konversi ini dapat kita lakukan dengan fungsi .tolist() [07] dari library NumPy. Sebelum memisahkan feature, kita perlu memilih feature yang akan digunakan terlebih dahulu. Pada project ini saya akan menggunakan column 'isbn', 'book_title', 'book_author', 'publisher', 'Language', dan 'Category'.

Setelah transformasi menjadi list, kita bisa membuat dataframe dengan Pandas. Untuk membuatnya disni kita akan menggunakan dictionary untuk memasangkannya dengan key-value. Hasilnya sebagai berikut:

![featuredataset](https://drive.google.com/uc?export=view&id=1CDoN2wdVi3iXNHBFv6j_LE_UmuhxvXmb)

Tahap selanjutnya adalah tahap yang ditunggu-tunggu yaitu Modelling.

## Modeling
Seperti yang kita ketahui bahwa output dari system rekomendasi adalah *Top-N* yang berarti tidak seperti model classification, regression, dll yang hanya memiliki satu output yang berupa prediksi. Lain halnya dengan sistem rekomendasi yang menyajikan banyak output. Contoh dari system rekomendasi adalah Youtube, video-video yang direkomendasikan Youtube adalah salah satu contoh output Top-N. Mengapa sistem rekomendasi menyajikan output berupa Top-N? Jawabannya sangat simpel yaitu, tidak mungkin system merekomendasikan hanya satu hal, ini dikarenakan jika kita tidak suka dengan rekomendasinya maka apa rekomendasi selanjutnya? Jadi jawabannya adalah agar kita bisa memilih mana yang kita suka.

### Vectorizer
Pertama-tama kita harus mulai dengan Vectorizer, pada project ini kita akan menggunakan CountVectorizer dari [08] library Scikit Learn. Sebenarnya kita bisa menggunakan TF-IDF Vectorizer tetapi disini saya menggunakan CountVectorizer dikarenakan TF-IDF bekerja dengan memberikan skor terhadap suatu kata, kata yang sering muncul dan jarang muncul akan diberikan skor berbeda, skor tersebut berfungsi untuk menentukan makna ataupun konteks dari suatu kalimat. Pada kasus ini kita tidak perlu sampai memahami makna dari teks, karena disini kita hanya perlu untuk memperoleh informasi sebanyak mungkin untuk menghitung derajat kemiripan dengan cosine similarity. Maka dari itu disini kita akan menggunakan CountVectorizer.

Pada kasus ini saya akan gunakan Category sebagai acuan untuk merekomendasikan buku. Pertanyaannya, apasih tujuan vectorizer? Tujuan dari vectorizer pada recommendation system adalah untuk mencari representasi yang tepat untuk merepresentasikan category.

Untuk memulai pastinya kita harus mengimpor library [08] terlebih dahulu, tetapi karena saya sudah mengimpor semua library pada awal cell, maka kita tidak perlu melakukan import ulang.

Pertama-tama kita harus mulai dengan Vectorizer, pada project ini kita akan gunakan TF-IDF Vectorizer [08] dari library Scikit Learn. Pada kasus ini saya akan gunakan Category sebagai acuan untuk merekomendasikan buku. Pertanyaannya, apasih tujuan vectorizer? Tujuan dari vectorizer pada recommendation system adalah untuk mencari representasi yang tepat untuk merepresentasikan category. Selanjutnya kita bisa merepresentasikan CountVectorizer dengan variable bernama Vec. Lalu vectorizer akan mengonversi kumpulan teks menjadi matriks jumlah token pada column category, agar vectorizer mengonversi teks kita harus gunakan .fit(). Setelah semuanya selesai kita bisa transformasi ke dalam bentuk matrix, dan agar vector menjadi bentuk matrix kita harus transformasi lagi menjadi dense dengan fungsi .todense().

### Cosine Similarity
Cosine similarity menghitung derajat kesamaan antara masing-masing category. Kita dapat menghitung cosine similarity dengan fungsi cosine_similarity() [09] dari library Scikit Learn. Output dari fungsi ini berupa matrix array sehingga kita bisa menyajikannya dalam bentuk dataframe.

![cosdataframe](https://drive.google.com/uc?export=view&id=1v9-0-H3T6c2CSqkvqN2ZOtlgQNItf1PB)

Kita dapat membuat fungsi untuk mengeluarkan output Top-N. Untuk menggunakan fungsi tersebut kita dapat menggunakan salah satu title dari buku kemudian kita bisa memilih nilai Top-N-nya yang direpresentasi dengan K. Berikut adalah salah satu contoh dari output Top-N:

![recommendations](https://drive.google.com/uc?export=view&id=17QxLvxrfOYzM2yesO4OTbiCYNAEU1hMr)

## Evaluation
Kita sudah membuat model untuk merekomendasikan buku dengan teknik Content Based Filtering. Sekarang kita berada di tahap evaluasi. Sekarang kita bisa menjawab semua permasalahan yang telah dijelaskan pada bagian Problem Statement:

- Bagaimana membuat sistem rekomendasi dengan teknik **content based filtering**?
- Bagaimana kecocokan rekomendasi buku yang diberikan dengan kesesuaian user?
- Bagaimana sistem rekomendasi dapat membantu dibidang bisnis?

Kita sudah membuat sistem rekomendasi dengan teknik Content Based Filtering. Sekarang kita bisa mendapatkan rekomendasi dengan fungsi get_recommendations(). Jika kita test fungsinya dengan judul buku 'Dragonshadow' kita bisa lihat output sebagai berikut:

![dragonshadowcategory](https://drive.google.com/uc?export=view&id=1MLLLVpqO-QwyWZds-O0SxTk38pIl2MZE)

Kita bisa lihat bahwa buku tersebut termasuk ke dalam category 'Fiction' atau fiksi, lalu hasil yang direkomendasikan oleh model juga merupakan category fiction, ini mengindikasikan kecocokan kesukaan user dengan yang direkomendasikan. Jadi kita dapat simpulkan bahwa buku yang direkomendasikan cocok bagi user sehingga user bisa membeli buku tersebut, sehingga bisnis toko buku kita bisa mendapatkan keuntungan dari user-user yang suka dengan hasil rekomendasi sistem.

Selanjutnya kita akan mengukur presisi dengan Precision Metric. Metrik ini dapat kita peroleh dengan cara membagi rekomendasi yang relevan dengan jumlah rekomendasi. Pertanyaannya bagaimana cara kita mengetahui rekomendasi yang relevan dengan yang tidak? Untuk mengetahuinya kita bisa lihat dari category rekomendasi apakah cocok dengan category yang kita tuliskan sebagai Input. Pada kasus ini kita bisa lihat pada gambar diatas bahwa semua rekomendasi kita memiliki category yang sama, yaitu Fiction. 

Kita akan menggunakan Precision Metric untuk mengukur presisi dari sistem rekomendasi. Metrik ini bekerja dengan membagi jumlah rekomendasi yang relevan dengan jumlah rekomendasi lalu kita mengkalikannya dengan 100 untuk menjadikannya sebagai persentase, atau secara matematis ditulis sebagai berikut:

```
precision percentage = relevant / number of top-n * 100
```

Secara teknis metrik ini bekerja dengan cara membandingkan jumlah rekomendasi yang relevan dengan jumlah seluruh rekomendasi sehingga kita bisa mendapatkan perbandingan. Sebagai contoh bayangkan kita memiliki jumlah rekomendasi yang relevan sebesar 9 dan jumlah seluruh rekomendasi kita adalah 13, sehingga kita mendapatkan perbandingan 9/13. Seperti yang kita ketahui untuk menjadikannya persentase kita perlu mengkalikan perbandingan dengan 100 sehingga kita memiliki bentuk akhir 9/13*100, lalu kita bisa menerapkan matematika dasar (kabataku) untuk mendapatkan output presisi dalam bentuk persentase sebesar 69.2%.

Kita daoat membuat fungsi untuk mengukur presisi menggunakan Precision Metric. Fungsi yang kita buat ini akan membutuhkan dua input, antara lain Number of Relevant Recommendations dan Number of Top-N. Input-input tersebut kemudian dibagi (sesuai dengan rumus diatas) lalu dikali dengan 100 agar menjadi satuan persen. Berikut adalah salah satu output persentase dari presisi sistem:

```
Recommendation System Precision Percentage: 100.0%
```

Kita bisa lihat bahwa presisi sistem kita mencapai 100%! Sampai disini kita sudah melihat bahwa system kita dapat merekomendasikan buku yang relevan dengan kesukaan si user/pelanggan.

**---Ini adalah bagian akhir laporan---**

# Daftar Referensi
<br />[[Kaggle]] Bhatia, R. (2021, February 17). Book-Crossing: User review ratings (Version 3) [A collection of book ratings]. Kaggle. https://www.kaggle.com/ruchi798/bookcrossing-dataset
<br />[[01]] Pandas Pydata. (n.d.-c). pandas.DataFrame.sort_values. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
<br />[[02]] Pandas Pydata. (n.d.-c). pandas.isnull. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.isnull.html
<br />[[03]] Kumar, S. (2020, July 24). 7 Ways to Handle Missing Values in Machine Learning. Towards Data Science. Retrieved January 16, 2022, from https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
<br />[[04]] Pandas Pydata. (n.d.-b). pandas.DataFrame.dropna. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
<br />[[05]] Panda Pydata. (n.d.). pandas.DataFrame.replace. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
<br />[[06]] Pandas Pydata. (n.d.). pandas.DataFrame.drop_duplicates. Pandas. Retrieved January 16, 2022, from https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
<br />[[07]] NumPy. (n.d.). numpy.ndarray.tolist. Retrieved January 16, 2022, from https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
<br />[[08]] Scikit Learn. (n.d.-a). sklearn.feature_extraction.text.TfidfVectorizer. Retrieved January 16, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
<br />[[09]] Scikit Learn. (n.d.). sklearn.metrics.pairwise.cosine_similarity. Retrieved January 16, 2022, from http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
<br />[[10]] Google. (n.d.). Colaboratory FAQ. Google Research. Retrieved January 16, 2022, from https://research.google.com/colaboratory/faq.html
<br />[[11]] Bola. (2021, February 16). 7 Manfaat Membaca Buku yang Masih Belum Banyak Diketahui. Retrieved January 16, 2022, from https://www.bola.com/ragam/read/4484476/7-manfaat-membaca-buku-yang-masih-belum-banyak-diketahui

<br />
<br />

[Kaggle]: https://www.kaggle.com/ruchi798/bookcrossing-dataset
[01]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
[02]: https://pandas.pydata.org/docs/reference/api/pandas.isnull.html
[03]: https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
[04]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
[05]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
[06]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html
[07]: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
[08]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
[09]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
[10]: https://research.google.com/colaboratory/faq.html
[11]: https://www.bola.com/ragam/read/4484476/7-manfaat-membaca-buku-yang-masih-belum-banyak-diketahui
