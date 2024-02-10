# Laporan Proyek Machine Learning Prediksi Harga Saham ANTM - Muhammad Fadhil Amri

## Domain Proyek

Saham adalah bukti kepemilikan atas suatu perusahaan. Saham menjadi salah satu instrumen investasi yang populer karena return yang ditawarkan tinggi dengan risiko yang tinggi pula. Risiko yang tinggi ini muncul karena tidak ada yang tahu pasti bagaimana harga sebuah saham di masa depan, terutama saham pertambangan. Saham ini memiliki tingkat volatilitas yang tinggi, misalnya saham ANTM. Harga saham ANTM bergerak mengikuti perubahan harga material dunia, terutama harga nikel dan emas.

Prediksi harga saham dibutuhkan untuk memberikan sebuah <i>baseline</i> bagi investor. Dengan demikian, mereka bisa menentukan waktu terbaik untuk menjual dan membeli saham sehingga keuntungan yang diperoleh menjadi maksimal. Prediksi harga saham ini bisa dilakukan dengan menggunakan teknik <i> time series forecasting </i> menggunakan pola yang dibentuk dari harga saham di masa lampau.

**Referensi:**

- [Prophet](https://facebook.github.io/prophet/)<br>
- [A survey on long short-term memory networks for time series prediction](https://www.sciencedirect.com/science/article/pii/S2212827121003796) <br>
- [Stock Forecasting Using Prophet vs. LSTM Model Applying
  Time-Series Prediction](http://paper.ijcsns.org/07_book/202202/20220224.pdf)

## Business Understanding

Berdasarkan uraian latar belakang dari permasalahan pada domain, dapat dijabarkan <i> problem statements </i> dan <i> goals </i> proyek ini sebagai berikut.

### Problem Statements

- Apakah terdapat suatu pola dari pergerakan harga saham ANTM?
- Bagaimana perkiraan pergerakan harga saham ANTM di masa depan?
- Kapan waktu dan harga yang tepat untuk membeli dan menjual saham ANTM?

### Goals

- Mengidentifikasi pola pergerakan harga saham ANTM
- Membuat model <i> machine learning </i> yang dapat memprediksi harga saham ANTM di masa depan menggunakan data harga pada masa lalu
- Mengetahui waktu dan harga yang tepat untuk membeli dan menjual saham ANTM di masa depan

### Solution statements

- Model FBProphet
- Model <i>Long-Short Term Memory</i> (LSTM)

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data harga saham ANTM dengan periode harian selama lima tahun (Februari 2019 - Februari 2024). Dataset yang digunakan terdiri atas 1.230 baris dan 6 kolom. Dataset ini diunduh dari <i>website</i> Yahoo Finance melalui tautan [Dataset Harga Saham ANTM](https://finance.yahoo.com/quote/ANTM.JK/history?period1=1609459200&period2=1703980800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true).

### Variabel-variabel pada Dataset Harga Saham ANTM

- Date : Tanggal dari harga saham yang tercatat.
- Open : Harga pembukaan saham.
- High : Harga tertinggi saham dalam satu hari.
- Low : Harga terendah saham dalam satu hari.
- Close : Harga penutupan saham.
- Adj Close: Harga penutupan saham dengan mempertimbangkan aksi korporasi seperti <i>split</i> dan dividen.

Fitur yang digunakan mulai dari EDA hingga tahap Forecasting hanya 'Date' dan 'Close' karena sudah mewakili harga saham yang akan diprediksi.

### <i>Exploratory Data Analysis </i> (EDA)

- Statitika deskriptif
  > Menampilkan data-data deskriptif seperti tipe data dari variabel, mean, min, max, kuartil, dan standard deviasi
- <i> Missing Value Identification </i>
  > Ditemukan satu baris yang memiliki nilai kosong pada fitur 'Close'
- <i> Time Series Plot </i>

  > Harga Saham ANTM bergerak <i> seasonal</i> dalam tren turun secara landai sebelum tahun 2020. Setelah itu, saham ANTM mengalami kenaikan drastis selama tahun 2020 yang disebabkan oleh krisis ekonomi selama pandemi COVID-19. Memasuki tahun 2021 hingga sekarang, harga saham ANTM bergerak secara <i>seasonal</i> dengan tren turun dalam jangka panjang.

  ![Gambar Harga Saham ANTM](https://drive.google.com/uc?export=download&id=1l6xBTXkD25lccLsX3-o2tyIcLAHzWmFu)

- Distribusi Nilai

  > Nilai pada fitur Close tidak terdistribusi normal

  ![Gambar Distribusi Nilai Fitur Close ](https://drive.google.com/uc?export=download&id=1lDm1-ncJXAowfo-G4AndHVO1AT4T5KCk)

## Data Preparation

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

### Metrik Evaluasi

Metrik Evaluasi yang digunakan pada proyek ini adalah <i> Root Mean Square Error </i> (RMSE). <br>
Formula RMSE:

$$ RMSE = \sqrt{\frac{1}{n} \sum*{i=1}^n (y*{i}-y\_{pred,i})}$$

Metrik RMSE bekerja dengan cara merata-ratakan jumlah dari selisih antara nilai sebenarnya dengan nilai prediksi (<i>error</i>) lalu rata-rata tersebut diakarkan sehingga nilai yang dihasilkan metrik tidak memiliki skala yang besar.

### Hasil Evaluasi
