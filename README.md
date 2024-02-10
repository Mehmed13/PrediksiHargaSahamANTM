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

  Menampilkan data-data deskriptif seperti tipe data dari variabel, mean, min, max, kuartil, dan standard deviasi

- <i> Missing Value Identification </i>

  Ditemukan satu baris yang memiliki nilai kosong pada fitur 'Close'

- <i> Time Series Plot </i>

  ![Gambar Harga Saham ANTM](https://drive.google.com/uc?export=download&id=1l6xBTXkD25lccLsX3-o2tyIcLAHzWmFu)

  Harga Saham ANTM bergerak <i> seasonal</i> dalam tren turun secara landai sebelum tahun 2020. Setelah itu, saham ANTM mengalami kenaikan drastis selama tahun 2020 yang disebabkan oleh krisis ekonomi selama pandemi COVID-19. Memasuki tahun 2021 hingga sekarang, harga saham ANTM bergerak secara <i>seasonal</i> dengan tren turun dalam jangka panjang.

- Distribusi Nilai

  ![Gambar Distribusi Nilai Fitur Close ](https://drive.google.com/uc?export=download&id=1lDm1-ncJXAowfo-G4AndHVO1AT4T5KCk)

  Nilai pada fitur Close tidak terdistribusi normal

- <i>Outliers Identification </i>

  ![Gambar Boxplot Fitur Close ](https://drive.google.com/uc?export=download&id=1lElwNPzGpClnkIpnUJYbeNfvZtwqX4Hq)

  Identifikasi dilakukan menggunakan teknik IQR dengan visualisasi dari boxplot. Hasil identifikasi tersebut adalah tidak ditemukan adanya outlier pada data

- <i> Seasonal Decomposition </i>

  ![Gambar Seasonal Decomposition](https://drive.google.com/uc?export=download&id=1lH3QNFSm2Lp7Tv21FHZ0782zmLT8oF5X)

  Grafik <i> time series </i> dipecah menjadi komponen-komponen tren, seasonal, dan residu. Komponen tren menunjukkan tren yang muncul, Seasonal menunjukkan ada atau tidaknya pola seasonal, dan residu menunjukkan berapa pergeseran dari nilai yang seharusnya. Dari visualisasi tersebut dapat dilihat bahwa secara jangka panjang terdapat tren naik, sedangkan secara jangkap pendek-menengah terdapat tren turun. Grafik harga juga memiliki pola seasonal bersamaan dengan tren.

## Data Preparation

- <i> Data Formatting </i> <br>
  Data yang sudah diambil dari sumber tersedia diformat agar berada dalam frekuensi harian. Data bursa yang hanya buka pada <i>trading day</i> membuat perlu dilakukannya pengisian nilai pada hari-hari libur. Pengisian nilai dilakukan dengan metode <i> backward-fill </i> karena data harga pada hari libur sama dengan harga pada hari sebelum libur. <br>
  Data Formatting ini dilakukan dengan tujuan membuat proses pembelajaran menjadi lebih <i> smooth </i>, terutama untuk model Prophet sehingga akurasi model dapat meningkat dengan signifikan.

- <i>Data Splitting </i> <br>
  Data dibagi menjadi data <i> train </i> dan data <i> validation </i> dengan perbandingan 80:20. Proses ini perlu dilakukan untuk memvalidasi dan menguji seberapa bagus akurasi dari model yang dibuat sebelum digunakan untuk memprediksi data di masa depan.

- <i>Data Scaling </i> <nr>
  Nilai 'Close' diubah menjadi berada di dalam interval 0..1 dengan menggunakan teknik <i> MinMax Scaling </i>. Teknik ini dipilih karena data tidak terdistribusi normal dan skala sangat penting dalam tahap prediksi. <i> Data Scaling </i> dilakukan untuk mengurangi <i>noise </i> dan mempercepat proses pembelajaran menjadi konvergen.

- <i> Data Transforming </i> <br>
  Data diubah bentuknya agar bisa digunakan oleh model yang bersangkutan. DataFrame diubah menjadi bentuk <i>batch</i> untuk bisa digunakan pada model LSTM. Sementara itu, nama kolom pada DataFrame yang ada juga diubah menjadi kolom 'ds' untuk 'Date' dan 'y' untuk 'Close' agar bisa digunakan oleh model Prophet.

## Modeling

1. Model Prophet <br>
   Pemodelan dilakukan dengan pertama-tama mengimpor model Prophet. Setelah itu, buat sebuah objek dari kelas Prophet dengan mengisi parameter 'interval_width' dengan 0.95 yang menandakan interval kepercayaan dari model adalah 95% dari <i> default </i> 80%. Terakhir, lakukan pelatihan menggunakan data <i> train </i> dengan memanggil <i> method </i> 'fit'.

   Parameter: <br>

   - interval-width: interval kepercayaan model

   Kelebihan: <br>

   - Mudah digunakan
   - Penanganan komponen <i>seasonal</i> secara otomatis
   - Mudah diinterpretasikan
   - Cepat

   Kekurangan: <br>

   - Tidak optimal untuk data kompleks
   - Kurang fleksibel

2. Model LSTM <br>
   Pemodelan diawali dengan membangun jaringan saraf tiruan. Pada proyek ini digunakan model sekuensial yang terdiri dari 6 <i> layer </i>. <i> Layer-layer </i> tersebut adalah 2 <i> LSTM layer </i>, 3 <i> Dense layer </i>, dan 1 <i> Drop layer </i>. Setelah dikonstruksi, model di-<i>compile </i> agar bisa dilatih. Terakhir, lakukan pelatihan menggunakan data <i> train </i> dengan memanggil <i> method </i> 'fit'.

   Parameter: <br>

   - loss: <i> Loss Function </i>
   - optimizer: <i> Optimizer Function </i>
   - metrics: Metrik evaluasi

   Kelebihan: <br>

   - Kemampuan pemodelan yang rumit
   - Memahami konteks data sebelumnya
   - Fleksibel

   Kekurangan: <br>

   - Berpotensi <i>overfitting</i>
   - Membutuhkan banyak data
   - Implementasi kompleks

Model final yang akan digunakan adalah model Prophet karena memiliki <i> error </i> yang lebih kecil dan lebih cocok untuk pemodelan yang sederhana dengan data yang cukup terbatas.

## Evaluation

### Metrik Evaluasi

Metrik Evaluasi yang digunakan pada proyek ini adalah <i> Mean Square Error </i> (MSE). <br>
Formula MSE:

$$ MSE = \frac{1}{n} \sum*{i=1}^n (y*{i}-y\_{pred,i})^2$$

Metrik MSE bekerja dengan cara merata-ratakan jumlah dari selisih antara nilai sebenarnya dengan nilai prediksi (<i>error</i>) lalu rata-rata tersebut diakarkan sehingga nilai yang dihasilkan metrik tidak memiliki skala yang besar.

### Hasil Evaluasi

![Gambar Evaluasi MSE](https://drive.google.com/uc?export=download&id=1lKAoi_spaS51Cx2ot1lmS_FPNINB9Asy)

Model Prophet memiliki <i> error </i> yang lebih kecil pada fase <i> train </i> dan fase <i> validation </i>.

### Hasil <i>Forecasting</i>

![Gambar Hasil Forecasting](https://drive.google.com/uc?export=download&id=1lVvv-mCtDoXINzLjr0WnJImoFSaSU0Hd)

Berdasarkan grafik prediksi harga saham ANTM selama satu tahun, dapat diperkirakan bahwa waktu yang terbaik untuk membeli saham ANTM pada saat harga berada pada kisaran 1.350 - 1.400 dan menjualnya untuk <i> take profit </i> terdekat pada level 1.700
