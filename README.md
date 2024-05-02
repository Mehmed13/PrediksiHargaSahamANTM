# Laporan Proyek Machine Learning Prediksi Harga Saham ANTM - Muhammad Fadhil Amri

## Domain Proyek

Saham adalah bukti kepemilikan atas suatu perusahaan [1]. Saham menjadi salah satu instrumen investasi yang populer karena return yang ditawarkan tinggi, tentunya dengan risiko yang tinggi pula. Risiko yang tinggi ini muncul karena tidak ada yang tahu pasti bagaimana harga sebuah saham di masa depan. Secara umum, harga saham ditentukan oleh respons pasar terhadap kinerja dan prospek suatu perusahaan. Perusahaan dengan kinerja dan prospek yang bagus cenderung akan mengalami kenaikan harga saham, ataupun sebaliknya.

Analisis harga saham dapat dikelompokkan menjadi dua teknik, yaitu analisis fundamental dan analisis teknikal. Analisis fundamental adalah analisis yang berfokus pada kinerja perusahaan. Analisis ini biasanya dilakukan dengan menggunakan data dari laporan keuangan perusahaan. Analisis ini bagus digunakan untuk investasi jangka panjang karena kinerja perusahaan dapat membentuk sebuah tren yang kokoh. Meskipun begitu, analisis fundamental tidak mampu memprediksi harga saham dalam jangka pendek karena adanya volatilitas harga yang disebabkan oleh sentimen pasar. Untuk itu, dibutuhkan analisis yang kedua, yaitu analisis teknikal. Analisis ini murni bergantung pada pola-pola yang dibentuk harga saham berdasarkan data historis yang ada [2]. Akibatnya, analisis teknikal mampu memprediksi harga saham dalam jangka pendek dengan lebih baik karena responsivitasnya terhadap perubahan harga.

Saham di sektor pertambangan adalah salah satu saham yang sulit untuk diprediksi pergerakan harganya. Hal ini disebabkan oleh faktor penggerak harga saham yang tidak hanya murni dari kinerja perusahaan, tetapi juga harga material tambang perusahaan tersebut [3]. Material tambang ini juga memiliki volatilitas yang tinggi dan dipengaruhi oleh banyak faktor lain juga. Misalnya, harga emas sangat dipengaruhi oleh kestabilan ekonomi dunia, sedangkan harga nikel belakangan ini sangat didorong oleh pengembangan kendaraan listrik. Faktor-faktor tersebut juga yang membuat saham di sektor pertambangan memiliki tingkat volatilitas yang tinggi, misalnya saham ANTM.

ANTM adalah kode saham dari perusahaan Badan Usaha Milik Negara (BUMN), PT Aneka Tambang Tbk yang selanjutnya akan disebut sebagai Antam. Perusahaan ini adalah perusahaan yang bergerak di sektor pertambangan dengan fokus utama pada penambangan emas dan nikel. Penjualan emas berkontribusi atas 62% total penjualan Antam, sedangkan penjualan nikel berkontribusi atas 33% total penjualan Antam. Akibatnya, harga saham ANTM bergerak mengikuti perubahan harga material dunia, terutama harga emas dan nikel. Sebagai contoh, harga saham ANTM melambung hingga lebih dari 100% sebagai akibat dari pandemi Covid-19 pada tahun 2020 lalu. Pada saat itu, terjadi ketidakstabilan ekonomi dan inflasi yang tinggi. Hal ini membuat masyarakat berbondong-bondong membeli emas yang dikenal <i>safe haven</i> untuk mengamankan aset mereka.

Prediksi harga saham dibutuhkan untuk memberikan sebuah <i>baseline</i> bagi investor. Dengan demikian, mereka bisa menentukan waktu terbaik untuk menjual dan membeli saham sehingga keuntungan yang diperoleh menjadi maksimal. Prediksi harga saham ini bisa dilakukan dengan menggunakan teknik <i> time series forecasting </i> menggunakan pola yang dibentuk dari harga saham di masa lampau.

## Business Understanding

Prediksi saham yang akurat dapat membuat investor mampu bergerak selangkah lebih maju. Hal ini tentunya dapat meningkatkan keuntungan dari investor dengan kemampuan untuk membeli saham pada harga rendah, dan menjualnya pada harga yang lebih tinggi. Selain memaksimalkan keuntungan, prediksi saham yang akurat juga mampu menghindarkan investor dair kerugian yang besar dengan kemampuan untuk menjual saham sebelumnya harganya menjadi turun lebih dalam. Dengan demikian, sistem prediksi saham diharapkan dapat membantu investor dalam membuat keputusan bisnis yang tepat sebelum harga saham yang diprediksi menjadi kenyataan.

Berdasarkan uraian latar belakang dari permasalahan pada domain dan latar belakang bisnis tersebut, dapat dijabarkan <i> problem statements </i> dan <i> goals </i> proyek ini sebagai berikut.

### Problem Statements

- Apakah terdapat suatu pola dari pergerakan harga saham ANTM?
- Bagaimana perkiraan pergerakan harga saham ANTM di masa depan?
- Kapan waktu dan harga yang tepat untuk membeli dan menjual saham ANTM?

### Goals

- Mengidentifikasi pola pergerakan harga saham ANTM
- Membuat model <i> machine learning </i> yang dapat memprediksi harga saham ANTM di masa depan menggunakan data harga pada masa lalu
- Mengetahui waktu dan harga yang tepat untuk membeli dan menjual saham ANTM di masa depan

### Solution statements

- Model FBProphet. Performa diukur melalui perhitungan nilai mean squared error. Model dikatakan baik apabila error yang ada maksimal 10% dari <i> range </i> data.
- Model <i>Long-Short Term Memory</i> (LSTM) [4]. Performa diukur melalui perhitungan nilai mean squared error. Model dikatakan baik apabila error yang ada maksimal 10% dari <i> range </i> data.

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

  ![Gambar Harga Saham ANTM](https://i.ibb.co/D1pD9BG/harga-Saham-ANTM.png)

  Gambar 1. Harga Saham ANTM 2019 - 2024

  Pada gambar 1, Harga Saham ANTM bergerak <i> seasonal</i> dalam tren turun secara landai sebelum tahun 2020. Setelah itu, saham ANTM mengalami kenaikan drastis selama tahun 2020 yang disebabkan oleh krisis ekonomi selama pandemi COVID-19. Memasuki tahun 2021 hingga sekarang, harga saham ANTM bergerak secara <i>seasonal</i> dengan tren turun dalam jangka panjang.

- Distribusi Nilai

  ![Gambar Distribusi Nilai Fitur Close ](https://i.ibb.co/rw0q7Lg/perubahan-Harga-Saham-ANTM.png)

  Gambar 2. Distribusi Nilai Fitur Close

  Pada gambar 2, nilai pada fitur Close tidak terdistribusi normal.

- <i>Outliers Identification </i>

  ![Gambar Boxplot Fitur Close ](https://i.ibb.co/YZBLFLg/boxplot-Harga-Saham.png)

  Gambar 3. Boxplot Fitur Close

  Pada gambar 3, identifikasi dilakukan menggunakan teknik IQR dengan visualisasi dari boxplot. Hasil identifikasi tersebut adalah tidak ditemukan adanya outlier pada data

- <i> Seasonal Decomposition </i>

  ![Gambar Seasonal Decomposition Fitur clode ](https://i.ibb.co/bJ78G92/Seasonal-Decomposition.png)

  Gambar 4. Seasonal Decomposition Fitur Close

  Pada gambar 4, grafik <i> time series </i> dipecah menjadi komponen-komponen tren, seasonal, dan residu. Komponen tren menunjukkan tren yang muncul, Seasonal menunjukkan ada atau tidaknya pola seasonal, dan residu menunjukkan berapa pergeseran dari nilai yang seharusnya. Dari visualisasi tersebut dapat dilihat bahwa secara jangka panjang terdapat tren naik, sedangkan secara jangkap pendek-menengah terdapat tren turun. Grafik harga juga memiliki pola seasonal bersamaan dengan tren.

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

![Gambar Hasil Evaluasi Mean Squared Error Model Prophet dan LSTM](https://i.ibb.co/tppph8s/Error-Evaluation.png)

Gambar 5. Hasil Evaluasi Mean Squared Error Model Prophet dan LSTM

Pada gambar 5, Model Prophet memiliki <i> error </i> yang lebih kecil pada fase <i> train </i> dan fase <i> validation </i>.

### Hasil <i>Forecasting</i>

![Gambar Hasil Forecasting Harga Saham ANTM 2024 - 2025](https://i.ibb.co/3fsBF4J/Forecasting.png)

Gambar 6. Hasil Forecasting Harga Saham ANTM 2024 - 2025

Berdasarkan grafik prediksi harga saham ANTM selama satu tahun pada gambar 6, dapat diperkirakan bahwa waktu yang terbaik untuk membeli saham ANTM pada saat harga berada pada kisaran 1.350 - 1.400 dan menjualnya untuk <i> take profit </i> terdekat pada level 1.700

## REFERENSI

[1] &nbsp;&nbsp;&nbsp;&nbsp; Manh, Ha & Duong, & Siliverstovs, Boriss & Manh, & Duong, Ha, "The Stock Market and Investment", 2006.

[2] &nbsp;&nbsp;&nbsp;&nbsp;Han, Yufeng and Liu, Yang and Zhou, Guofu and Zhu, Yingzi, "Technical Analysis in the Stock Market", 2021.

[3] &nbsp;&nbsp;&nbsp;&nbsp; Antono, Zakia & Jaharadak, Adam Amril & Khatibi, Abdul, "Analysis of factors affecting stock prices in mining sector: Evidence from Indonesia Stock Exchange. Management Science Letters. 9. 1701-1710. 10.5267/j.msl.2019.5.018", 2019.

[4]&nbsp;&nbsp;&nbsp;&nbsp; Bandhu, Kailash Chandra & Litoriya, Ratnesh & Jain, Anshita & Shukla, Anand & Vaidya, Swati, "An improved technique for stock price prediction on real-time exploiting stream processing and deep learning. Multimedia Tools and Applications. 1-21. 10.1007/s11042-023-17130-x", 2023.
