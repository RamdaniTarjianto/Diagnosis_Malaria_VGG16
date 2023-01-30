## Abstrack
Malaria merupakan penyakit darah yang disebabkan oleh Plasmodium yang ditularkan melalui gigitan nyamuk Anopheles betina. Penyakit ini sudah menyebabkan lebih dari 200 juta infeksi dan 400.000 kematian setiap tahun di seluruh dunia. Umumnya pendeteksian malaria bersifat mikroskopis. Selain itu, malaria sangat sulit untuk didiagnosis karena gejalanya yang umum sehingga memerlukan waktu yang lama untuk mendeteksi parasit Plasmodium dan juga bergantung pada keterampilan dan keahlian seorang ahli mikroskop. Saat ini, banyak peneliti yang menerapkan machine learning untuk mendeteksi penyakit ini, seperti penggunaan metode deep learning yang dirancang untuk membedakan antara sampel darah yang sehat dan yang terinfeksi dengan nilai akurasi sebesar 95%. Terlepas dari hasil yang memuaskan, model ini tentunya memiliki kekurangan, baik dalam hal kecepatan maupun akurasi. Selain itu, model yang digunakan terlalu besar atau berat sehingga terdapat beberapa kesalahan dalam prediksi. Pada penelitian ini kami mengusulkan untuk menggunakan VGG16 yang merupakan model transfer learning untuk mengklasifikasikan citra sel darah merah. VGG16 tidak memerlukan mekanisme ekstraksi fitur dan tidak ada fase pemilihan fitur menengah penggunaannya, sehingga dapat mempercepat proses pendeteksian parasit Plasmodium pada sel darah merah. Model VGG16 ini memperoleh nilai terbaik di semua metrik seperti akurasi (96%), presisi (96%), recall (96%) dan skor F1 (96%) pada 27.558 citra sel darah merah.

## Latar Belakang
Malaria merupakan penyakit darah yang disebabkan oleh parasit Plasmodium yang ditularkan gigitan nyamuk Anopheles betina. Jenis spesies Plasmodium yang menyebabkan malaria adalah Plasmodium falciparum, Plasmodium vivax, Plasmodium ovale, dan Plasmodium Malariae. Plasmodium falciparum adalah spesies yang bertanggung jawab atas demam malaria di sebagian besar kasus malaria. Penyakit malaria menjadi masalah kesehatan dunia terutama di kawasan tropis dan subtropis negara berkembang. Penyakit ini dengan mudah ditularkan ke manusia dan menyebabkan lebih dari 200 juta infeksi dan 400.000 kematian setiap tahun, di mana anak-anak balita merupakan mayoritas kematian akibat malaria di seluruh dunia. Menurut WHO, pada tahun 2016 malaria telah menyerang 91 negara, salah satunya adalah Indonesia. Angka pasien yang terinfeksi malaria di Indonesia sendiri pada tahun 2016 adalah 218.450 pasien, dimana 161 pasien diantaranya dinyatakan meninggal.

Pada penelitian ini penulis akan menggunakan model Convolutional Neural Network (CNN) dengan arsitektur VGG-16 untuk mengklasifikasikan citra sel darah merah, untuk menghasilkan nilai akurasi dan prediksi apakah sel darah terinfeksi malaria atau tidak. VGG-16 digunakan karena memiliki keunggulan yaitu tidak memerlukan mekanisme ekstraksi fitur dan tidak ada fase pemilihan fitur menengah, sehingga membuat VGG-16 sangat efektif untuk pengolahan citra karena fitur dapat muncul dimana saja pada citra (Kaur and Gandhi, 2019).  Selain itu arsitektur VGG-16 merupakan arsitektur CNN yang tingkat ketelitiannya cukup tinggi serta memiliki waktu pelatihan yang singkat. Arsitektur ini berhasil meraih peringkat teratas dalam localization dan classification. Sehingga diharapkan model yang dibuat nantinya dapat memproses klasifikasi dan deteksi pada sel darah merah yang terinfeksi malaria lebih cepat.

## Tujuan
1. Membuat klasifikasi penyakit malaria yang disebabkan oleh parasit yang menginfeksi sel darah melalui kumpulan data citra sel darah.
2. Membuat model klasifikasi dengan menggunakan model Transfer Learning yaitu VGG-16 untuk mendiagnosis malaria pada sel darah merah.
3. Mengetahui kemampuan dari Transfer Learning dalam melakukan klasifikasi pada jumlah dataset yang terbatas.

## Metode
![design eksperiment](https://raw.githubusercontent.com/RamdaniTarjianto/Diagnosis_Malaria_VGG16/main/image/alur%20gambar.png)

Tahap pertama dari penelitian ini adalah menyiapkan Malaria Cell Image Dataset yang berasal dari situs Kaggle dengan total data sebanyak 27.558 gambar, kemudian dataset gambar diubah menjadi bentuk Dataframe, lalu di dibagi menjadi data Training, data Testing dan data Validasi menggunakan train test split dengan perbandingan 90% Training, 5% Testing, 5% Validasi. Selanjutnya model VGG-16 dilatih, dengan menggunakan Optimizer RMSProp, Loss menggunakan categorical crossentropy, dan nilai matriks yang diukur pada penelitian ini menggunakan nilai accuracy sebagai nilai pengukurannya. dan juga kami mencoba membandingan kinerja model VGG16 dengan model CNN. dalam penelitian ini model evaluasi menggunakan confusion matriks untuk melihat nilai Presisi, Recall dan F1-score dalam setiap kelas yang diprediksi.

## Dataset
Penelitian ini menggunakan Malaria Cell Image Dataset yang berasal dari situs kaggle (Kaggle,2017). Data ini diambil dari 150 pasien yang terinfeksi P. falciparum dan 50 pasien sehat yang berasal dari Chittago Medical College Hospital, Bangladesh. Dataset ini terdiri dari 27.558 gambar sel darah merah yang diklasifikasikan menjadi dua kelompok yaitu infected dan uninfected yang jumlah datanya terbagi sama rata. 

| Total Number Of Data | Infected | Uninfected |
|----------|----------|----------|
| 27558, |13779 | 13779 |

![design eksperiment](https://raw.githubusercontent.com/RamdaniTarjianto/Diagnosis_Malaria_VGG16/main/image/gambar%20dataset.PNG)


Sel darah merah yang terinfeksi malaria (a) dan Sel darah merah normal (b)

## Data Preprocessing
Menggunakan Image Data Generator gambar di rescale dengan preprocessing function scalar untuk memastikan bahwa nilai pixel pada gambar di antara -1 dan 1. Demikian pula horizontal flip dipilih untuk memberikan lebih banyak variasi pada gambar yang membantu meningkatkan kualitas kumpulan data yang menghasilkan hasil yang lebih baik.  

## Model 
![model vgg](https://raw.githubusercontent.com/RamdaniTarjianto/Diagnosis_Malaria_VGG16/main/image/layers%20vgg16.png)
Dalam penelitian penulis, mencoba memodifikasi arsitektur dari model VGG-16. Umumnya, VGG-16 memiliki 3 Fully Connected Layer berurut pada bagian akhir layer sebagai penghubung unit aktivasi layer sebelumnya. Disini kami mencoba untuk membuat terapan baru dengan melakukan variasi pada bagian Fully Connected Layer. Kami menambahkan sebuah flatten layer setelah konvolusi layer pada bagian akhir model.  Flatten layer berfungsi untuk mengubah input data menjadi array 1 dimensi untuk dijadikan input ke layer berikutnya. Flatten layer juga berfungsi untuk menyamakan output dari konvolusi layer menjadi single long feature vector. Setelah itu, kami menambahkan sebuah dropout layer sebesar 0.2 di antara fully connected layer untuk mencegah overfitting pada data. 

## Eksperimen
Penelitian ini menggunakan model transfer learning, yaitu VGG-16, untuk mendeteksi malaria pada sel darah merah. Selain itu, kami juga membandingkan dengan berbagai model algoritma implementasi dari CNN, seperti basic CNN, Resnet, Inception, Xception, dan VGG-19. Penelitian ini juga menggunakan berbagai parameter optimalisasi untuk meningkatkan hasil akurasi, menekan angka loss, mencegah overfitting, serta membuat model menjadi efektif dan efisien. Berbagai parameter optimalisasi yang kami gunakan, diantaranya dropout, flatten, optimizer, loss function, metrics, dan lainnya.

penulis menggunakan optimizer RMSprop guna mengoptimalisasi model. Menggunakan Dropout sebesar 0.2, batch size sebesar 80, dan Flatten Layer, yang berfungsi untuk menaikan kecepatan pada proses training dan juga mencegah overfitting. Selain itu, penulis menggunakan aktivasi ReLU di setiap Convolution Layer yang diselingi oleh pooling layer serta aktivasi softmax pada bagian output layer. Kami juga menggunakan categorical crossentropy sebagai loss function pada model yang dikombinasikan dengan metrics accuracy. 
![eksperimen](https://raw.githubusercontent.com/RamdaniTarjianto/Diagnosis_Malaria_VGG16/main/image/hasil.PNG)

Pada pengambilan keputusan biasanya lebih mementingkan nilai Recall dan Precision yang tinggi, dimana hal ini menunjukkan bahwa algoritma memiliki nilai false positive dan false negatif yang lebih sedikit. Sedangkan F1-score yang merupakan Harmonic Mean dari nilai Recall dan Precision sehingga dapat digunakan sebagai penentu keputusan. Dapat dilihat pada tabel xx model VGG-16 yang penulis modifikasi mendapatkan nilai F1- Score yang lebih unggul dibandingkan dengan model lainnya, sedangkan model ResNet memiliki nilai F1-Score yang paling rendah dibandingkan model lainnya. 



