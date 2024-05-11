
# project  UTS PCD

Teori tambahan:
OpenCV (Open Source Computer Vision Library) adalah pustaka perangkat lunak open-source yang dirancang untuk pengolahan gambar dan komputer vision. OpenCV menyediakan berbagai fungsi dan algoritma untuk membaca, menulis, dan memanipulasi gambar, serta untuk melakukan berbagai tugas dalam bidang visi komputer seperti deteksi objek, pengenalan pola, dan segmentasi gambar.

Histogram Warna Histogram warna adalah representasi visual dari distribusi frekuensi intensitas warna dalam sebuah gambar. Histogram warna dapat membantu dalam menganalisis dan memahami distribusi warna dalam gambar, serta dalam menentukan batas warna untuk segmentasi warna.




import cv2 //
import matplotlib.pyplot as plt
import numpy as np

pic = cv2.imread("herawatii.jpg") // digunakan untuk membaca gambar dengan nama file "herawatii.jpg"  dengan menggunakan OpenCV.

pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB) // digunakan untuk mengubah skema warna gambar dari BGR menjadi RGB.

biru = pic[:, :, 0]
hijau = pic[:, :, 1]
merah = pic[:, :, 2] //digunakan untuk mengambil masing-masing saluran warna (biru, hijau, dan merah) dari gambar yang telah dimuat dan diubah menjadi skema warna RGB sebelumnya. 

plt.figure(figsize=(15, 5)) // digunakan untuk membuat sebuah gambar  dengan ukuran tertentu sebelum melakukan plot menggunakan matplotlib.

plt.subplot(2, 2, 1) // digunakan untuk membuat subplot di dalam gambar yang telah dibuat.
plt.imshow(pic) // digunakan untuk menampilkan gambar.
plt.title('citra asli') // digunakan untuk menambahkan judul pada subplot 
plt.axis('off') // digunakan untuk menghilangkan sumbu (axis) dari plot atau subplot.
plt.show() //  digunakan untuk menampilkan semua plot atau subplot.

plt.figure(figsize=(15, 5)) //  Ini membuat sebuah gambar baru dengan ukuran 15x5
plt.subplot(2, 2, 2) // membuat subplot kedua di dalam gambar. Subplot ini berada di posisi kanan atas dari grid 2x2 yang telah dibuat.
plt.imshow(biru, cmap='gray') //  menampilkan array biru sebagai gambar dalam subplot yang sedang aktif. Argumen cmap='gray' digunakan untuk menentukan colormap yang digunakan untuk menampilkan gambar
plt.title('warna Biru') // menambahkan judul "warna Biru" 
plt.axis('off') // menghilangkan sumbu dari subplot
plt.show() //  menampilkan gambar lengkap yang telah dibuat 

plt.figure(figsize=(15, 5)) //  membuat sebuah gambar baru dengan ukuran 15x5 
plt.subplot(2, 2, 3) // membuat subplot ketiga di dalam gambar
plt.imshow(hijau, cmap='gray') // menampilkan array hijau sebagai gambar dalam subplot
plt.title('warna hijau') // menambahkan judul "warna hijau" 
plt.axis('off') // menghilangkan sumbu dari subplot
plt.show() // menampilkan gambar lengkap yang telah dibuat 

plt.figure(figsize=(15, 5)) // membuat sebuah gambar baru dengan ukuran 15x5 
plt.subplot(2, 2, 4) // membuat subplot ketiga di dalam gambar
plt.imshow(merah, cmap='gray') //  menampilkan array merah sebagai gambar dalam subplot
plt.title('warna merah') // menambahkan judul "warna merah"
plt.axis('off') // menghilangkan sumbu dari subplot
plt.show() //  menampilkan gambar lengkap yang telah dibuat 

fig, axs = plt.subplots(2,2, figsize=(20,6)) // membuat sebuah gambar dengan empat subplot yang memiliki grid 2x2
axs[0, 0].hist(pic.ravel(),256,[0,256]) // menambahkan histogram untuk gambar asli dibaris pertama, kolom pertama dari grid. Fungsi ravel() digunakan untuk mengubah array gambar menjadi satu dimensi. 256 menunjukkan jumlah bin yang digunakan untuk histogram, dan [0, 256] menunjukkan rentang nilai yang dihitung oleh histogram.
axs[0, 0].set_title('Histogram Citra Asli') // menambahkan judul "Histogram Citra Asli" 

axs[0, 1].hist(biru.ravel(), 256, [0, 256], color='blue') // menambahkan histogram untuk saluran warna biru di posisi baris pertama, kolom kedua dari grid. Histogram ini diplot dengan warna biru.
axs[0, 1].set_title('Histogram citra Biru') // menambahkan judul "Histogram citra Biru"

axs[1, 0].hist(hijau.ravel(), 256, [0, 256], color='green') // menambahkan histogram untuk saluran warna hijau di posisi baris kedua, kolom pertama dari grid. 

axs[1, 0].set_title('Histogram citra Hijau') // menambahkan judul "Histogram citra Hijau"

axs[1, 1].hist(merah.ravel(), 256, [0, 256], color='red') // menambahkan histogram untuk saluran warna merah di posisi baris kedua, kolom kedua dari grid. Histogram ini diplot dengan warna merah.
axs[1, 1].set_title('Histogram citra Merah') // menambahkan judul "Histogram citra Merah"
plt.show() // menampilkan gambar lengkap yang telah dibuat 

hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB) // untuk mengubah gambar dari skema warna BGR menjadi skema warna RGB.
hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV) //  untuk mengubah gambar ke RGB.

biru_bawah = np.array([100, 43, 46])
biru_atas = np.array([130, 255, 255])

merah_bawah = np.array([160, 43, 46])
merah_atas = np.array([180, 255, 255])

hijau_bawah = np.array([36, 43, 46])
hijau_atas = np.array([70, 255, 255])

biru_mask = cv2.inRange(hsv, biru_bawah, biru_atas)
merah_mask = cv2.inRange(hsv, merah_bawah, merah_atas)
hijau_mask = cv2.inRange(hsv, hijau_bawah, hijau_atas)

biru = cv2.bitwise_and(pic, pic, mask=biru_mask)
merah = cv2.bitwise_and(pic, pic, mask=merah_mask)
hijau = cv2.bitwise_and(pic, pic, mask=hijau_mask)

#menampilkan gambar asli dan gambar yang dideteksi dengan menggunakan matplotlib
plt.figure(figsize=(10, 5))

# Gambar Asli
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
plt.title('Gambar Asli')
plt.axis('off')

# Gambar Biru
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(biru, cv2.COLOR_BGR2RGB))
plt.title('Gambar Biru')
plt.axis('off')

# Gambar Hijau
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(hijau, cv2.COLOR_BGR2RGB))
plt.title('Gambar Hijau')
plt.axis('off')
# Gambar Merah
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(merah, cv2.COLOR_BGR2RGB))
plt.title('Gambar Merah')
plt.axis('off')

plt.tight_layout()
plt.show()
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)


## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Used By

This project is used by the following companies:

- Company 1
- Company 2


## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```


## Tech Stack

**Client:** React, Redux, TailwindCSS

**Server:** Node, Express


## Support

For support, email fake@fake.com or join our Slack channel.

