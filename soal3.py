import cv2
import numpy as np
import matplotlib.pyplot as plt

def hitung_histogram(gambar):
    """
    Menghitung histogram dari sebuah citra grayscale.

    Parameter:
        gambar (np.ndarray): Array 2D yang berisi nilai piksel citra grayscale (0-255).

    Return:
        histogram (np.ndarray): Array 1D dengan ukuran 256 yang merepresentasikan histogram citra.
    """
    # Pastikan gambar adalah array numpy
    if not isinstance(gambar, np.ndarray):
        raise TypeError("Input gambar harus berupa numpy array.")

    # Validasi apakah semua nilai berada di rentang 0-255
    if gambar.min() < 0 or gambar.max() > 255:
        raise ValueError("Nilai piksel harus berada di rentang 0-255.")

    # Inisialisasi array histogram dengan 256 elemen bernilai 0
    histogram = np.zeros(256, dtype=int)

    # Iterasi melalui setiap nilai piksel untuk menghitung frekuensi
    for nilai in gambar.ravel():
        histogram[nilai] += 1

    return histogram

# Buka kamera dan ambil satu gambar
if __name__ == "__main__":
    # Buka kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        exit()

    # Baca satu frame dari kamera
    ret, frame = cap.read()

    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
    else:
        # Ubah frame menjadi grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hitung histogram
        histogram = hitung_histogram(gray_frame)

        # Tampilkan frame grayscale
        cv2.imshow('Grayscale Frame', gray_frame)

        # Simpan citra grayscale
        cv2.imwrite('grayscale_image.png', gray_frame)

        # Cetak histogram
        print("Histogram:")
        for i, frekuensi in enumerate(histogram):
            if frekuensi > 0:
                print(f"Nilai piksel {i}: {frekuensi} kali")

        # Simpan histogram sebagai grafik
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), histogram, color='gray')
        plt.title('Histogram Citra Grayscale')
        plt.xlabel('Nilai Piksel')
        plt.ylabel('Frekuensi')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('histogram.png')
        print("Histogram disimpan sebagai 'histogram.png' dan citra disimpan sebagai 'grayscale_image.png'.")

        # Tampilkan grafik histogram
        plt.show()

        # Tunggu hingga pengguna menekan tombol apa saja untuk keluar
        print("Tekan tombol apa saja pada jendela gambar untuk keluar.")
        cv2.waitKey(0)

    # Tutup kamera dan jendela
    cap.release()
    cv2.destroyAllWindows()
