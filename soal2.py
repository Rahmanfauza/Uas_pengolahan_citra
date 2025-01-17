import cv2
import numpy as np
import os

def multi_thresholding_from_camera(output_folder="output_frames"):
    # Pastikan folder output ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Gagal membuka kamera. Pastikan kamera terhubung.")

    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
    else:
        # Langkah 1: Membaca dan Menyimpan Citra Asli
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_path = os.path.join(output_folder, "original_image.png")
        cv2.imwrite(original_path, frame)

        # Langkah 2: Mengekstrak dan Menyimpan Komponen Red
        red_channel = frame[:, :, 2]  # Komponen Red ada di indeks 2 dalam format BGR
        red_channel_path = os.path.join(output_folder, "red_channel.png")
        cv2.imwrite(red_channel_path, red_channel)

        # Langkah 3: Multi Thresholding pada Komponen Red
        threshold_1 = 100
        threshold_2 = 240

        # Inisialisasi region berdasarkan nilai intensitas piksel
        region_1 = (red_channel <= threshold_1).astype(np.uint8) * 1
        region_2 = ((red_channel > threshold_1) & (red_channel <= threshold_2)).astype(np.uint8) * 2
        region_3 = (red_channel > threshold_2).astype(np.uint8) * 3

        # Gabungkan semua region
        multi_threshold_result = region_1 + region_2 + region_3

        # Visualisasi hasil thresholding dengan colormap
        result_colored = cv2.applyColorMap((multi_threshold_result * 85).astype(np.uint8), cv2.COLORMAP_JET)
        threshold_path = os.path.join(output_folder, "multi_threshold_result.png")
        cv2.imwrite(threshold_path, result_colored)

        # Tampilkan hasil
        cv2.imshow('Citra Asli', frame)
        cv2.imshow('Komponen Red', red_channel)
        cv2.imshow('Hasil Multi Thresholding', result_colored)

        # Tunggu hingga tombol ditekan untuk menutup jendela
        cv2.waitKey(0)

    # Tutup kamera dan jendela
    cap.release()
    cv2.destroyAllWindows()

# Panggil fungsi
multi_thresholding_from_camera()
