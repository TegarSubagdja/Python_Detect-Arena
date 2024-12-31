import cv2
import numpy as np

def detect_paper_edges(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    original = image.copy()
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplikasikan threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Temukan kontur pada gambar threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar hasil dengan batas
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Gambar batas dengan warna hijau
    
    # Tampilkan semua hasil
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Hasil Deteksi Batas', result)
    
    # Tunggu sampai tombol keyboard ditekan
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'image.png'  # Ganti dengan path gambar Anda
    try:
        detect_paper_edges(image_path)
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()