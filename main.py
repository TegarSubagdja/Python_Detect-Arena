import cv2
import numpy as np

def detect_and_crop_paper(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    original = image.copy()
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplikasikan threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Temukan kontur pada gambar threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pilih kontur dengan area terbesar
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Temukan bounding box dari kontur terbesar
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Potong area berdasarkan bounding box
        cropped = original[y:y+h, x:x+w]
        
        # Gambar bounding box pada gambar asli (opsional, untuk debugging)
        result = original.copy()
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Tampilkan hasil
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Bounding Box', result)
        cv2.imshow('Cropped Area', cropped)
        
        # Tunggu sampai tombol keyboard ditekan
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Tidak ada kontur yang terdeteksi.")

def main():
    image_path = 'image.png'  # Ganti dengan path gambar Anda
    try:
        detect_and_crop_paper(image_path)
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()
