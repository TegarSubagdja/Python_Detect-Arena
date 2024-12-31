import cv2
import numpy as np

def detect_and_crop_paper_landscape(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    original = image.copy()
    
    # Tampilkan gambar asli
    cv2.imshow('Original Image', original)
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)
    
    # Aplikasikan threshold
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('Threshold Image', thresh)
    
    # Temukan kontur pada gambar threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pilih kontur dengan area terbesar
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Gambar kontur terbesar pada gambar asli
        contour_drawn = original.copy()
        cv2.drawContours(contour_drawn, [largest_contour], -1, (0, 255, 0), 2)
        cv2.imshow('Largest Contour', contour_drawn)
        
        # Dapatkan minimum area rectangle untuk sudut rotasi
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        angle = rect[-1]  # Sudut rotasi
        
        # Gambar rectangle pada gambar asli
        rectangle_drawn = original.copy()
        cv2.drawContours(rectangle_drawn, [box], -1, (255, 0, 0), 2)
        cv2.imshow('Rotated Rectangle', rectangle_drawn)
        
        # Perbaiki sudut jika diperlukan
        if angle < -45:
            angle += 90
        
        # Rotasi gambar untuk meluruskan
        (h, w) = original.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(original, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imshow('Rotated Image', rotated)
        
        # Temukan bounding box lagi setelah rotasi
        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, thresh_rotated = cv2.threshold(gray_rotated, 200, 255, cv2.THRESH_BINARY)
        contours_rotated, _ = cv2.findContours(thresh_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour_rotated = max(contours_rotated, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour_rotated)
        cropped = rotated[y:y+h, x:x+w]
        
        # Tampilkan bounding box baru
        result_with_box = rotated.copy()
        cv2.rectangle(result_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Bounding Box After Rotation', result_with_box)
        
        # Pastikan orientasi landscape
        if h > w:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        
        # Tampilkan hasil akhir
        cv2.imshow('Cropped Area (Landscape)', cropped)
        
        # Tunggu sampai tombol keyboard ditekan
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Tidak ada kontur yang terdeteksi.")

def main():
    image_path = 'image.png'  # Ganti dengan path gambar Anda
    try:
        detect_and_crop_paper_landscape(image_path)
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()
