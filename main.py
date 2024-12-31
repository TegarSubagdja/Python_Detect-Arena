import cv2
import numpy as np

def detect_and_crop_paper_landscape(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print("Gambar tidak ditemukan.")
        return
    
    original = image.copy()

    # Konversi ke grayscale dan threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Temukan kontur terbesar
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Tidak ada kontur yang terdeteksi.")
        return
    
    largest_contour = max(contours, key=cv2.contourArea)

    # Dapatkan minimum area rectangle untuk rotasi
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    angle = rect[-1]

    # Koreksi sudut rotasi
    if angle < -45:
        angle += 90

    # Rotasi gambar untuk meluruskan
    (h, w) = original.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(original, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Temukan bounding box pada gambar yang sudah diluruskan
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh_rotated = cv2.threshold(gray_rotated, 200, 255, cv2.THRESH_BINARY)
    contours_rotated, _ = cv2.findContours(thresh_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour_rotated = max(contours_rotated, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour_rotated)
    cropped = rotated[y:y+h, x:x+w]

    # Pastikan orientasi landscape
    if h > w:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    # Tampilkan hasil akhir
    cv2.imshow('Cropped Area (Landscape)', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'image.png'  # Ganti dengan path gambar Anda
    try:
        detect_and_crop_paper_landscape(image_path)
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    main()
