import os
import cv2

# Đường dẫn đến thư mục chứa hình ảnh tải xuống
input_dir_fire = 'dataset/images_download/fire_images'
input_dir_non_fire = 'dataset/images_download/non_fire_images'
output_dir_fire = 'dataset/images/train/fire_images'
output_dir_non_fire = 'dataset/images/train/non_fire_images'

# Tạo các thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_dir_fire, exist_ok=True)
os.makedirs(output_dir_non_fire, exist_ok=True)

def process_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".webp") or filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Chuyển đổi kích thước hình ảnh về 640x640
                img_resized = cv2.resize(img, (640, 640))
                # Lưu hình ảnh dưới định dạng .jpg
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
                cv2.imwrite(output_path, img_resized)

# Xử lý hình ảnh chứa lửa
process_images(input_dir_fire, output_dir_fire)
# Xử lý hình ảnh không chứa lửa
process_images(input_dir_non_fire, output_dir_non_fire)
