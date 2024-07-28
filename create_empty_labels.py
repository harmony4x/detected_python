import os

# Đường dẫn đến thư mục chứa hình ảnh không có lửa
non_fire_images_dir = 'dataset/images_download/non_fire_images'
labels_dir = 'dataset/labels/train/non_fire_images'

# Tạo thư mục labels nếu chưa tồn tại
os.makedirs(labels_dir, exist_ok=True)

# Tạo file nhãn rỗng
for filename in os.listdir(non_fire_images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp"):
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
        open(label_path, 'w').close()
