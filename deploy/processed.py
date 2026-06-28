import cv2
import numpy as np
import torch

def preprocess_image(image_path, image_size):
    # đọc ảnh
    image = cv2.imread(image_path)

    # kiêm tra xem có đọc đc ảnh ko
    if image is None:
        raise ValueError("File tải lên không phải là ảnh hợp lệ.")

    # đổi kênh mầu ảnh
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # resize kích thước ảnh
    image = cv2.resize(image,(image_size, image_size))

    image = np.transpose(image, (2, 0, 1)) / 255.0

    image = image[None, :, :, :]

    image = torch.from_numpy(image).float()

    return image