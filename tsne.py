import os
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

# 指定图像路径
image_path = '/home/lyf/FedICRA/data/ODOC/Domain4/train/imgs/V0004.png'

# 打开并获取图像尺寸
try:
    with Image.open(image_path) as img:
        width, height = img.size
        print(f"The dimensions of the image are {width}x{height}.")
except Exception as e:
    print(f"Error opening the image: {e}")

def load_and_preprocess_image(file_path, size=img.size):
    try:
        # 加载并调整图像到指定大小
        img = Image.open(file_path)
        img = img.resize(size)
        img_array = np.array(img)
        # 展平图像，并标准化RGB值
        return img_array.flatten() / 255.0
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None

# 定义路径
base_path = '/home/lyf/FedICRA/data/ODOC'
domains = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5']

data = []
labels = []

for i, domain in enumerate(domains):
    domain_path = os.path.join(base_path, domain, 'test', 'imgs')
    for img_file in os.listdir(domain_path):
        image_path = os.path.join(domain_path, img_file)
        if os.path.isfile(image_path):
            # 加载和处理每个图像
            img_vector = load_and_preprocess_image(image_path)
            if img_vector is not None:
                data.append(img_vector)
                labels.append(i)

# 确保数据已成功加载
if not data:
    print("No image data loaded. Please check the file paths and formats.")
else:
    # 转换数据为numpy数组以进行t-SNE
    data = np.array(data)

    try:
        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(data)

        # 可视化结果并保存图片
        plt.figure(figsize=(10, 8))
        for i in range(len(domains)):
            plt.scatter(X_embedded[np.array(labels) == i, 0], X_embedded[np.array(labels) == i, 1], label=f'Domain {i+1}')
        plt.title("t-SNE of ODOC Domains")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()

        # 保存结果图片
        plt.savefig('/home/lyf/FedICRA/data/ODOC/tsne_plot.png')  
        print("t-SNE plot saved successfully!")

    except Exception as e:
        print(f"Error performing t-SNE: {e}")
