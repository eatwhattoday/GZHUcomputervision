import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # 使用 imageio.v2 以避免弃用警告

# 加载图像
image = imageio.imread(r'D:\desktop\documents\computervision\objectsegmentation\image.png')

# 计算直方图
histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))

# 绘制直方图
plt.figure()
plt.title('Histogram of the Image')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.xlim([0, 255])  # 设置x轴的范围
plt.plot(bin_edges[0:-1], histogram)  # 绘制直方图
plt.show()

# 目测直方图形状并设定阈值
# 假设我们通过直方图目测决定阈值为128
threshold = 128

# 二值化处理
binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)

# 将二值化后的数组转换为PIL图像
from PIL import Image
binary_image_pil = Image.fromarray(binary_image)

# 显示二值化后的图像
binary_image_pil.show()

# 保存二值化后的图像
binary_image_pil.save('binary_image.png')  # 保存为PNG格式，因为二值图没有透明度信