import numpy as np
import cv2

def otsu_threshold(image_array):
    # 计算每个灰度级别的像素数，返回一个长度为256的数组，每个元素表示对应灰度级别的像素数量
    pixel_counts = np.bincount(image_array.ravel(), minlength=256)
    total_pixels = image_array.size  # 获取图像的总像素数

    # 计算所有像素的灰度值总和
    sum_total = np.dot(np.arange(256), pixel_counts)
    sum_background = 0  # 背景像素灰度值总和
    weight_background = 0  # 背景像素数
    weight_foreground = 0  # 前景像素数
    max_variance = 0  # 最大类间方差
    threshold = 0  # 最佳阈值

    # 遍历所有可能的阈值（0到255）
    for i in range(256):
        weight_background += pixel_counts[i]  # 累加当前灰度级别的像素数到背景像素数
        if weight_background == 0:  # 如果背景像素数为0，跳过当前循环
            continue

        weight_foreground = total_pixels - weight_background  # 计算前景像素数
        if weight_foreground == 0:  # 如果前景像素数为0，结束循环
            break

        sum_background += i * pixel_counts[i]  # 累加当前灰度级别的灰度值到背景灰度值总和
        mean_background = sum_background / weight_background  # 计算背景的平均灰度值
        mean_foreground = (sum_total - sum_background) / weight_foreground  # 计算前景的平均灰度值

        # 计算类间方差
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # 如果当前类间方差大于最大类间方差，则更新最大类间方差和最佳阈值
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = i

    return threshold  # 返回最佳阈值

# 示例用法
if __name__ == "__main__":
    # 定义图像路径
    image_path = r'D:\desktop\documents\computervision\objectsegmentation\image.png'  # 将路径替换为你的图像路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

    if image is None:  # 检查图像是否成功读取
        raise FileNotFoundError(f"在路径 {image_path} 找不到图像")  # 如果读取失败，抛出文件未找到错误

    threshold = otsu_threshold(image)  # 计算OTSU阈值
    print(f"OTSU阈值: {threshold}")  # 输出OTSU阈值

    # 应用OTSU阈值进行二值化，返回二值化后的图像
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # 保存二值化后的图像为binary_image.jpg
    cv2.imwrite("binary_image.jpg", binary_image)
    # 显示二值化后的图像
    cv2.imshow("Binary Image", binary_image)
    # 等待按键按下，若不调用此函数则窗口会立即关闭
    cv2.waitKey(0)
    # 销毁所有窗口
    cv2.destroyAllWindows()
