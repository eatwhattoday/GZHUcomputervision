"""


import cv2
import numpy as np

# 图像路径
image_path = r'D:\desktop\documents\computervision\project\test.jpg'

# 读取图像
image = cv2.imread(image_path)

# 检查图像是否正确加载
if image is not None:
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用中值滤波器去除椒盐噪声
    denoised = cv2.medianBlur(gray, 5)

    # 应用高斯模糊，减少图像噪声
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # 应用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 特征提取
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Contour Area: {area}, Perimeter: {perimeter}, Bounding Box: (x={x}, y={y}, w={w}, h={h})")

    # 显示边缘检测结果
    cv2.imshow('Edge Detection', edges)

    # 等待用户按键，再关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load image.")

"""



# 确保文件路径正确，没有多余的空格

import cv2
import numpy as np

# 图像路径
image_path = r'D:\desktop\documents\computervision\project\test.jpg'

# 读取图像
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    # 应用高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(image, (5, 5), 4)

    # 应用中值滤波器去除椒盐噪声
    denoised = cv2.medianBlur(blurred, 5)

    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(denoised, 100, 200)

    # 使用HoughCircles方法检测圆形
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=70, param2=28, minRadius=20, maxRadius=145)  # 我设置了minRadius和maxRadius的范围

    # 确保检测到圆形
    if circles is not None:
        # 创建一个黑色的空白图像，尺寸与原图像相同
        black_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

        # 遍历检测到的圆形，并在黑色图像上绘制
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (255, 192, 203), 4)
            cv2.rectangle(black_image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # 显示结果
        cv2.imshow('Detected Circles on Black Background', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles were found.")
else:
    print(f"Error: Unable to load image from {image_path}")


