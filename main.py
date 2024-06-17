import cv2
import numpy as np

# 寻找二值图像中的起始点
def find_start_point(binary_image):
    for i in range(len(binary_image)):  # 遍历图像的每一行
        if 1 in binary_image[i]:  # 如果当前行中有值为1的像素
            for j in range(len(binary_image[i])):  # 遍历当前行的每一个像素
                if binary_image[i][j] == 1:  # 如果当前像素的值为1
                    return i, j  # 返回起始点的坐标
    return None  # 如果没有找到值为1的像素，则返回None

# 跟踪并记录轮廓的点
def track_contour(binary_image, start_row, start_col):
    points = [(start_row, start_col)]  # 初始化点列表，包含起始点
    current_row, current_col = start_row, start_col  # 当前点坐标
    dir = 0  # 初始化方向码

    while True:
        next_point = None  # 下一个点初始化为None
        # 遍历当前点周围的8个方向
        for offset in [(dir - 1) % 8, (dir - 2) % 8, dir, (dir + 1) % 8, (dir + 2) % 8]:
            # 计算下一个点的坐标
            row = current_row + (offset // 4) - (offset % 4 // 2)
            col = current_col + (offset % 4) - (offset // 4 // 2)
            # 如果下一个点在图像范围内且值为1
            if 0 <= row < len(binary_image) and 0 <= col < len(binary_image[0]) and binary_image[row][col] == 1:
                next_point = (row, col)
                break  # 找到下一个点，跳出循环

        if next_point is None:  # 如果没有找到下一个边界点，则结束跟踪
            break

        # 更新当前点和方向码
        points.append(next_point)
        current_row, current_col = next_point
        # 更新方向码，根据当前方向码和移动方向
        dir = (dir + (1 if dir % 2 == 0 else 2)) % 8

        # 检查是否完成轮廓跟踪
        if (current_row, current_col) == points[1] and (current_row, current_col) == points[-2]:
            break

    return points

# 绘制轮廓的函数，添加了thickness和lineType参数
def draw_contours(image, contours, colors, thickness=1, lineType=cv2.LINE_4):
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]  # 为每个轮廓选择颜色
        cv2.drawContours(image, [contour], 0, color, thickness, lineType)  # 绘制轮廓

# 图片路径
image_path = r'D:\desktop\documents\computervision\pythonProject\test1.png'  # 替换为您的图片路径

# 读取图片
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

# 自动阈值二值化
_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 寻找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制细轮廓
contour_img = img.copy()  # 创建一个副本来绘制轮廓
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # 轮廓颜色
draw_contours(contour_img, contours, colors, thickness=2, lineType=cv2.LINE_4)  # 绘制轮廓

# 显示图像
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 假设contour是找到的轮廓之一
contour = contours[0]
# 计算周长
perimeter = cv2.arcLength(contour, True)
# 近似轮廓以简化计算
approx_curvature = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
# 计算面积
area = cv2.contourArea(contour)
# 计算圆形度
circularity = (4 * np.pi * (area / (perimeter ** 2))) if perimeter != 0 else 0
# 计算矩形度
# 假设外接矩形的长和宽分别为width和height
x, y, w, h = cv2.boundingRect(contour)
AR = w * h
rectangularity = area / AR if AR != 0 else 0
# 打印结果
print(f"Perimeter: {perimeter}")
print(f"Area: {area}")
print(f"Circularity: {circularity}")
print(f"Rectangularity: {rectangularity}")S