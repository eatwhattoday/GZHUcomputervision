import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

imgname1 = r'C:\Users\25317\Desktop\visual_test\test\ETH-MatchDataset\ETH-MatchDataset\Ref\SUCHARD1.tif'
img1 = cv.imread(imgname1)
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)

folder_path = r'C:\Users\25317\Desktop\visual_test\test\ETH-MatchDataset\ETH-MatchDataset\Ref'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]  # 假设所有图片都是.tif格式

bf = cv.BFMatcher()
top_matches = []  # 存储每张图片与img1的匹配得分
good = []

for image_file in image_files:
    if image_file == os.path.basename(imgname1):
        continue  # 跳过原始图片

    imgname2 = os.path.join(folder_path, image_file)
    img2 = cv.imread(imgname2)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子
    # sift.detectAndComputer(gray， None)计算出图像的关键点和sift特征向量   参数说明：gray表示输入的图片
    # des1表示sift特征向量，128维
    print("图片1的关键点数目：" + str(len(kp1)))
    # print(des1.shape)
    kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子
    print("图片2的关键点数目：" + str(len(kp2)))

    img3 = cv.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
    # img3 = cv2.drawKeypoints(gray, kp, img) 在图中画出关键点   参数说明：gray表示输入图片, kp表示关键点，img表示输出的图片
    # print(img3.size)
    img4 = cv.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈

    img5 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv.imshow("BFmatch", img5)
    cv.waitKey(0)
    cv.destroyAllWindows()

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # 计算匹配得分，这里简单地使用good匹配的数量作为得分
    score = len(good)
    top_matches.append((image_file, score))

# 按匹配得分排序
top_matches.sort(key=lambda x: x[1], reverse=True)

# 打印 top 10 最相似的图片
top_10 = top_matches[:10]
for i, (image_file, score) in enumerate(top_10):
    print(f"{i+1}. {image_file} with score {score}")

# 如果你想显示这些图片，可以添加以下代码
for i, (image_file, _) in enumerate(top_10):
    img_path = os.path.join(folder_path, image_file)
    img = cv.imread(img_path)
    cv.imshow(f"Top {i+1} - {image_file}", img)
    cv.waitKey(0)
cv.destroyAllWindows()