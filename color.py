import cv2
import numpy as np


def rgb_to_hsi(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        b, g, r = cv2.split(image)
        b, g, r = [i / 255.0 for i in (b, g, r)]

        # Compute Intensity
        I = (r + g + b) / 3.0

        # Compute Saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        S = 1 - (3 / (r + g + b + 1e-6)) * min_rgb

        # Compute Hue
        numerator = 0.5 * ((r - g) + (r - b))
        denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
        theta = np.arccos(numerator / (denominator + 1e-6))
        H = np.zeros_like(r)
        H[b > g] = 2 * np.pi - theta[b > g]
        H[g >= b] = theta[g >= b]
        H = H / (2 * np.pi)

        HSI = cv2.merge((H, S, I))
        return HSI


def extract_skin(image):
    HSI = rgb_to_hsi(image)
    H, S, I = cv2.split(HSI)

    # Define skin color range in HSI
    lower_H1 = 0.0
    upper_H1 = 60 / 360.0
    lower_H2 = 340 / 360.0
    upper_H2 = 1.0
    lower_S = 0.2
    upper_S = 0.65
    lower_I = 0.182
    upper_I = 0.9

    # Apply thresholds for H, S, I
    skin_mask = ((H >= lower_H1) & (H <= upper_H1) | (H >= lower_H2) & (H <= upper_H2)) & \
                (S >= lower_S) & (S <= upper_S) & \
                (I >= lower_I) & (I <= upper_I)
    skin_mask = skin_mask.astype(np.uint8) * 255

    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin, H, S, I


def main():
    # Step 1: Capture an image (for this code, we will read an image from a file)
    image = cv2.imread(r'D:\desktop\documents\computervision\color\1.jpg')  # replace with your image file

    # Step 2: Convert image to HSI color space
    HSI_image = rgb_to_hsi(image)
    H, S, I = cv2.split(HSI_image)

    # Normalize H, S, I for display
    H_display = (H * 255).astype(np.uint8)
    S_display = (S * 255).astype(np.uint8)
    I_display = (I * 255).astype(np.uint8)

    # Step 3: Extract skin regions based on HSI thresholds
    skin_image, H, S, I = extract_skin(image)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('HSI Image', HSI_image)
    cv2.imshow('Hue Channel', H_display)
    cv2.imshow('Saturation Channel', S_display)
    cv2.imshow('Intensity Channel', I_display)
    cv2.imshow('Skin Detected', skin_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
