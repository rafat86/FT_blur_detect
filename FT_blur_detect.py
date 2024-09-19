# Machine Vision #
# Assignment 05 #
# Simple blur detection using Fourier Transform #
# Ra'fat Naserdeen #

import cv2
import numpy as np

energy_threshold = 4500  # average energy threshold
D0 = 30                  # diameter around the center of the image

# Load Image as a grey scale image
image = cv2.imread('blur5.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a 2D Fourier Transform to the image
ft_image = np.fft.fft2(image)
shifted_image = np.fft.fftshift(ft_image)

# Remove the low-frequency components by setting them to zero around D0 distance from center
M, N = image.shape                      # Determine the number of rows and columns for the image
H = np.zeros((M, N), dtype=np.float32)  # creat a new image_array to store frequencies

for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
        if D <= D0:
            H[u, v] = 0
        else:
            H[u, v] = np.abs(shifted_image[u, v])

# Calculate the energy in the remaining high-frequency elements
hf_energy = np.sum(np.abs(H))

# Calculate the average energy in the remaining high-frequency elements
average_energy = hf_energy / (M*N)
print("Total remaining energy for high frequency elements: ", hf_energy)
print("Average energy for the image                      : ", average_energy)

# Compare the energy to a threshold to determine if the image is blurry or not
if average_energy < energy_threshold:
    print("\033[94mThis image is Blurred\033[0m")
else:
    print("\033[91mThis image is Not-Blurred\033[0m")
