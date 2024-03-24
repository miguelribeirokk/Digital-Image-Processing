import cv2
import matplotlib.pyplot as plt

# implementando 5 metodos de binarização do opencv

import numpy as np
from collections import Counter

def otsu_threshold(image):
    # Inicializa as variáveis
    pixel_counts = Counter(image.flatten())
    pixel_intensities = np.array(list(pixel_counts.keys()))
    total_pixels = sum(pixel_counts.values())
    sum_intensities = np.sum(pixel_intensities * np.array([pixel_counts[i] for i in pixel_intensities]))
    max_variance = 0
    threshold = 0

    # Loop para calcular o limiar de Otsu
    for t in range(len(pixel_intensities)):
        # Calcula a probabilidade de cada classe (background e foreground)
        w1 = sum([pixel_counts[pixel_intensities[i]] for i in range(t)]) / total_pixels
        w2 = 1 - w1

        # Calcula as médias das intensidades de cada classe
        sum1 = sum([pixel_intensities[i] * pixel_counts[pixel_intensities[i]] for i in range(t)])
        mean1 = sum1 / (total_pixels * w1) if w1 != 0 else 0
        sum2 = sum_intensities - sum1
        mean2 = sum2 / (total_pixels * w2) if w2 != 0 else 0

        # Calcula a variância interclasse
        variance = w1 * w2 * ((mean1 - mean2) ** 2)

        # Atualiza o valor do limiar se a variância for maior que a anterior
        if variance > max_variance:
            max_variance = variance
            threshold = pixel_intensities[t]

    return threshold



threshold = 0
max_value = 255

image = cv2.imread("lena.jpg", 0)
threshold = otsu_threshold(image)
print(threshold)
cv2.imshow('entrada', image)

# Define os elementos menores que 100 como zero
image[image < threshold] = 0
image[image >= threshold] = 255


# Imprime a matriz resultante

cv2.imshow("saida", image)
cv2.waitKey(0)




