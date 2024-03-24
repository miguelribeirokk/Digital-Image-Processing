import cv2
import numpy as np

def niblack_threshold(img, window_size=15, k=-0.2):
    # Calcula a média local
    mean = cv2.blur(img, (window_size, window_size))
    
    # Calcula a média do quadrado local
    mean_square = cv2.blur(img**2, (window_size, window_size))
    
    # Calcula o desvio padrão local
    std_dev = np.sqrt(mean_square - mean**2)
    
    # Calcula o limiar de Niblack
    threshold = mean + k * std_dev
    
    # Aplica o limiar de Niblack
    binary = img > threshold
    
    return binary.astype(np.uint8) * 255

# Carrega a imagem
img = cv2.imread('imagem1.png', cv2.IMREAD_GRAYSCALE)

# Aplica o limiar de Niblack
thresholded_img = niblack_threshold(img)

# Exibe a imagem original e a imagem limiarizada
cv2.imshow(' Imagem Original', img)
cv2.imshow('Imagem limiarizada com Algoritmo Niblack ', thresholded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
