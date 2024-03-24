import cv2
import numpy as np

def multi_level_otsu_thresholding(img, num_thresholds):
    # Calcula o histograma da imagem
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    
    # Calcula as probabilidades para cada intensidade de pixel
    hist_norm = hist / np.sum(hist)
    
    # Inicializa o vetor de limiares
    thresholds = np.zeros(num_thresholds, dtype=np.uint8)
    
    # Calcula os limiares usando o algoritmo de Otsu
    for i in range(num_thresholds):
        # Calcula a variância interclasse para cada intensidade de pixel
        interclass_variances = np.zeros_like(hist_norm)
        for t in range(1, len(hist_norm)):
            w0 = np.sum(hist_norm[:t])
            w1 = np.sum(hist_norm[t:])
            mean0 = np.sum(np.arange(t) * hist_norm[:t]) / max(1e-5, w0)
            mean1 = np.sum(np.arange(t, len(hist_norm)) * hist_norm[t:]) / max(1e-5, w1)
            interclass_variances[t] = w0 * w1 * (mean0 - mean1) ** 2
        
        # Encontra o limiar que maximiza a variância interclasse
        thresholds[i] = np.argmax(interclass_variances)
        
        # Atualiza o histograma normalizado excluindo a região em torno do limiar selecionado
        hist_norm[max(0, thresholds[i] - 10):min(255, thresholds[i] + 10)] = 0
    
    return thresholds

# Carrega a imagem
img = cv2.imread('imagem1.png', cv2.IMREAD_GRAYSCALE)

# Define o número de limiares desejado
num_thresholds = 3

# Aplica a limiarização multinível de Otsu
thresholds = multi_level_otsu_thresholding(img, num_thresholds)

# Binariza a imagem usando os limiares obtidos
binary_images = [(img > threshold).astype(np.uint8) * 255 for threshold in thresholds]

# Exibe a imagem original e as imagens binarizadas
cv2.imshow('Original Image', img)
for i, binary_image in enumerate(binary_images):
    cv2.imshow(f'Binary Image {i+1}', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
