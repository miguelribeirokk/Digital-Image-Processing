import cv2
import numpy as np

# Carregar o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

# Funções de limiarização
def niblack_threshold(image, block_size=255, k=-0.2):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, k)

def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 20)

def gaussian_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 20)

# Ler o primeiro frame
ret, frame_prev = cap.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

while True:
    # Ler o próximo frame
    ret, frame = cap.read()
    if not ret:
        break

    frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])  # aplica equalização no canal V

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular a diferença entre os frames consecutivos
    diff = cv2.absdiff(gray, gray_prev)

    # Aplicar as diferentes técnicas de limiarização
    niblack_binary = niblack_threshold(diff)
    adaptive_binary = adaptive_threshold(diff)
    gaussian_binary = gaussian_threshold(diff)
    _, otsu_binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Exibir as imagens na tela
    cv2.imshow("Niblack", niblack_binary)
    cv2.imshow("Adaptive", adaptive_binary)
    cv2.imshow("Gaussian", gaussian_binary)
    cv2.imshow("Otsu", otsu_binary)

    # Atualizar o frame anterior
    gray_prev = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
