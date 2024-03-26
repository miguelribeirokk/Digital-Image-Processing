import cv2
import numpy as np

# Carregar o vídeo
video_path = 'cars2.mp4'
cap = cv2.VideoCapture(video_path)

# Função de limiarização de Wellner
def wellner_threshold(image, window_size=15):
    threshold = cv2.blur(image, (window_size, window_size))
    binary_img = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary_img

# Ler o primeiro frame
ret, frame_prev = cap.read()
gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

while True:
    # Ler o próximo frame
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular a diferença entre os frames consecutivos
    diff = cv2.absdiff(gray, gray_prev)

    # Aplicar a limiarização de Wellner
    wellner_binary = wellner_threshold(diff)

    # Exibir as imagens na tela
    cv2.imshow("Wellner", wellner_binary)

    # Atualizar o frame anterior
    gray_prev = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
