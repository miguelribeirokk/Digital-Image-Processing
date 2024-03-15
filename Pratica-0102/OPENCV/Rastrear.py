import cv2
import numpy as np

# Lista para armazenar as coordenadas do centro do objeto nas últimas N frames
last_positions = []

# Número de frames anteriores a serem considerados para rastrear o movimento do objeto
NUM_LAST_POSITIONS = 100

# Função para rastrear o objeto azul na imagem
def track_blue_object(frame):
    global last_positions

    # Convertendo a imagem de BGR para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definindo o intervalo de cor para o azul na escala HSV
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Criando uma máscara para isolar a cor azul na imagem
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplicando operações morfológicas para melhorar a máscara
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Aplicando a máscara ao frame original
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Encontrando os contornos do objeto azul
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se houver contornos encontrados
    if contours:
        # Encontrando o maior contorno
        max_contour = max(contours, key=cv2.contourArea)

        # Calculando o centro do contorno
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Adicionando as coordenadas do centro do objeto à lista de últimas posições
        last_positions.append((cx, cy))

        # Mantendo apenas as últimas N posições na lista
        if len(last_positions) > NUM_LAST_POSITIONS:
            last_positions = last_positions[-NUM_LAST_POSITIONS:]

        # Desenhando um círculo no centro do objeto rastreado
        cv2.circle(masked_frame, (cx, cy), 10, (0, 255, 0), -1)

        # Desenhando o rastro do objeto
        for i in range(1, len(last_positions)):
            cv2.line(masked_frame, last_positions[i - 1], last_positions[i], (0, 255, 0), 2)

    return masked_frame

# Função principal
def main():
    # Capturando o feed da webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Lendo o frame da webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Invertendo horizontalmente o frame
        frame = cv2.flip(frame, 1)

        # Rastreando o objeto azul
        tracked_frame = track_blue_object(frame)

        # Exibindo o frame rastreado
        cv2.imshow('Tracked Frame', tracked_frame)

        # Saindo do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberando os recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
