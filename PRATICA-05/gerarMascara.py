#1) usando espaço de cores e morfologia, utilize o filme de jogos e realize uma segmentação que resulte em gravacao de um filme apresentando apenas as mascaras, e outro, apresentando somente os jogadores do cruzeiro. No relatório, apresente fotos dos resultados desta segmentaçao por cor, e o script deve gerar o filme. Pode usar ou o filme que está no arquivo ZIP ou baixar um novo filme da internet à escolha de vcs.

import cv2
import numpy as np

# Parâmetros de cor para segmentação (azul em HSV)
lower_color = np.array([110, 125, 100])
upper_color = np.array([130, 255, 255])

#lower_color = np.array([95, 70, 75])
#upper_color = np.array([130, 255, 255])



# Função para processar cada frame
def process_frame(frame, lower_color, upper_color):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converter para HSV
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.inRange(hsv, lower_color, upper_color)  # Aplicar máscara de cor
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

# Carregar o vídeo
input_video_path = 'images/jogoCruzeiro.mp4'
cap = cv2.VideoCapture(input_video_path)

# Verificar se o vídeo foi carregado corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Obter as propriedades do vídeo de entrada
fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#apresentando apenas as mascaras, e outro, apresentando somente os jogadores do cruzeiro. 

# Definir o codec e criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
video_mascara = 'images/video_mascara.mp4'
video_jogadores = 'images/video_jogadores.mp4'
output_mascara = cv2.VideoWriter(video_mascara, fourcc, fps, (video_width, video_height), isColor=False)
output_jogadores = cv2.VideoWriter(video_jogadores, fourcc, fps, (video_width, video_height))


# Processar o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Processar o frame
    mask = process_frame(frame, lower_color, upper_color)

    # Bitwise AND
    jogadores = cv2.bitwise_and(frame, frame, mask=mask)

    # Escrever o frame no arquivo de vídeo
    output_mascara.write(mask)
    output_jogadores.write(jogadores)

    # Mostrar o frame 
    cv2.imshow('Frame original', frame)

    cv2.imshow('mask', mask)

    cv2.imshow('Frame Processado', jogadores)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# Liberar a captura e o escritor
cap.release()
output_mascara.release()
output_jogadores.release()
cv2.destroyAllWindows()

