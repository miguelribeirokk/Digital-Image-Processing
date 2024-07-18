#1) usando espaço de cores e morfologia, utilize o filme de jogos e realize uma segmentação que resulte em gravacao de um filme apresentando apenas as mascaras, e outro, apresentando somente os jogadores do cruzeiro. No relatório, apresente fotos dos resultados desta segmentaçao por cor, e o script deve gerar o filme. Pode usar ou o filme que está no arquivo ZIP ou baixar um novo filme da internet à escolha de vcs.

import cv2
import numpy as np

# Parâmetros de cor para segmentação (azul em BGR)
bgr_lower_color = np.array([100, 0, 0])
bgr_upper_color = np.array([255, 100, 120])


# Parâmetros de cor para segmentação (azul em HSV)
hsv_lower_color = np.array([110, 115, 100])
hsv_upper_color = np.array([130, 255, 255])


# Parâmetros de cor para segmentação (azul em hls)
hls_lower_color = np.array([110, 70, 70])  # Exemplo: Ajuste conforme necessário
hls_upper_color = np.array([140, 200, 255])  # Exemplo: Ajuste conforme necessário


# Parâmetros de cor para segmentação (azul em Lab)
lab_lower_color = np.array([20, 60, 0])
lab_upper_color = np.array([200, 210, 100])


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


# Definir o codec e criar o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
bgr_mascara = 'images/bgr_mascara.mp4'
bgr_jogadores = 'images/bgr_jogadores.mp4'
hsv_mascara = 'images/hsv_mascara.mp4'
hsv_jogadores = 'images/hsv_jogadores.mp4'
hls_mascara = 'images/hls_mascara.mp4'
hls_jogadores = 'images/hls_jogadores.mp4'
lab_mascara = 'images/lab_mascara.mp4'
lab_jogadores = 'images/lab_jogadores.mp4'


bgr_output_mascara = cv2.VideoWriter(bgr_mascara, fourcc, fps, (video_width, video_height), isColor=False)
bgr_output_jogadores = cv2.VideoWriter(bgr_jogadores, fourcc, fps, (video_width, video_height))
hsv_output_mascara = cv2.VideoWriter(hsv_mascara, fourcc, fps, (video_width, video_height), isColor=False)
hsv_output_jogadores = cv2.VideoWriter(hsv_jogadores, fourcc, fps, (video_width, video_height))
hls_output_mascara = cv2.VideoWriter(hls_mascara, fourcc, fps, (video_width, video_height), isColor=False)
hls_output_jogadores = cv2.VideoWriter(hls_jogadores, fourcc, fps, (video_width, video_height))
lab_output_mascara = cv2.VideoWriter(lab_mascara, fourcc, fps, (video_width, video_height), isColor=False)
lab_output_jogadores = cv2.VideoWriter(lab_jogadores, fourcc, fps, (video_width, video_height))


# Processar o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    cv2.imshow('frame_original', frame)

    bgr_mask = cv2.inRange(frame, bgr_lower_color, bgr_upper_color)  # Aplicar máscara de cor
    #cv2.imshow('mask_original', bgr_mask)
    bgr_mask = cv2.morphologyEx(bgr_mask, cv2.MORPH_CLOSE, kernel)
    bgr_mask = cv2.morphologyEx(bgr_mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('mask_operacoes', bgr_mask)
    bgr_mask_final = cv2.bitwise_and(frame, frame, mask=bgr_mask)
    cv2.imshow('bgr_mask_final', bgr_mask_final)
    # Escrever o frame no arquivo de vídeo
    bgr_output_mascara.write(bgr_mask)
    bgr_output_jogadores.write(bgr_mask_final)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converter para HSV
    hsv_mask = cv2.inRange(hsv, hsv_lower_color, hsv_upper_color)  # Aplicar máscara de cor
    #cv2.imshow('mask_original', hsv_mask)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('mask_operacoes', hsv_mask)
    hsv_mask_final = cv2.bitwise_and(frame, frame, mask=hsv_mask)
    cv2.imshow('hsv_mask_final', hsv_mask_final)
    # Escrever o frame no arquivo de vídeo
    hsv_output_mascara.write(hsv_mask)
    hsv_output_jogadores.write(hsv_mask_final)


    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)  # Converter para HLS
    hls_mask = cv2.inRange(hls, hls_lower_color, hls_upper_color)  # Aplicar máscara de cor
    #cv2.imshow('mask_original', hls_mask)
    hls_mask = cv2.morphologyEx(hls_mask, cv2.MORPH_CLOSE, kernel)
    hls_mask = cv2.morphologyEx(hls_mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('mask_operacoes', hls_mask)
    hls_mask_final = cv2.bitwise_and(frame, frame, mask=hls_mask)
    cv2.imshow('hls_mask_final', hls_mask_final)
    # Escrever o frame no arquivo de vídeo
    hls_output_mascara.write(hls_mask)
    hls_output_jogadores.write(hls_mask_final)


    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Converter para LAB
    lab_mask = cv2.inRange(lab, lab_lower_color, lab_upper_color)  # Aplicar máscara de cor
    #cv2.imshow('mask_original', lab_mask)
    lab_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE, kernel)
    lab_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('mask_operacoes', lab_mask)
    lab_mask_final = cv2.bitwise_and(frame, frame, mask=lab_mask)
    cv2.imshow('lab_mask_final', lab_mask_final)
    # Escrever o frame no arquivo de vídeo
    lab_output_mascara.write(lab_mask)
    lab_output_jogadores.write(lab_mask_final)



    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# Liberar a captura e o escritor
cap.release()
bgr_output_mascara.release()
bgr_output_jogadores.release()
hsv_output_mascara.release()
hsv_output_jogadores.release()
hls_output_mascara.release()
hls_output_jogadores.release()
lab_output_mascara.release()
lab_output_jogadores.release()
cv2.destroyAllWindows()

