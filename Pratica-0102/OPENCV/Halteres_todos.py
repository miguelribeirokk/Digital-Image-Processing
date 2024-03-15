import numpy as np
import cv2

def on_mouse(event, x, y, flags, param):
    global pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        matrix_pixels = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                blue, green, red = img[y + j, x + i]
                matrix_pixels.append([blue, green, red])
        matrix_pixels = np.array(matrix_pixels)
        pixels = np.append(pixels, matrix_pixels, axis=0)



def process_image(img, pixels, mediaB, mediaG, mediaR, tolerancia, window_size):
    rows, cols, _ = img.shape
    for i in range(rows):
        for j in range(cols):
            b, g, r = img[i, j]
            count = 0
            for m in range(max(0, i - window_size // 2), min(rows, i + window_size // 2 + 1)):
                for n in range(max(0, j - window_size // 2), min(cols, j + window_size // 2 + 1)):
                    b_neighbor, g_neighbor, r_neighbor = img[m, n]
                    if ((mediaB - tolerancia) < b_neighbor < (mediaB + tolerancia)) and \
                            ((mediaG - tolerancia) < g_neighbor < (mediaG + tolerancia)) and \
                            ((mediaR - tolerancia) < r_neighbor < (mediaR + tolerancia)):
                        count += 1
            if count >= (window_size * window_size) // 2:
                img[i, j] = (mediaB, mediaG, mediaR)
    return img



# Carrega a imagem
img = cv2.imread("imagens/ayrton-senna.jpg")
print("Imagem carregada")

# Cria uma janela para exibir a imagem
cv2.namedWindow('frame1')

# Cria uma matriz para armazenar os píxeis selecionados pelo usuário
pixels = np.zeros((1, 3), dtype=np.uint8)

# Define uma matriz para armazenar as médias de cada cor
medias_cores = np.zeros((3, 6), dtype=np.uint8)  # 3 linhas para B, G e R; 6 colunas para as 6 cores

# Define a função de callback para eventos de mouse
cv2.setMouseCallback("frame1", on_mouse)

conjunto_cor = ["Azul", "Preto", "Amarelo", "Vermelho", "Cinza", "Verde"]

# Loop para capturar as 6 cores
for cor in range(6):
    print(f"Selecione a cor: {conjunto_cor[cor]}")
    while True:
        cv2.imshow("frame1", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Remove a primeira linha da matriz de píxeis, inicializada com zeros
    pixels = np.delete(pixels, 0, axis=0)

    # Calcula a média dos valores BGR dos pixels selecionados
    media = np.mean(pixels, axis=0)
    print("Média: ", media)
    mediaB = media[0]
    mediaG = media[1]
    mediaR = media[2]

    # Armazena a média na matriz de médias de cores
    medias_cores[:, cor] = [mediaB, mediaG, mediaR]
    print("Média da cor: ", mediaB, mediaG, mediaR)
    print(medias_cores)

    # Esvazia a matriz de pixels
    pixels = np.zeros((1, 3), dtype=np.uint8)


# Define uma margem de tolerância para os valores de cor
tolerancia = 60

# Define o tamanho da janela ao redor de cada pixel para verificar os vizinhos
window_size = 5

# Processa a imagem para cada uma das 6 cores

# Processando para Azul:
print("Processando para Azul")
azul = process_image(img.copy(), pixels, medias_cores[0, 0], medias_cores[1, 0], medias_cores[2, 0], tolerancia, window_size)
cv2.imshow("Azul", azul)
cv2.waitKey(0)

# Processando para Preto:
print("Processando para Preto")
preto = process_image(azul.copy(), pixels, medias_cores[0, 1], medias_cores[1, 1], medias_cores[2, 1], tolerancia, window_size)
cv2.imshow("Preto", preto)
cv2.waitKey(0)

# Processando para Amarelo:
print("Processando para Amarelo")
amarelo = process_image(preto.copy(), pixels, medias_cores[0, 2], medias_cores[1, 2], medias_cores[2, 2], tolerancia, window_size)
cv2.imshow("Amarelo", amarelo)
cv2.waitKey(0)

# Processando para Vermelho:
print("Processando para Vermelho")
vermelho = process_image(amarelo.copy(), pixels, medias_cores[0, 3], medias_cores[1, 3], medias_cores[2, 3], tolerancia, window_size)
cv2.imshow("Vermelho", vermelho)
cv2.waitKey(0)

# Processando para Cinza:
print("Processando para Cinza")
cinza = process_image(vermelho.copy(), pixels, medias_cores[0, 4], medias_cores[1, 4], medias_cores[2, 4], tolerancia, window_size)
cv2.imshow("Cinza", cinza)
cv2.waitKey(0)

# Processando para Verde:
print("Processando para Verde")
verde = process_image(cinza.copy(), pixels, medias_cores[0, 5], medias_cores[1, 5], medias_cores[2, 5], tolerancia, window_size)
cv2.imshow("Verde", verde)
cv2.waitKey(0)






cv2.waitKey(0)
cv2.destroyAllWindows()
