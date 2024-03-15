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
img = cv2.imread("imagens/halteres.jpg")

# Cria uma janela para exibir a imagem
cv2.namedWindow('frame1')

# Cria uma matriz para armazenar os píxeis selecionados pelo usuário
pixels = np.zeros((1, 3), dtype=np.uint8)

# Define a função de callback para eventos de mouse
cv2.setMouseCallback("frame1", on_mouse)

# Loop principal para exibir a imagem e aguardar por interação do usuário
while True:
    cv2.imshow("frame1", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):  # Se a tecla 'c' for pressionada, sai do loop
        break

# Remove a primeira linha da matriz de píxeis, inicializada com zeros
pixels = np.delete(pixels, 0, axis=0)

# Calcula a média dos valores BGR dos pixels selecionados
media = np.mean(pixels, axis=0)
mediaB = media[0]
mediaG = media[1]
mediaR = media[2]
print("Média B=%d, Média G=%d, Média R=%d" % (mediaB, mediaG, mediaR))

# Define uma margem de tolerância para os valores de cor
tolerancia = 60

# Define o tamanho da janela ao redor de cada pixel para verificar os vizinhos
window_size = 8

print(mediaB, mediaG, mediaR, tolerancia, window_size)
# Processa a imagem para a primeira cor
img_processed = process_image(img.copy(), pixels, mediaB, mediaG, mediaR, tolerancia, window_size)

# Exibe a imagem processada
cv2.imshow("frame12", img_processed)
cv2.waitKey(0)
cv2.destroyAllWindows()
