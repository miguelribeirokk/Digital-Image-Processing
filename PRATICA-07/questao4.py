import numpy as np
import cv2
import matplotlib.pyplot as plt

import cv2
import numpy as np


def aplicando_perspectiva(orig, points):
    # Definir os pontos de destino para a transformação de perspectiva
    width, height = 205, 445  # Ajuste o tamanho da nova perspectiva conforme necessário
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    # Calcular a matriz de transformação de perspectiva
    src_points = np.array(points, dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Aplicar a transformação de perspectiva
    warped = cv2.warpPerspective(orig, M, (width, height))
    return warped

# Função de callback para registrar os pontos clicados
def click_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Clique nos cantos:(superior esquerdo), (superior direito), (inferior direito), (inferior esquerdo)", image)

image = cv2.imread('quadratenis2.png' )

orig = image.copy()
cv2.imshow("Clique nos cantos:(superior esquerdo), (superior direito), (inferior direito), (inferior esquerdo)", image)

points = []
# Configurar a janela para capturar os cliques do mouse
cv2.namedWindow("Clique nos cantos:(superior esquerdo), (superior direito), (inferior direito), (inferior esquerdo)")
cv2.setMouseCallback("Clique nos cantos:(superior esquerdo), (superior direito), (inferior direito), (inferior esquerdo)", click_points)

# Esperar até que 4 pontos sejam clicados
while True:
    cv2.imshow("Clique nos cantos:(superior esquerdo), (superior direito), (inferior direito), (inferior esquerdo)", image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressione 'Esc' para sair sem selecionar os pontos
        break
    if len(points) == 4:
        break

if len(points) != 4:
    exit(0)

for i in range(len(points)):
    x, y = points[i]
    if i == 0:
        points[i] = (x - 5, y - 5)
    elif i == 1:
        points[i] = (x + 5, y - 5)
    elif i == 2:
        points[i] = (x + 5, y + 5)
    elif i == 3:
        points[i] = (x - 5, y + 5)

image_perspectiva = aplicando_perspectiva(orig, points)
# Mostrar a imagem transformada
cv2.destroyAllWindows()


cv2.imwrite('image_perspectiva.png', image_perspectiva)

print("Clique nos 4 pontos: superior esquerdo, meio, inferior esquerdo, distancia de 1,37")
#img = cv2.imread('image_perspectiva.png')

img = image_perspectiva.copy()

# Função de callback para registrar os pontos clicados
def click_points_referencia(event, x, y, flags, param):
    global points_referencia
    if event == cv2.EVENT_LBUTTONDOWN:
        points_referencia.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Clique nos 4 pontos: superior esquerdo, meio, inferior esquerdo, distancia de 1,37", img)


points_referencia = []
# Configurar a janela para capturar os cliques do mouse
cv2.namedWindow("Clique nos 4 pontos: superior esquerdo, meio, inferior esquerdo, distancia de 1,37")
cv2.setMouseCallback("Clique nos 4 pontos: superior esquerdo, meio, inferior esquerdo, distancia de 1,37", click_points_referencia)

# Esperar até que 2 pontos sejam clicados
while True:
    #print(len(points_referencia))
    cv2.imshow("Clique nos 4 pontos: superior esquerdo, meio, inferior esquerdo, distancia de 1,37", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressione 'Esc' para sair sem selecionar os pontos
        break
    if len(points_referencia) == 4:
        break


referencia = (points_referencia[3][0] - points_referencia[2][0])

comprimento_metade_superior = (points_referencia[1][1] - points_referencia[0][1]) / referencia * 1.37
comprimento_metade_inferior = (points_referencia[2][1] - points_referencia[1][1]) / referencia * 1.37
comprimento = (points_referencia[2][1] - points_referencia[0][1]) / referencia * 1.37
print("Cada metade da quadra tem 11,88 metros")
print("Apos usar a perspectiva cada metade da quadra ficou com:")
print(f"Metade superior: {comprimento_metade_superior} metros")
print(f"Metade inferior:{comprimento_metade_inferior} metros")
print(f"Comprimento total: {comprimento}")
cv2.destroyAllWindows()

cv2.imshow("Image transformada", image_perspectiva)
cv2.waitKey(0)
cv2.destroyAllWindows()