import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def detect_harris_corners(img, blockSize=2, ksize=3, k=0.04, threshold=0.01):
    # Detecção de cantos usando Harris
    dst = cv.cornerHarris(img, blockSize, ksize, k)

    # Normalizar e converter a imagem
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)

    # Aplicar um limiar para a detecção de cantos
    corners = np.zeros_like(dst)
    corners[dst > threshold * dst.max()] = 1

    return corners, dst_norm_scaled


def find_lines(img):
    # Detectar bordas usando Canny
    edges = cv.Canny(img, 50, 150, apertureSize=3)

    # Detectar linhas usando a Transformada de Hough
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    return lines


def find_intersections(lines):
    # Encontra todas as interseções das linhas detectadas
    points = set()

    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]

                # Calcula a interseção das linhas
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    continue

                inter_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                inter_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

                if (min(x1, x2) <= inter_x <= max(x1, x2) and
                        min(y1, y2) <= inter_y <= max(y1, y2) and
                        min(x3, x4) <= inter_x <= max(x3, x4) and
                        min(y3, y4) <= inter_y <= max(y3, y4)):
                    points.add((int(inter_x), int(inter_y)))

    return points


def draw_lines(img, lines):
    img_with_lines = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    # Desenhar linhas detectadas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_with_lines


def draw_intersections(img, intersections):
    img_with_intersections = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    # Desenhar interseções
    for (x, y) in intersections:
        cv.circle(img_with_intersections, (x, y), 5, (0, 0, 255), -1)

    return img_with_intersections


# Carregar a imagem da quadra
img0 = cv.imread('quadratenis2.png', cv.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada
if img0 is None:
    raise FileNotFoundError("A imagem 'quadra.jpg' não foi encontrada.")

# Aplicar desfoque gaussiano para reduzir o ruído
img = cv.GaussianBlur(img0, (5, 5), 0)

# Detectar cantos usando Harris
cantos, img_cantos = detect_harris_corners(img)

# Encontrar linhas na imagem
lines = find_lines(img)

# Encontrar interseções das linhas
intersections = find_intersections(lines)

# Desenhar linhas na imagem
img_with_lines = draw_lines(img0, lines)

# Marcar os cantos detectados com Harris na imagem
annotatedImg_cantos = cv.cvtColor(img0, cv.COLOR_GRAY2RGB)
for i in range(img0.shape[0]):
    for j in range(img0.shape[1]):
        if cantos[i, j] == 1:
            cv.circle(annotatedImg_cantos, (j, i), 5, (0, 0, 255), -1)

# Desenhar interseções na imagem
img_with_intersections = draw_intersections(img0, intersections)

# Exibir as imagens
plt.figure(figsize=(15, 15))

plt.subplot(1, 3, 1)
plt.imshow(annotatedImg_cantos)
plt.title('Cantos Detectados com Harris')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_with_lines)
plt.title('Linhas Detectadas')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_with_intersections)
plt.title('Interseções das Linhas')
plt.axis('off')

plt.show()

# Salvar as imagens
cv.imwrite('quadra_cantos_detectados.jpg', cv.cvtColor(annotatedImg_cantos, cv.COLOR_RGB2BGR))
cv.imwrite('quadra_linhas_detectadas.jpg', cv.cvtColor(img_with_lines, cv.COLOR_RGB2BGR))
cv.imwrite('quadra_intersecoes.jpg', cv.cvtColor(img_with_intersections, cv.COLOR_RGB2BGR))
