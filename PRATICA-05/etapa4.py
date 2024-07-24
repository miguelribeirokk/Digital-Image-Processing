# 4) realize a segmentação de pele nas fotos do sada e do bayer , usando o mesmo inrange para as duas imagens.
import cv2
import numpy as np

bayer = cv2.imread("images/bayer.png")
sada = cv2.imread("images/sada.jpg")


lower = np.array([1, 15, 0])
upper = np.array([55, 255, 255])

# Convertendo a imagem para o espaço de cores HSV
hsa_bayer = cv2.cvtColor(bayer, cv2.COLOR_BGR2HSV)
hsa_sada = cv2.cvtColor(sada, cv2.COLOR_BGR2HSV)

hsa_bayer_mask = cv2.inRange(hsa_bayer, lower, upper)
hsa_sada_mask = cv2.inRange(hsa_sada, lower, upper)

# Aplicando um filtro morfológico para melhorar a segmentação
kernel = np.ones((5, 5), np.uint8)
hsa_bayer_mask = cv2.morphologyEx(hsa_bayer_mask, cv2.MORPH_OPEN, kernel)
hsa_sada_mask = cv2.morphologyEx(hsa_sada_mask, cv2.MORPH_OPEN, kernel)

bayer_segmentada = cv2.bitwise_and(bayer, bayer, mask=hsa_bayer_mask)
sada_segmentada = cv2.bitwise_and(sada, sada, mask=hsa_sada_mask)

cv2.imshow('bayer_segmentada', bayer_segmentada)
cv2.imshow('sada_segmentada', sada_segmentada)
cv2.waitKey(0)
cv2.destroyAllWindows()