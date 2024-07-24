#2) usando a imagem barras, apresente, para cada cor, um gráfico mostrando a evolução dos componentes RGB e HSV para cada cor. Trace o mesmo para as 3 cores, e também no espaço HSV.
import cv2
import numpy as np
import matplotlib.pyplot as plt

y_linha = None
linha_bgr = None
fechar_imagem = False


def mouse_callback(event, x, y, flags, param):
    global linha_bgr
    global y_linha
    global fechar_imagem

    if event == cv2.EVENT_RBUTTONDOWN:
        # Obtém a linha correspondente à posição do mouse
        linha_bgr = img[y, :]
        y_linha = y

        cv2.setMouseCallback('Imagem', lambda *args: None)  # Remove o callback do mouse
        fechar_imagem = True

# Carrega a imagem do disco
img = cv2.imread('images/barras.png')
img = cv2.imread("images/cuboIndoor.png")


# Cria uma janela para exibir a imagem
cv2.namedWindow('Imagem')
# Registra a função callback para o evento de clique do mouse
cv2.setMouseCallback('Imagem', mouse_callback)
# Exibe a imagem na janela
cv2.imshow('Imagem', img)

# Aguarda que o usuário pressione a tecla 'q' ou selecione 1 linha
while True:
    if (cv2.waitKey(1) & 0xFF == ord('q')) | (fechar_imagem == True):
        break


cv2.destroyAllWindows()
linha_bgr = linha_bgr[10:-14]

# Separar os valores BGR
red_values = linha_bgr[:, 2]
green_values = linha_bgr[:, 1]
blue_values = linha_bgr[:, 0]

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Converter para HSV
linha_hsv = img_hsv[y_linha, :][10:-14]

# Separar os valores HSV
h = linha_hsv[:, 0]
s = linha_hsv[:, 1]
v = linha_hsv[:, 2]

# Plotar os valores RGB
plt.figure(figsize=(10, 5))

plt.plot(red_values, label='Red', color='red')
plt.plot(green_values, label='Green', color='green')
plt.plot(blue_values, label='Blue', color='blue')
plt.xlabel('Largura')
plt.ylabel('RGB Value')
plt.title(f'Valores RGB ao longo da largura na altura {y_linha}')
plt.legend() 

# Plotar os valores HSV
plt.figure(figsize=(10, 5))

plt.plot(h, label='H')
plt.plot(s, label='S')
plt.plot(v, label='V')
plt.xlabel('Largura')
plt.ylabel('HSV Value')
plt.title(f'Valores HSV ao longo da largura na altura {y_linha}')
plt.legend() 
plt.show()

