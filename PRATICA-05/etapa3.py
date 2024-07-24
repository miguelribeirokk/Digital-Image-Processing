# 3) usando as imagens do cubo indoor e cubo outdoor, trace os gráficos ou imagens que achar necessário para apresentar os valores de H,S ,V, R, G e B de cada quadrado do cubo indoor comparando-o com o cubo Outdoor

# 

import cv2
import numpy as np
import matplotlib.pyplot as plt

cubo_indoor_bgr = cv2.imread("images/cuboIndoor.png")
cubo_outdoor_bgr = cv2.imread("images/cuboOutdoor.png")
altura, largura, _ = cubo_indoor_bgr.shape

cubo_indoor_hsv = cv2.cvtColor(cubo_indoor_bgr, cv2.COLOR_BGR2HSV) 
cubo_outdoor_hsv = cv2.cvtColor(cubo_outdoor_bgr, cv2.COLOR_BGR2HSV)  

def plotar_grafico_valores(bgr, hsv, quadrado):
    B = bgr[:, 2]
    G = bgr[:, 1]
    R = bgr[:, 0]
    H = hsv[:, 0]
    S = hsv[:, 1]
    V = hsv[:, 2]
    
    plt.figure(figsize=(10, 5))
    plt.plot(R, label='Red', color='red')
    plt.plot(G, label='Green', color='green')
    plt.plot(B, label='Blue', color='blue')
    plt.plot(H, label='H', color='yellow')
    plt.plot(S, label='S', color='black')
    plt.plot(V, label='V', color='brown')
    plt.xlabel('Largura')
    plt.ylabel('Pix Value')
    plt.title(f'{quadrado}')
    plt.legend()
    #plt.savefig(f"{quadrado}.png")
    #plt.show()

quadrados_indoor = ["Quadrado1 Indoor", "Quadrado2 Indoor", "Quadrado3 Indoor", 
                    "Quadrado4 Indoor", "Quadrado5 Indoor", "Quadrado6 Indoor",  
                    "Quadrado7 Indoor", "Quadrado8 Indoor", "Quadrado9 Indoor"]

quadrados_outdoor = ["Quadrado1 Outdoor", "Quadrado2 Outdoor", "Quadrado3 Outdoor", 
                    "Quadrado4 Outdoor", "Quadrado5 Outdoor", "Quadrado6 Outdoor",  
                    "Quadrado7 Outdoor", "Quadrado8 Outdoor", "Quadrado9 Outdoor"]

y = 100 # linha
quadrado = 0
for i in range(0, 3):
    x_incio = 22
    x_fim = 122
    for j in range(0, 3):
        bloco_indoor_bgr = cubo_indoor_bgr[y, :][x_incio:x_fim]
        bloco_outdoor_bgr = cubo_outdoor_bgr[y, :][x_incio:x_fim]
        bloco_indoor_hsv = cubo_indoor_hsv[y, :][x_incio:x_fim]
        bloco_outdoor_hsv = cubo_outdoor_hsv[y, :][x_incio:x_fim]
        plotar_grafico_valores(bloco_indoor_bgr, bloco_indoor_hsv, quadrados_indoor[quadrado])
        plotar_grafico_valores(bloco_outdoor_bgr, bloco_outdoor_hsv, quadrados_outdoor[quadrado])
        plt.show()
        x_incio += 103
        x_fim += 103
        quadrado += 1
    
    y += 100



