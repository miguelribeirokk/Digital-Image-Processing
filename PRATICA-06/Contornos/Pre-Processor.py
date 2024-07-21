import cv2
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split


def preProcess(img):
    """
    Pré-processa a imagem para detecção de contornos.
    - Aplica desfoque gaussiano
    - Detecta bordas usando Canny
    - Aplica dilatação para reforçar as bordas
    - Aplica erosão para remover ruídos
    """
    imgPre = cv2.GaussianBlur(img, (5, 5), 4)  # Desfoque
    imgPre = cv2.Canny(imgPre, 50, 150)  # Detecção de bordas
    kernel = np.ones((5, 5), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=2)  # Dilatação
    imgPre = cv2.erode(imgPre, kernel, iterations=1)  # Erosão
    return imgPre


def recortar_e_salvar(img_path, output_path):
    """
    Recorta o maior contorno da imagem original e salva em um novo arquivo.
    - Lê a imagem
    - Pré-processa a imagem
    - Encontra contornos
    - Recorta a região do maior contorno e salva
    """
    img = cv2.imread(img_path)  # Lê a imagem
    imgPre = preProcess(img)  # Pré-processa a imagem
    contours, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontra contornos

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Encontra o maior contorno
        x, y, w, h = cv2.boundingRect(largest_contour)  # Encontra a caixa delimitadora do maior contorno
        recorte = img[y:y + h, x:x + w]  # Recorta a imagem
        recorte = cv2.resize(recorte, (224, 224))  # Redimensiona o recorte
        recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY) #greyscale
        cv2.imwrite(output_path, recorte)  # Salva o recorte


def processar_pasta(pasta_origem, pasta_modelo, pasta_teste, test_size=0.2):
    """
    Processa todas as imagens na estrutura de subpastas e salva os recortes em uma estrutura correspondente.
    - Cria as pastas de modelo e teste se não existirem
    - Divide as imagens em conjuntos de modelo e teste
    - Pré-processa e recorta cada imagem, salvando nos destinos correspondentes
    """
    if not os.path.exists(pasta_modelo):
        os.makedirs(pasta_modelo)  # Cria a pasta modelo se não existir
    if not os.path.exists(pasta_teste):
        os.makedirs(pasta_teste)  # Cria a pasta teste se não existir

    for root, dirs, files in os.walk(pasta_origem):
        for dir_name in dirs:
            subdir_origem = os.path.join(root, dir_name)
            arquivos = [f for f in os.listdir(subdir_origem) if f.endswith(".png")]

            train_files, test_files = train_test_split(arquivos, test_size=test_size, random_state=42)

            subdir_modelo = os.path.join(pasta_modelo, dir_name)
            subdir_teste = os.path.join(pasta_teste, dir_name)

            if not os.path.exists(subdir_modelo):
                os.makedirs(subdir_modelo)  # Cria a subpasta modelo se não existir
            if not os.path.exists(subdir_teste):
                os.makedirs(subdir_teste)  # Cria a subpasta teste se não existir

            # Processar imagens de treino
            for filename in train_files:
                input_path = os.path.join(subdir_origem, filename)
                output_path = os.path.join(subdir_modelo, filename)
                recortar_e_salvar(input_path, output_path)  # Recorta e salva no modelo

            # Copiar imagens de teste sem processamento
            for filename in test_files:
                input_path = os.path.join(subdir_origem, filename)
                output_path = os.path.join(subdir_teste, filename)
                shutil.copy(input_path, output_path)  # Copia a imagem para a pasta teste


def main():
    """
    Função principal que organiza o fluxo do programa.
    - Define as pastas de origem e destino
    - Processa as imagens para recorte
    """
    # REMOVA O COMENTÁRIO DAS LINHAS ABAIXO PARA EXECUTAR O PROGRAMA
    # AO EXECUTAR O PROGRAMA AS PASTAS /TESTE E /MODELO SERÃO SOBRESCRITAS
    #logos_path = 'Logos'
    #modelo_path = 'modelo'
    #teste_path = 'teste'

    # Processar imagens
    #processar_pasta(logos_path, modelo_path, teste_path)
    print("Remova o comentário das linhas 93 a 98 para executar o programa.")


if __name__ == "__main__":
    main()
