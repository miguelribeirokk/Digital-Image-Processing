from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import os


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

#reconhece imagem de rosto do imput img
def reconhecer_rosto(img):
    


def recortar_imagem(img):
    """
    Recorta o maior contorno da imagem original.
    - Encontra contornos e recorta a região do maior contorno
    """
    imgPre = preProcess(img)
    contours, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        recorte = img[y:y + h, x:x + w]
        recorte = cv2.resize(recorte, (224, 224))
        recorte = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        return recorte
    return img  # Retorna a imagem original se nenhum contorno for encontrado


def preparar_imagem(img_path):
    """
    Pré-processa a imagem para o modelo.
    - Lê, pré-processa e recorta a imagem
    - Normaliza os valores dos pixels
    """
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem no caminho: {img_path}")

    img = recortar_imagem(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para RGB
    img = Image.fromarray(img)  # Converte de volta para imagem PIL
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array


def classificar_imagem(modelo, img_path):
    """
    Classifica uma imagem usando o modelo.
    """
    img_array = preparar_imagem(img_path)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = img_array
    prediction = modelo.predict(data)
    return prediction


def carregar_labels(labels_path):
    """
    Carrega os rótulos do arquivo e cria um dicionário para acessar os nomes das classes.
    """
    class_names = {}
    with open(labels_path, "r") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                index, name = parts
                class_names[int(index)] = name.lower()
    return class_names


def verificar_classificacao(prediction, class_names, pasta_nome):
    """
    Verifica se a classificação do modelo corresponde ao nome da pasta.
    """
    index = np.argmax(prediction)
    class_name = class_names.get(index, "Unknown").strip()  # Remove espaços extras
    pasta_nome = pasta_nome.strip().lower()  # Remove espaços e converte para minúsculas
    confidence_score = prediction[0][index]

    if class_name == pasta_nome:
        resultado = "Correto"
    else:
        resultado = "Incorreto"

    return class_name, confidence_score, resultado


def processar_pasta_teste(modelo, pasta_teste, class_names):
    """
    Processa todas as imagens na pasta teste e classifica cada uma.
    - Conta acertos e erros
    """
    acertos = 0
    erros = 0
    total = 0

    for root, dirs, files in os.walk(pasta_teste):
        for file_name in files:
            if file_name.endswith(".png"):  # Processa apenas arquivos PNG
                img_path = os.path.join(root, file_name)
                pasta_nome = os.path.basename(root)

                try:
                    prediction = classificar_imagem(modelo, img_path)
                    class_name, confidence_score, resultado = verificar_classificacao(prediction, class_names,
                                                                                      pasta_nome)

                    # Imprimir nome da pasta, nome do arquivo, previsão e pontuação de confiança
                    print(f"Pasta: {pasta_nome}")
                    print(f"Imagem: {file_name}")
                    print(f"Classificação Prevista: {class_name}")
                    print(f"Pontuação de Confiança: {confidence_score}")
                    print(f"Resultado: {resultado}\n")

                    # Contar acertos e erros
                    if resultado == "Correto":
                        acertos += 1
                    else:
                        erros += 1

                    total += 1

                except ValueError as e:
                    print(e)

    # Calcular a porcentagem de acertos
    if total > 0:
        porcentagem_acertos = (acertos / total) * 100
    else:
        porcentagem_acertos = 0

    print(f"\nTotal de Imagens: {total}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(f"Porcentagem de Acertos: {porcentagem_acertos:.2f}%")


def main():
    """
    Função principal para classificar imagens na pasta teste usando o modelo.
    """
    model_path = 'keras_Model.h5'
    labels_path = 'labels.txt'
    pasta_teste = 'teste/'  # Caminho para a pasta de teste

    # Carregar o modelo e rótulos
    model = load_model(model_path, compile=False)
    class_names = carregar_labels(labels_path)

    # Processar a pasta de teste
    processar_pasta_teste(model, pasta_teste, class_names)


if __name__ == "__main__":
    main()
