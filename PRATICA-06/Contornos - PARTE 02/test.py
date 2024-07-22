def processar_pasta_teste(modelo, pasta_teste, class_names):
    """
    Processa todas as imagens na pasta teste e classifica cada uma.
    - Conta acertos e erros
    - Calcula a porcentagem de acertos
    """
    acertos = 0
    erros = 0
    total = 0

    # Percorre todos os arquivos na pasta teste
    for root, dirs, files in os.walk(pasta_teste):
        for file_name in files:
            if file_name.endswith(".png"):
                img_path = os.path.join(root, file_name)
                pasta_nome = os.path.basename(root)

                try:
                    # Classifica a imagem
                    prediction = classificar_imagem(modelo, img_path)
                    # Verifica a classificação prevista
                    class_name, confidence_score, resultado = verificar_classificacao(prediction, class_names, pasta_nome)

                    print(f"Pasta: {pasta_nome}")
                    print(f"Imagem: {file_name}")
                    print(f"Classificação Prevista: {class_name}")
                    print(f"Pontuação de Confiança: {confidence_score}")
                    print(f"Resultado: {resultado}\n")

                    if resultado == "Correto":
                        acertos += 1
                    else:
                        erros += 1

                    total += 1

                except ValueError as e:
                    print(e)

    if total > 0:
        porcentagem_acertos = (acertos / total) * 100
    else:
        porcentagem_acertos = 0

    print(f"\nTotal de Imagens: {total}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(f"Porcentagem de Acertos: {porcentagem_acertos:.2f}%")
