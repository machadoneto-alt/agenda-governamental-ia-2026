"""
Classifica um conjunto de unidades de análise com o classificador do ParlaCAP,
treinado em discursos de parlamentos europeus (corpus ParlaMint).

Modelo: classla/ParlaCAP-Topic-Classifier (acesso livre, sem token).

Nota metodológica: a ficha técnica do ParlaCAP descreve um rótulo "Mix" para
previsões com confiança inferior a 0,60. Ao acessar o modelo diretamente via
transformers.pipeline, como feito aqui, esse rótulo nunca é atribuído — o
pipeline sempre devolve a categoria de maior probabilidade, mesmo quando essa
probabilidade é baixa. O comportamento de sinalização descrito na ficha
técnica parece ser aplicado pela interface de demonstração oficial do
ParlaCAP, não pelo modelo em si.
"""

import pandas as pd
from transformers import pipeline
import os

# --- CONFIGURAÇÕES ---
MODELO = "classla/ParlaCAP-Topic-Classifier"
ARQUIVO_ENTRADA = "./dados/unidades_analise.csv"   # mesmas 485 unidades usadas na Babel Machine
ARQUIVO_SAIDA = "./resultados/resultado_parlacap.csv"


def processar_discurso():
    print("EXECUTANDO MODELO PARLACAP")

    # 1. Carregamento do pipeline
    classifier = pipeline(
        "text-classification", model=MODELO, use_safetensors=False
    )

    # 2. Leitura dos dados
    df = pd.read_csv(ARQUIVO_ENTRADA)
    textos = df["text"].astype(str).tolist()

    # 3. Inferência (rótulo mais provável + confiança)
    resultados = classifier(textos, truncation=True, max_length=512)

    # 4. Organização dos resultados
    df["topico_cap"] = [res["label"] for res in resultados]
    df["confianca"] = [f"{res['score']:.2%}" for res in resultados]

    # 5. Salvamento
    os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")
    print(f"Resultado salvo em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    processar_discurso()
