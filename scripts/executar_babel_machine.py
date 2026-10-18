"""
Classifica um conjunto de unidades de análise com o modelo da Babel Machine
especializado em discursos do Executivo em português (domínio "execspeech").

Modelo: poltextlab/xlm-roberta-large-portuguese-execspeech-cap-v3
Acesso restrito (gated) no Hugging Face: solicite autorização ao desenvolvedor
e gere um token pessoal em https://huggingface.co/settings/tokens.
"""

import pandas as pd
from transformers import pipeline
import os

# --- CONFIGURAÇÕES ---
# Defina a variável de ambiente HF_TOKEN antes de rodar o script.
# NUNCA grave o token em texto puro em um script compartilhado.
MEU_TOKEN = os.environ.get("HF_TOKEN", "SEU_TOKEN_AQUI")

MODELO = "poltextlab/xlm-roberta-large-portuguese-execspeech-cap-v3"
ARQUIVO_ENTRADA = "./dados/unidades_analise.csv"   # 485 unidades, colunas: id, text
ARQUIVO_SAIDA = "./resultados/resultado_babel_machine.csv"


def executar_teste():
    print("EXECUTANDO MODELO BABEL MACHINE (execspeech-cap-v3)")

    # 1. Carregamento com autenticação (modelo de acesso restrito/gated)
    classifier = pipeline(
        "text-classification",
        model=MODELO,
        token=MEU_TOKEN,
        use_safetensors=False,  # necessário para esta versão do modelo
    )

    # 2. Leitura do arquivo com as unidades de análise
    df = pd.read_csv(ARQUIVO_ENTRADA)
    textos = df["text"].astype(str).tolist()

    # 3. Inferência (rótulo mais provável + confiança), truncando em 512 tokens
    resultados = classifier(textos, truncation=True, max_length=512)

    # 4. Mapeamento dos resultados para o DataFrame
    df["label_modelo"] = [res["label"] for res in resultados]
    df["score_confianca"] = [f"{res['score']:.2%}" for res in resultados]

    # 5. Exportação
    os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")
    print(f"Resultados salvos em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    executar_teste()
