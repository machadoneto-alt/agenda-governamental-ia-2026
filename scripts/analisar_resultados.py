"""
Compara a codificação humana (padrão de referência) com as classificações do
ParlaCAP e da Babel Machine, calculando todas as métricas e figuras
reportadas no relatório (seções 3.1 a 3.3): concordância geral, Alpha de
Krippendorff, F1 por categoria (macro e ponderado), detalhamento do código de
maior desacordo e as Figuras 1 e 2.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import krippendorff
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONFIGURAÇÕES ---
ARQUIVO = "./dados/comparativo_manual_parlacap_babelmachine.xlsx"
PLANILHA = "codificação"
LINHA_CABECALHO = 1          # a planilha tem um cabeçalho em duas linhas
COLUNA_HUMANO = "Código"
COLUNA_PARLACAP = "topico_cap_codigo"
COLUNA_BABEL = "codigo_cap"
CODIGO_RESIDUAL_BABEL = 999  # rótulo nativo da Babel Machine p/ "sem política"
CODIGO_RESIDUAL_FINAL = 99   # código residual unificado usado na comparação
CODIGO_FOCO = 20             # categoria detalhada nas Tabelas 3 e 4 do relatório

FERRAMENTAS = {"ParlaCAP": COLUNA_PARLACAP, "Babel Machine": COLUNA_BABEL}


def carregar_dados():
    df = pd.read_excel(ARQUIVO, sheet_name=PLANILHA, header=LINHA_CABECALHO)
    df[COLUNA_BABEL] = df[COLUNA_BABEL].replace(CODIGO_RESIDUAL_BABEL, CODIGO_RESIDUAL_FINAL)
    valido = df.dropna(subset=[COLUNA_HUMANO]).copy()
    print(f"Unidades totais: {len(df)} | Unidades válidas: {len(valido)}")
    return valido


def metricas_por_ferramenta(valido, nome, coluna):
    y_true = valido[COLUNA_HUMANO].astype(int)
    y_pred = valido[coluna].astype(int)
    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))

    acc = accuracy_score(y_true, y_pred)
    dados_alpha = np.array([y_true.values, y_pred.values])
    alpha = krippendorff.alpha(reliability_data=dados_alpha, level_of_measurement="nominal")

    relatorio = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    relatorio_df = pd.DataFrame(relatorio).transpose()

    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    matriz_df = pd.DataFrame(matriz, index=labels, columns=labels)

    print(f"\n==== {nome} ==== acc={acc:.4%} | alpha={alpha:.4f}")
    print(relatorio_df[["precision", "recall", "f1-score", "support"]].round(2))

    return {"labels": labels, "acc": acc, "alpha": alpha,
            "relatorio_df": relatorio_df, "matriz_df": matriz_df}


def detalhar_codigo_foco(resultados, codigo=CODIGO_FOCO):
    """Reproduz as Tabelas 3 e 4 do relatório para um código específico."""
    for nome, r in resultados.items():
        matriz_df = r["matriz_df"]
        if codigo not in matriz_df.index:
            continue
        linha = matriz_df.loc[codigo]
        total = linha.sum()
        convergiu = linha.get(codigo, 0)
        residual = linha.get(CODIGO_RESIDUAL_FINAL, 0)
        outro = total - convergiu - residual
        print(f"\n{nome} — {total} unidades no código {codigo}: "
              f"convergiu={convergiu}, residual={residual}, outro={outro}")
        destinos = linha.drop(labels=[codigo, CODIGO_RESIDUAL_FINAL], errors="ignore")
        for cod_destino, qtd in destinos[destinos > 0].sort_values(ascending=False).items():
            print(f"    código {int(cod_destino)}: {qtd}")


def figura_f1_por_categoria(resultados, caminho="graficos/figura1_f1_categoria.png"):
    labels = sorted(set(resultados["ParlaCAP"]["labels"]) | set(resultados["Babel Machine"]["labels"]))
    labels = [c for c in labels if c != CODIGO_RESIDUAL_FINAL]
    f1 = {nome: [r["relatorio_df"].loc[str(c), "f1-score"] if str(c) in r["relatorio_df"].index else 0.0
                 for c in labels] for nome, r in resultados.items()}
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=200)
    ax.bar(x - w/2, f1["ParlaCAP"], w, label="ParlaCAP", color="#4C72B0")
    ax.bar(x + w/2, f1["Babel Machine"], w, label="Babel Machine", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in labels])
    ax.set_xlabel("Código do macrotópico (CAP)"); ax.set_ylabel("F1-Score")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(caminho, dpi=300, facecolor="white")
    print(f"\nFigura 1 salva em: {caminho}")


def figura_frequencia_absoluta(valido, caminho="graficos/figura2_frequencia_categoria.png"):
    h = valido[COLUNA_HUMANO].astype(int).value_counts()
    p = valido[COLUNA_PARLACAP].astype(int).value_counts()
    b = valido[COLUNA_BABEL].astype(int).value_counts()
    labels = sorted(set(h.index) | set(p.index) | set(b.index))
    tab = pd.DataFrame({"Humano": h, "ParlaCAP": p, "Babel Machine": b}).reindex(labels).fillna(0).astype(int)
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
    ax.bar(x - w, tab["Humano"], w, label="Humano (padrão de referência)", color="#555555")
    ax.bar(x, tab["ParlaCAP"], w, label="ParlaCAP", color="#4C72B0")
    ax.bar(x + w, tab["Babel Machine"], w, label="Babel Machine", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in labels])
    ax.set_xlabel("Código do macrotópico (CAP); 99 = categoria residual")
    ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(caminho, dpi=300, facecolor="white")
    print(f"Figura 2 salva em: {caminho}")


def main():
    import os
    os.makedirs("graficos", exist_ok=True)
    valido = carregar_dados()
    resultados = {nome: metricas_por_ferramenta(valido, nome, col) for nome, col in FERRAMENTAS.items()}
    detalhar_codigo_foco(resultados)
    figura_f1_por_categoria(resultados)
    figura_frequencia_absoluta(valido)


if __name__ == "__main__":
    main()
