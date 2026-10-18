# Inteligência artificial na pesquisa sobre agenda governamental

Código de reprodutibilidade do Relatório Final de Iniciação Científica
"Inteligência artificial na pesquisa sobre agenda governamental: comparando
métodos de análise dos pronunciamentos do governo Bolsonaro (2019-2022)"
(Edital PROPe Unesp nº 08/2025 — PIBIC/Unesp).

**Autor:** Luiz Rodrigues Machado Neto (machado.neto@unesp.br)
**Orientadora:** Profa. Dra. Ana Cláudia Niedhardt Capella (ana.capella@unesp.br)

## O que este repositório contém

Este repositório reúne os scripts usados para aplicar duas ferramentas de
classificação automatizada de tópicos de política pública — a **CAP Babel
Machine** e o **ParlaCAP** — a um corpus de pronunciamentos presidenciais já
codificado manualmente, e para comparar essas classificações com a
codificação humana (seções 2 e 3 do relatório).

```
scripts/
  executar_babel_machine.py   # aplica o modelo da Babel Machine ao corpus
  executar_parlacap.py        # aplica o modelo do ParlaCAP ao corpus
  analisar_resultados.py      # calcula as métricas e gera as Figuras 1 e 2
dados/
  (coloque aqui o arquivo comparativo; não incluído neste repositório — ver nota abaixo)
graficos/
  (saída dos gráficos gerados por analisar_resultados.py)
requirements.txt
```

## Como rodar

```bash
pip install -r requirements.txt

python scripts/executar_babel_machine.py   # requer HF_TOKEN (ver abaixo)
python scripts/executar_parlacap.py

python scripts/analisar_resultados.py
```

Antes de rodar `executar_babel_machine.py`, solicite acesso ao modelo
`poltextlab/xlm-roberta-large-portuguese-execspeech-cap-v3` no Hugging Face e
defina seu token como variável de ambiente:

```bash
export HF_TOKEN="seu_token_aqui"
```

## Dados

O arquivo comparativo (`comparativo_manual_parlacap_babelmachine.xlsx`), com
a codificação humana e as classificações das duas ferramentas para as 485
unidades de análise, está disponível mediante solicitação ao autor ou à
orientadora, em razão do volume do arquivo e de direitos sobre a codificação
manual da pesquisa anterior do grupo.

## Citação

MACHADO NETO, Luiz Rodrigues. Inteligência artificial na pesquisa sobre
agenda governamental: comparando métodos de análise dos pronunciamentos do
governo Bolsonaro (2019-2022). Orientadora: Ana Cláudia Niedhardt Capella.
Relatório Final de Iniciação Científica (PIBIC/Unesp, Edital PROPe nº
08/2025) — Faculdade de Ciências e Letras, Universidade Estadual Paulista
"Júlio de Mesquita Filho", Araraquara, 2026.
