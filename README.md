# Limiarização de documentos manuscritos

Comparação de métodos de limiarização (binarização) em imagens de documentos manuscritos, avaliados contra um *ground truth* com as métricas do DIBCO.

## Métodos

Todos implementados com `scikit-image`:

| Método  | Descrição                                               |
| ------- | ------------------------------------------------------- |
| Global  | Limiar fixo escolhido a priori (`GLOBAL_T = 0.5`)       |
| Otsu    | Limiar global automático (`threshold_otsu`)             |
| Local   | Média gaussiana da vizinhança (`threshold_local`)       |
| Sauvola | Limiar local, padrão na literatura de documentos        |

## Métricas

- **F-Measure (FM)**, **Precisão** e **Revocação** (%) — maior é melhor
- **PSNR** (dB) — maior é melhor
- **DRD** (*Distance Reciprocal Distortion*) — menor é melhor

## Estrutura

```
data/
  original/   imagens de entrada (ex.: 1.bmp)
  gt/         ground truth correspondente (ex.: 1_gt.bmp)
outputs/      figuras e tabelas geradas
main.py       script principal
```

As imagens são pareadas pelo nome, ignorando sufixos como `_gt`.

## Como executar

Requer Python 3.14+ e [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run main.py
```

Os parâmetros (limiar global, tamanho de bloco, janela e `k` do Sauvola etc.) ficam no topo de `main.py`.

## Saídas

Gerados em `outputs/`:

- `histogramas.png` — histogramas de cada imagem com os limiares de cada método
- `comparacao_<n>.png` — resultado de cada método e mapa de erros (vermelho = falso positivo, azul = falso negativo)
- `metricas_boxplot.png` — distribuição de FM, PSNR e DRD por método
- `metricas_por_imagem.csv` — métricas de cada imagem e método
- `metricas_resumo.csv` — média e desvio padrão por método

## Resultados

Média ± desvio padrão nas 10 imagens:

| Método  | FM (%)        | PSNR (dB)    | DRD          |
| ------- | ------------- | ------------ | ------------ |
| Global  | 76.55 ± 22.53 | 16.91 ± 5.06 | 7.15 ± 6.04  |
| Otsu    | 86.54 ± 7.26  | 17.78 ± 4.46 | 5.52 ± 4.32  |
| Local   | 75.20 ± 8.87  | 13.98 ± 1.97 | 15.22 ± 8.44 |
| Sauvola | 82.23 ± 14.70 | 17.06 ± 3.55 | 6.33 ± 4.60  |
