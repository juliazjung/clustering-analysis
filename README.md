# clustering-analysis

Análise comparativa de algoritmos de clusterização aplicados à identificação de coalizões políticas nas votações nominais da Câmara dos Deputados do Brasil.
 
Desenvolvido como parte do Trabalho de Conclusão de Curso em Ciência de Computação na Universidade Regional Integrada do Alto Uruguai e das Missões (URI) — Santo Ângelo, RS.
 
---
 
## Sobre
 
Este repositório contém o código-fonte dos experimentos descritos no artigo *"Análise de Algoritmos de Clusterização na Identificação de Coalizões Políticas na Câmara de Deputados"* (Jung & Silva, 2025). O trabalho avalia comparativamente três algoritmos — K-means, Agglomerative Clustering e HDBSCAN — com base em métricas de qualidade de agrupamento, eficiência computacional e correspondência a episódios históricos de coalizões.
 
---
 
## Estrutura
 
```
├── clustering-analysis.py       # Script principal: clusterização + visualização PCA
└── README.md
```
 
---
 
## Requisitos
 
- Python 3.13+
- scikit-learn 1.7.2
- branalysis 1.0.12
- pandas
- matplotlib
- numpy
Instale as dependências:
 
```bash
pip install scikit-learn branalysis pandas matplotlib numpy
```
 
---
 
## Como executar
 
```bash
clustering-analysis.py
```
 
Os parâmetros de período e número de clusters podem ser ajustados diretamente no script nas variáveis `ANO_MIN`, `ANO_MAX` e `n_clusters`.
 
---
 
## Metodologia
 
Os dados são coletados via biblioteca [Branalysis](https://github.com/mcreuch/branalysis) (Kreuch, 2024), que acessa as APIs públicas da Câmara dos Deputados e armazena os registros localmente em SQLite.
 
Os votos são codificados numericamente (Sim = 1, Não = −1, Abstenção/Obstrução/Art. 17 = 0) e organizados em uma matriz parlamentar × votação. Parlamentares com menos de 20% de participação nas votações do período são excluídos. A visualização utiliza PCA com dois componentes aplicado a posteriori, sem interferência no processo de clusterização.

---
 
## Referência
 
Se utilizar este código, por favor cite:
 
```
Jung, J. e Silva, D. R. (2025) "Análise de Algoritmos de Clusterização na Identificação
de Coalizões Políticas na Câmara de Deputados", URI Santo Ângelo.
```