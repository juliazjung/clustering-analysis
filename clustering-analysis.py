
from branalysis import Camara
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def distribuicao_partidos_por_cluster(labels, nome_algoritmo):    
    print(f"\n{'='*60}")
    print(f"Distribuição de partidos por cluster — {nome_algoritmo}")
    print(f"{'='*60}")
    
    partido_por_id = df_votos.groupby('parlamentar_id')['partido'].first()
    
    df = pd.DataFrame({
        'partido': [partido_por_id.get(pid, 'DESCONHECIDO') for pid in matriz_final.index],
        'cluster': labels
    })
    
    df = df[df['cluster'] != -1]
    
    # Total de parlamentares por partido (considerando todos os clusters)
    total_por_partido = df['partido'].value_counts()
    
    for cluster_id in sorted(df['cluster'].unique()):
        membros = df[df['cluster'] == cluster_id]
        total_cluster = len(membros)
        contagem = membros['partido'].value_counts()
        
        print(f"\nCluster {cluster_id} ({total_cluster} parlamentares):")
        print(f"  {'Partido':<15} {'N':<6} {'% cluster':<12} {'% partido'}")
        print(f"  {'-'*45}")
        for partido, n in contagem.items():
            pct_cluster = n / total_cluster * 100
            pct_partido = n / total_por_partido[partido] * 100
            print(f"  {str(partido):<15} {n:<6} {pct_cluster:<12.1f} {pct_partido:.1f}%")

def calcular_metricas(labels, time):
    silhouette = silhouette_score(matriz_final, labels)

    calinski = calinski_harabasz_score(matriz_final, labels)

    davies = davies_bouldin_score(matriz_final, labels)

    print(f"Silhouette Score:        {silhouette:.4f}")
    print(f"Calinski-Harabasz Index: {calinski:.4f}")
    print(f"Davies-Bouldin Index:    {davies:.4f}")
    print(f"Tempo de execução:       {time:.4f} segundos")


### Etapa 1 - Preparação dos dados ###

ANO_MIN = 2023
ANO_MAX = 2026

n_clusters = 7

parlamentares_totais = []
votacoes_totais = []
votos_totais = []

# Dicionário: (parlamentar_id, ano) -> partido
id_ano_para_partido = {}

for ano in range(ANO_MIN, ANO_MAX):
    plenario = Camara(ano)
    parlamentares_totais += plenario.parlamentares()
    votacoes_totais += plenario.votacoes()
    votos_totais += plenario.votos()

df_votos = pd.DataFrame([{
    'parlamentar_id': voto.parlamentar.id,
    'parlamentar_nome': voto.parlamentar.nome,
    'partido': voto.partido,
    'votacao_ano': voto.votacao.ano,
    'uf': voto.parlamentar.uf,
    'votacao_id': voto.votacao.id,
    'voto': voto.voto
} for voto in votos_totais])

mapeamento = {
    'SIM': 1, 'NÃO': -1, 'ABSTENÇÃO': 0,
    'OBSTRUÇÃO': 0, 'ARTIGO 17': 0, 'SEM REGISTRO': 0
}

df_votos['voto'] = df_votos['voto'].str.upper().str.strip()
df_votos['voto_numerico'] = df_votos['voto'].map(mapeamento)

matriz = df_votos.pivot_table(
    index='parlamentar_id',
    columns='votacao_id',
    values='voto_numerico',
    aggfunc='first'
)

total_votacoes = matriz.shape[1]
participacao = matriz.notna().sum(axis=1) / total_votacoes
parlamentares_ativos = participacao[participacao >= 0.20].index
matriz_final = matriz.loc[parlamentares_ativos].fillna(0)

print(f"Shape final da matriz: {matriz_final.shape}")

### Etapa 2 - Rodando os três algoritmos ###

# K-means
km = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto',
            max_iter=300, tol=1e-4, random_state=42, algorithm='lloyd')
start = time.time()
km_labels = km.fit_predict(matriz_final)
km_time = time.time() - start

print("Métricas calculadas — K-means:")
calcular_metricas(km_labels, km_time)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
start = time.time()
agg_labels = agg.fit_predict(matriz_final)
agg_time = time.time() - start

print("Métricas calculadas — Agglomerative:")
calcular_metricas(agg_labels, agg_time)

# HDBSCAN
hdb = HDBSCAN(min_cluster_size=5, min_samples=None,
              metric='euclidean', cluster_selection_method='eom', alpha=1.0)
start = time.time()
hdb_labels = hdb.fit_predict(matriz_final)
hdb_time = time.time() - start

n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_ruido = sum(hdb_labels == -1)
print(f"HDBSCAN — Clusters: {n_clusters_hdb}, Ruído: {n_ruido}")

print("Métricas calculadas — HDBSCAN:")
calcular_metricas(hdb_labels, hdb_time)

# Etapa 3
# Análise de distribuição dos partidos por cluster

distribuicao_partidos_por_cluster(km_labels, "K-means")
distribuicao_partidos_por_cluster(agg_labels, "Agglomerative")
distribuicao_partidos_por_cluster(hdb_labels, "HDBSCAN")

### PCA (aplicado uma única vez) ###

pca = PCA(n_components=2)
matriz_pca = pca.fit_transform(matriz_final)
variancia = pca.explained_variance_ratio_
variancia_total = sum(variancia)
print(f"Variância explicada: PC1={variancia[0]:.1%}, PC2={variancia[1]:.1%}, Total={variancia_total:.1%}")

### Preparando cores por partido ###

partido_por_id = df_votos.groupby('parlamentar_id')['partido'].first()
partidos_ordenados = [partido_por_id.get(pid, 'DESCONHECIDO') for pid in matriz_final.index]

# Lista de partidos únicos
partidos_unicos = sorted(set(partidos_ordenados))
n_partidos = len(partidos_unicos)

# Gerar uma cor para cada partido usando colormap
cmap = plt.cm.get_cmap('tab20', n_partidos)
cor_map = {partido: cmap(i) for i, partido in enumerate(partidos_unicos)}

# Lista de cores na mesma ordem que matriz_final
cor_por_partido = [cor_map[p] for p in partidos_ordenados]

# Legenda
patches = [mpatches.Patch(color=cor_map[p], label=p) for p in partidos_unicos]

### Visualização — 2 linhas x 3 colunas ###

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    f'Comparação dos Algoritmos de Clusterização\n'
    f'PCA: PC1={variancia[0]:.1%}, PC2={variancia[1]:.1%}, Total={variancia_total:.1%}',
    fontsize=13, fontweight='bold', y=1.01
)

algoritmos = [
    ('K-means',            km_labels,  n_clusters),
    ('Agglomerative',      agg_labels, n_clusters),
    ('HDBSCAN',            hdb_labels, n_clusters_hdb),
]

# LINHA 1 — colorido por cluster
for col, (nome, labels, n_clusters) in enumerate(algoritmos):
    ax = axes[0, col]

    mask_validos = labels != -1

    scatter = ax.scatter(
        matriz_pca[mask_validos, 0],
        matriz_pca[mask_validos, 1],
        c=labels[mask_validos],
        cmap=plt.cm.get_cmap('tab10', len(np.unique(labels[mask_validos]))),
        vmin=labels[mask_validos].min(),
        vmax=labels[mask_validos].max(),
        alpha=0.6,
        s=30
    )

    # Outliers do HDBSCAN em cinza
    if -1 in labels:
        mask_ruido = labels == -1
        ax.scatter(
            matriz_pca[mask_ruido, 0],
            matriz_pca[mask_ruido, 1],
            c='lightgray',
            alpha=0.3,
            s=20,
            label=f'Ruído (n={n_ruido})'
        )
        ax.legend(fontsize=8)

    ax.set_title(f'{nome} — por cluster (k={n_clusters})', fontsize=11)
    ax.set_xlabel(f'PC1 ({variancia[0]:.1%})', fontsize=9)
    ax.set_ylabel(f'PC2 ({variancia[1]:.1%})', fontsize=9)
    plt.colorbar(scatter, ax=ax, label='Cluster')

# LINHA 2 — colorido por partido
for col, (nome, labels, n_clusters) in enumerate(algoritmos):
    ax = axes[1, col]

    scatter2 = ax.scatter(
        matriz_pca[:, 0],
        matriz_pca[:, 1],
        c=cor_por_partido,
        alpha=0.6,
        s=30
    )

    ax.set_title(f'{nome} — por partido', fontsize=11)
    ax.set_xlabel(f'PC1 ({variancia[0]:.1%})', fontsize=9)
    ax.set_ylabel(f'PC2 ({variancia[1]:.1%})', fontsize=9)

# Legenda apenas no último gráfico
axes[1, 2].legend(handles=patches, loc='best', fontsize=6, ncol=2)

grafico_nome = f'grafico_pca_k{n_clusters}_{ANO_MIN}-{ANO_MAX}.png'

plt.tight_layout()
plt.savefig(grafico_nome, dpi=300, bbox_inches='tight')
plt.show()
print("Gráfico salvo como ${grafico_nome}")

