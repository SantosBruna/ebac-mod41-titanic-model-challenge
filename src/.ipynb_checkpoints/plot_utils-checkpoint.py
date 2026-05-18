import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import itertools
from plotly.subplots import make_subplots
import numpy as np

def plot_boxplot(df, coluna, titulo=None, ylabel="Valores"):
    """
    Plota um boxplot de uma coluna de um DataFrame.

    Parâmetros:
    - df: DataFrame
    - coluna: nome da coluna (str)
    - titulo: título do gráfico (opcional)
    - ylabel: rótulo do eixo Y (default: 'Valores')
    """
    
    df.boxplot(column=coluna)
    
    if titulo is None:
        titulo = f'Box Plot de {coluna}'
        
    plt.title(titulo)
    plt.ylabel(ylabel)
    plt.show()

    
def plot_histograma(
    df, 
    coluna, 
    bins=20, 
    kde=True, 
    color='blue',
    titulo=None,
    xlabel=None,
    ylabel='Frequência',
    figsize=(10, 6)
):
    """
    Plota um histograma de uma coluna de um DataFrame.

    Parâmetros:
    - df: DataFrame
    - coluna: nome da coluna (str)
    - bins: número de bins (default: 20)
    - kde: adicionar curva KDE (default: True)
    - color: cor do histograma
    - titulo: título do gráfico (opcional)
    - xlabel: rótulo do eixo X (opcional)
    - ylabel: rótulo do eixo Y (default: 'Frequência')
    - figsize: tamanho da figura
    """

    plt.figure(figsize=figsize)
    
    sns.histplot(
        data=df, 
        x=coluna, 
        bins=bins, 
        kde=kde, 
        color=color
    )

    if titulo is None:
        titulo = f'Histograma de {coluna}'
        
    if xlabel is None:
        xlabel = coluna

    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_countplot(
    df,
    coluna,
    titulo=None,
    xlabel=None,
    ylabel='Contagem',
    rotacao=45,
    figsize=(10, 5)
):
    """
    Plota um countplot de uma coluna categórica.

    Parâmetros:
    - df: DataFrame
    - coluna: nome da coluna (str)
    - titulo: título do gráfico (opcional)
    - xlabel: rótulo do eixo X (opcional)
    - ylabel: rótulo do eixo Y (default: 'Contagem')
    - rotacao: rotação dos labels do eixo X
    - figsize: tamanho da figura
    """

    plt.figure(figsize=figsize)

    sns.countplot(data=df, x=coluna)

    if titulo is None:
        titulo = f'Distribuição da coluna "{coluna}"'

    if xlabel is None:
        xlabel = coluna

    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotacao)

    plt.show()

    
def distribuicao(df):
    colunas = df.select_dtypes(include='number').columns
    n = len(colunas)
    
    fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(18, n * 1.5))
    axes = axes.flatten()
    
    for i, coluna in enumerate(colunas):
        sns.histplot(df[coluna], kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(coluna)
        axes[i].set_xlabel('')
    
    # oculta subplots vazios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribuição das variáveis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def mapa_correlacao(df):
    correlation_matrix = df.select_dtypes(include=['number', 'bool']).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Matriz de Correlação')
    plt.show()

def analisar_por_target(df, target, colunas=None):
    """
    Gera gráficos de análise de cada coluna em relação a uma variável target.
    
    Parâmetros:
        df      : DataFrame
        target  : str → nome da coluna target (ex: 'Credit Score')
        colunas : list → colunas a analisar. Se None, usa todas exceto o target
    """
    if colunas is None:
        colunas = [col for col in df.columns if col != target]

    for col in colunas:
        if df[col].dtype in ['int64', 'float64']:
            mediana = df.groupby(target)[col].median().reset_index()
            fig = px.bar(mediana, x=target, y=col,
                         title=f'Mediana de {col} por {target}')
        else:
            fig = px.histogram(df, x=col, color=target, barmode='group',
                               title=f'Distribuição de {col} por {target}')
        fig.show()

def analisar_correlacoes(df, colunas=None):
    """
    Gera gráficos de correlação entre as variáveis do dataset.
    
    Parâmetros:
        df      : DataFrame
        colunas : list → colunas a analisar. Se None, usa todas as numéricas
    """
    df_num = df.select_dtypes(include='number')

    if colunas is not None:
        df_num = df_num[colunas]

    for i, col1 in enumerate(df_num.columns):
        for col2 in df_num.columns[i+1:]:
            fig = px.scatter(
                df_num, x=col1, y=col2,
                title=f'Correlação entre {col1} e {col2}',
                trendline='ols'
            )
            fig.show()

def plot_pairplot(df, colunas=None, hue=None, titulo=None):
    """
    Plota um pairplot do DataFrame.

    Parâmetros:
    - df: DataFrame
    - colunas: lista de colunas (opcional)
    - hue: coluna para diferenciação de categorias (opcional)
    - titulo: título geral do gráfico (opcional)
    """

    if colunas is not None:
        df_plot = df[colunas]
    else:
        df_plot = df

    g = sns.pairplot(df_plot, hue=hue)

    if titulo:
        g.fig.suptitle(titulo, y=1.02)

    plt.show()

def plot_clusters(
    df,
    labels,
    centroides,
    colunas,
    titulo="Clusters"
):
    """
    Plota clusters para duas colunas específicas.

    Parâmetros:
    - df: DataFrame original
    - labels: array com labels dos clusters
    - centroides: array com centroides (escala original)
    - colunas: lista com 2 colunas [x, y]
    - titulo: título do gráfico
    """

    if len(colunas) < 2:
        raise ValueError("É necessário pelo menos duas colunas.")

    col_x, col_y = colunas[0], colunas[1]

    df_plot = df[colunas].copy()
    df_plot["cluster"] = labels.astype(str)

    fig = px.scatter(
        df_plot,
        x=col_x,
        y=col_y,
        color="cluster",
        opacity=0.7,
        title=titulo
    )

    fig.add_scatter(
        x=centroides[:, 0],
        y=centroides[:, 1],
        mode="markers",
        marker=dict(color="red", symbol="x", size=14),
        name="Centroides"
    )

    fig.update_layout(
        xaxis_title=col_x,
        yaxis_title=col_y,
        legend_title="Cluster"
    )

    fig.show()

def plot_clusters_combinacoes(
    df,
    labels,
    centroides,
    colunas
):
    """
    Plota clusters para todas as combinações de 2 colunas.

    Parâmetros:
    - df: DataFrame original
    - labels: array com labels dos clusters
    - centroides: array com centroides (escala original)
    - colunas: lista de colunas
    """

    if len(colunas) < 2:
        raise ValueError("É necessário pelo menos duas colunas.")

    combinacoes = list(itertools.combinations(range(len(colunas)), 2))

    for i, j in combinacoes:
        col_x = colunas[i]
        col_y = colunas[j]

        df_plot = df[colunas].copy()
        df_plot["cluster"] = labels.astype(str)

        fig = px.scatter(
            df_plot,
            x=col_x,
            y=col_y,
            color="cluster",
            title=f"Clusters: {col_x} vs {col_y}",
            opacity=0.7
        )

        fig.add_scatter(
            x=centroides[:, i],
            y=centroides[:, j],
            mode="markers",
            marker=dict(color="red", symbol="x", size=12),
            name="Centroides"
        )

        fig.show()

def metodo_cotovelo(
    x,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    titulo: str = "Seleção de K",
) -> dict:
    """
    Plota a curva do método do cotovelo e retorna as inércias calculadas.

    Parâmetros
    ----------
    x : Array ou dataframe já transformado
        Dataset de entrada.
    k_max : int
        Número máximo de clusters a testar (padrão: 10).
    random_state : int
        Semente para reprodutibilidade (padrão: 42).
    n_init : int
        Número de inicializações do KMeans (padrão: 10).
    titulo : str
        Título do gráfico.

    Retorno
    -------
    dict com:
        - 'inertias' : list[float] — inércia para cada k
        - 'silhouettes' : list[float] — silhouttes para cada k
        - 'k_range' : list[float] - k_ranges
    """

    dados_fit = x if not isinstance(x, pd.DataFrame) else x.values
    # ── Cálculo das inércias ──────────────────────────────────────────
    k_range = range(2, k_max + 1)
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = kmeans.fit_predict(dados_fit)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(dados_fit, labels))

    # ── Gráficos lado a lado ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Método do Cotovelo", "Silhouette Score")
    )

    ks = list(k_range)

    fig.add_trace(
        go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inércia"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ks, y=silhouettes, mode="lines+markers",
                   name="Silhouette", line=dict(color="orange")),
        row=1, col=2
    )

    fig.update_xaxes(title_text="k", tickmode="linear", dtick=1)
    fig.update_yaxes(title_text="Inércia (WCSS)", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_layout(title_text=titulo, template="plotly_white", showlegend=False)
    fig.show()

    return {
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_range": ks,
    }


def plot_distribuicao_categorica_por_cluster(df, coluna, cluster_col="cluster"):
    """
    Mostra a distribuição percentual de uma variável categórica dentro de cada cluster.
    """

    dist = (
        df.groupby(cluster_col)[coluna]
        .value_counts(normalize=True)
        .rename("proporcao")
        .reset_index()
    )

    fig = px.bar(
        dist,
        x=cluster_col,
        y="proporcao",
        color=coluna,
        barmode="stack",
        title=f"Distribuição de {coluna} por cluster"
    )

    fig.show()

def plot_comparacao_variavel_por_cluster(df, coluna, cluster_col="cluster"):
    """
    Compara como os clusters se distribuem dentro de cada categoria.
    """

    fig = px.histogram(
        df,
        x=coluna,
        color=cluster_col,
        barmode="group",
        title=f"{coluna} por cluster"
    )

    fig.show()

def heatmap_perfil_clusters(df, colunas_categoricas, cluster_col="cluster"):
    """
    Heatmap consolidado com TODAS as variáveis categóricas.
    """

    lista = []

    for col in colunas_categoricas:
        temp = (
            df.groupby(cluster_col)[col]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        # adiciona prefixo com nome da variável
        temp.columns = [f"{col} | {cat}" for cat in temp.columns]

        lista.append(temp)

    # junta tudo horizontalmente
    df_plot = pd.concat(lista, axis=1)

    plt.figure(figsize=(16, 6))
    sns.heatmap(df_plot, annot=True, cmap="Blues", fmt=".2f")

    plt.title("Perfil dos clusters")
    plt.xlabel("Variáveis / Categorias")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

def plot_clusters_colorido_por_categoria(
    df,
    labels,
    colunas,
    categoria
):
    df_plot = df[colunas].copy()
    df_plot["cluster"] = labels.astype(str)
    df_plot[categoria] = df[categoria]

    fig = px.scatter(
        df_plot,
        x=colunas[0],
        y=colunas[1],
        color=categoria,
        symbol="cluster",
        title=f"Clusters coloridos por {categoria}"
    )

    fig.show()

def plot_variancia_acumulada(cumulative_variance):
    """
    Plota a variância explicada acumulada por componente do PCA.

    Parâmetros:
    -----------
    cumulative_variance : array-like
        Array com os valores de variância acumulada por componente.
    """
    n_componentes = range(1, len(cumulative_variance) + 1)

    plt.plot(n_componentes, cumulative_variance, marker='o', linestyle='--')
    plt.title('Variância Explicada Acumulada por Componente')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.grid(True)
    plt.show()