from sklearn.cluster import KMeans
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def criar_pipeline_kmeans(n_clusters=5, random_state=42):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state))
    ])
    
    return pipeline

def treinar_pipeline(df, colunas, pipeline):
    X = df[colunas]
    pipeline.fit(X)
    return pipeline

def criar_pipeline_random_forest_cw(
    n_components=10,
    n_estimators=100,
    max_depth=None,
    class_weight='balanced'
):
    """
    Pipeline Random Forest usando class_weight para balanceamento (sem SMOTE).
    Parâmetros:
    -----------
    n_components : int
        Número de componentes do PCA.
    n_estimators : int
        Número de árvores. Padrão: 100.
    max_depth : int or None
        Profundidade máxima. Padrão: None.
    class_weight : str or None
        Peso das classes. Padrão: 'balanced'.
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('pca',    PCA(n_components=n_components)),
        ('modelo', RandomForestClassifier(
                       n_estimators=n_estimators,
                       max_depth=max_depth,
                       class_weight=class_weight,
                       random_state=42
                   ))
    ])

def criar_pipeline_svm(
    random_state=42,
    smote_k_neighbors=5,
    n_components=None,      # ← None = sem PCA por padrão
    C=1.0,
    kernel='rbf',
    gamma='scale',
    class_weight=None,
    probability=False
):
    steps = [
        ('scaler', RobustScaler()),
    ]

    if n_components is not None:
        steps.append(('pca', PCA(n_components=n_components)))   # ← só adiciona se informado

    steps += [
      ('balanceamento', SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)),
        ('modelo', SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability
            class_weight=class_weight,
            random_state=random_state
        ))
    ]

    return Pipeline(steps)  

def criar_pipeline_xgboost(
    random_state=42,
    smote_k_neighbors=5,
    n_components=None,      # ← None = sem PCA por padrão
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=None
):
    """
    Cria um pipeline com RobustScaler, PCA (opcional), SMOTE e XGBoost.

    Parâmetros:
    -----------
    random_state : int
        Semente para reprodutibilidade (SMOTE e modelo).
    smote_k_neighbors : int
        Número de vizinhos usados pelo SMOTE.
    n_components : int ou None
        Número de componentes do PCA. None = sem PCA.
    n_estimators : int
        Número de árvores (rodadas de boosting).
    max_depth : int
        Profundidade máxima das árvores. Padrão XGBoost = 6.
    learning_rate : float
        Taxa de aprendizado (eta).
    scale_pos_weight : float ou None
        Peso da classe positiva para desbalanceamento.

    Retorna:
    --------
    pipeline : imblearn.pipeline.Pipeline
    """
    steps = [
        ('scaler', RobustScaler()),
    ]

    if n_components is not None:
        steps.append(('pca', PCA(n_components=n_components)))

    xgb_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss',
    )

    if scale_pos_weight is not None:
        xgb_params['scale_pos_weight'] = scale_pos_weight

    steps += [
        ('balanceamento', SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)),
        ('modelo',        XGBClassifier(**xgb_params))
    ]

    return Pipeline(steps)

def criar_pipeline_random_forest(
    n_components=10,
    random_state=42,
    smote_k_neighbors=5,
    n_estimators=100,
    max_depth=None,
    class_weight=None
):
    """
    Cria um pipeline com StandardScaler, SMOTE e Random Forest.

    Parâmetros:
    -----------
    random_state : int
        Semente para reprodutibilidade (SMOTE e modelo).
    smote_k_neighbors : int
        Número de vizinhos usados pelo SMOTE.
    n_estimators : int
        Número de árvores da floresta.
    max_depth : int ou None
        Profundidade máxima das árvores. None = sem limite.
    class_weight : dict ou 'balanced' ou None
        Peso das classes no modelo. Útil se não quiser usar SMOTE.

    Retorna:
    --------
    pipeline : imblearn.pipeline.Pipeline
    """
    return Pipeline([
            ('scaler',        RobustScaler()),          # 1º: escala
            ('pca',           PCA(n_components=n_components)),  # 2º: reduz dimensão
            ('balanceamento', SMOTE(k_neighbors=smote_k_neighbors, random_state=42)),  # 3º: balanceia
            ('modelo',        RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                class_weight=class_weight,
                                random_state=42
                              ))
        ])


def criar_pipeline_regressao_logistica(
    n_components=10,
    random_state=42,
    smote_k_neighbors=5,
    C=1.0,
    max_iter=1000,
    class_weight=None
):
    """
    Cria um pipeline com StandardScaler, SMOTE e Regressão Logística.

    Parâmetros:
    -----------
    random_state : int
        Semente para reprodutibilidade (SMOTE e modelo).
    smote_k_neighbors : int
        Número de vizinhos usados pelo SMOTE.
    C : float
        Inverso da força de regularização do modelo.
    max_iter : int
        Número máximo de iterações do solver.
    class_weight : dict ou 'balanced' ou None
        Peso das classes no modelo. Útil se não quiser usar SMOTE.

    Retorna:
    --------
    pipeline : imblearn.pipeline.Pipeline
    """
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()), # 1º: escala
        ('pca',           PCA(n_components=n_components)),  # 2º: reduz dimensão
        ('balanceamento', SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)),  # 3º: balanceia
        ('modelo', LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])

    return pipeline

from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def criar_cross_validation(
    pipeline,
    X,
    y,
    folds=5,
    random_state=5,
    scoring='accuracy'
):
    """
    Executa cross-validation com KFold em um pipeline.

    Parâmetros:
    -----------
    pipeline : Pipeline
        Pipeline já criado (com scaler, balanceamento e modelo).
    X : array-like
        Features.
    y : array-like
        Target.
    folds : int
        Número de folds do KFold. Padrão: 5.
    random_state : int
        Semente para reprodutibilidade do KFold. Padrão: 5.
    scoring : str
        Métrica de avaliação. Padrão: 'accuracy'.

    Retorna:
    --------
    dict com KFold, pontuações por fold, média e desvio padrão.
    """
    crossvalidation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state
    )

    pontuacoes = cross_val_score(
        pipeline,
        X,
        y,
        cv=crossvalidation,
        scoring=scoring
    )

    print(f"Pontuações por fold : {pontuacoes}")
    print(f"Média               : {pontuacoes.mean():.4f}")
    print(f"Desvio padrão       : {pontuacoes.std():.4f}")

    return {
        'kfold': crossvalidation,
        'pontuacoes': pontuacoes,
        'media': pontuacoes.mean(),
        'desvio_padrao': pontuacoes.std()
    }

def extrair_resultados_kmeans_pipeline(pipeline, X):
    """
    Extrai os labels previstos e os centroides (na escala original)
    de um pipeline contendo um modelo KMeans.

    Parâmetros:
    - pipeline: objeto sklearn Pipeline já treinado.
        Deve conter:
            - 'scaler': etapa de padronização (ex: StandardScaler)
            - 'kmeans': modelo KMeans treinado

    - X: DataFrame ou array com os dados de entrada.
        Deve conter exatamente as mesmas colunas utilizadas no treinamento.
        Cada coluna representa uma feature do modelo, por exemplo:
            - 'Age': idade do cliente
            - 'Annual Income (k$)': renda anual
            - 'Spending Score': score de consumo

    Retorno:
    - labels: array com o cluster atribuído a cada linha de X
        Exemplo: [0, 1, 1, 2, 0, ...]

    - centroides: array com as coordenadas dos centroides
        no espaço original (antes da padronização).
        Cada linha representa um cluster e cada coluna corresponde
        a uma feature de X.

        Exemplo (3 clusters, 3 variáveis):
        [
            [idade, renda, score],
            [idade, renda, score],
            [idade, renda, score]
        ]
    """

    kmeans = pipeline.named_steps["kmeans"]
    scaler = pipeline.named_steps["scaler"]

    # Gera os labels com base nos dados de entrada
    labels = pipeline.predict(X)

    # Converte os centroides da escala padronizada para a escala original
    centroides = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, centroides