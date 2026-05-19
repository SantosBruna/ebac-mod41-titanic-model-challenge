# 📊 Projeto: Titanic — Machine Learning from Disaster

## 📌 Sobre o Projeto

Projeto desenvolvido como desafio de machine learning utilizando a base de dados do **Kaggle (Titanic)**, com foco na aplicação e comparação de algoritmos de classificação para prever a sobrevivência de passageiros ao naufrágio do Titanic.

O **objetivo principal** é classificar se um passageiro sobreviveu ou não com base em características demográficas e de embarque, comparando o desempenho de quatro abordagens: **XGBoost simples**, **XGBoost com hiperparâmetros**, **SVM simples** e **SVM com hiperparâmetros (kernel RBF)**. O melhor modelo é submetido à plataforma Kaggle para participar do campeonato.

---

## 🎯 Objetivos

* Carregar as bases de treino e teste do Kaggle (`train.csv` e `test.csv`)
* Tratar dados nulos com estratégias de imputação comparadas visualmente
* Realizar análise exploratória univariada com detecção de outliers
* Aplicar Label Encoding na variável `Sex` e One-Hot Encoding em `Embarked`
* Criar a variável derivada `Cabin_known` e remover colunas irrelevantes
* Analisar correlações entre variáveis e o target `Survived`
* Testar diferentes combinações de features para seleção da melhor configuração
* Treinar XGBoost com e sem hiperparâmetros usando Pipeline e Cross Validation
* Treinar SVM com e sem hiperparâmetros usando Pipeline e Cross Validation
* Comparar os modelos com ROC-AUC e relatório de classificação
* Submeter as previsões do melhor modelo ao Kaggle

---

## 📁 Estrutura do Projeto

```
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── plot_utils.py
│   ├── data_utils.py
│   └── model_utils.py
├── titanic_model_challenge.ipynb
├── submission.csv
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**
* **Pandas** - Manipulação e análise de dados
* **NumPy** - Operações numéricas
* **Matplotlib** - Visualização de dados
* **Seaborn** - Visualizações estatísticas
* **Scikit-learn** - Pré-processamento, Pipeline, GridSearchCV, StratifiedKFold, SVM e métricas de avaliação
* **XGBoost** - Modelagem preditiva com gradient boosting
* **Jupyter Notebook** - Ambiente de desenvolvimento

---

## 📊 Descrição dos Dados

O dataset contém informações de passageiros do Titanic com as seguintes variáveis:

| Variável | Descrição |
|----------|-----------|
| **PassengerId** | Identificador único do passageiro |
| **Survived** | Variável alvo — se o passageiro sobreviveu (0 = Não, 1 = Sim) |
| **Pclass** | Classe da passagem (1ª, 2ª ou 3ª) |
| **Name** | Nome do passageiro (removida na análise) |
| **Sex** | Gênero do passageiro |
| **Age** | Idade do passageiro |
| **SibSp** | Número de irmãos/cônjuges a bordo |
| **Parch** | Número de pais/filhos a bordo |
| **Ticket** | Número do bilhete (removida na análise) |
| **Fare** | Tarifa paga pela passagem |
| **Cabin** | Número da cabine (transformada em `Cabin_known`) |
| **Embarked** | Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 📈 Etapas Realizadas

### Pré-processamento

- Carregamento dos CSVs de treino e teste com `pandas`
- Verificação de dados nulos com cálculo percentual
- **Comparação de 3 estratégias de imputação para `Age`:**
  1. Imputação pela mediana global
  2. Imputação condicional pelo gênero (`Sex`)
  3. Imputação condicional por grupo (`Sex` + `Embarked`) ✅ **Escolhida**
- Análise univariada com `describe()`, comparação média/mediana e boxplots
- Detecção de outliers via método IQR para `Age` e `Fare` — mantidos após análise contextual

### Tratamento de Variáveis Categóricas

- **`Sex`**: Label Encoding (`male` → 0, `female` → 1)
- **`Embarked`**: One-Hot Encoding com `pd.get_dummies()` (gerou colunas `Embarked_C`, `Embarked_Q`, `Embarked_S`)
- **`Cabin`**: Transformada em `Cabin_known` (1 se cabine conhecida, 0 caso contrário)
- Colunas `Name`, `Ticket` e `Cabin` originais removidas

### Análise de Correlação e Seleção de Features

Correlação calculada com o target `Survived`. Três combinações de features foram testadas:

1. `['Sex', 'Fare', 'Cabin_known', 'Pclass', 'Embarked_S', 'Embarked_C']`
2. `['Sex', 'Fare', 'Cabin_known', 'Pclass', 'Embarked_S']` ✅ **Melhor resultado**
3. `['Sex', 'Fare', 'Cabin_known', 'Embarked_S']`

**Principais achados:**
- `Sex` apresentou a maior correlação com o target
- Colunas de `Embarked` com alta correlação entre si foram removidas para evitar redundância
- O dataset apresenta desbalanceamento entre as classes de `Survived`

---

### Modelagem — XGBoost

#### Pipeline sem Hiperparâmetros

```python
pipe_boost_simple = mu.criar_pipeline_xgboost(
    n_components=None,
    random_state=42
)
```

Avaliação com Cross Validation de 5 folds utilizando `roc_auc` como métrica.

#### Pipeline com Hiperparâmetros (GridSearchCV)

```python
param_grid_xgboost = {
    'modelo__learning_rate'    : [0.01, 0.05, 0.1, 0.3],
    'modelo__max_depth'        : [3, 4, 6],
    'modelo__n_estimators'     : [50, 100, 200],
    'modelo__subsample'        : [0.5, 0.8, 1.0],
    'modelo__colsample_bytree' : [0.5, 0.8, 1.0],
}
```

| Configuração | Valor |
|---|---|
| `cv` | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| `scoring` | `roc_auc` |
| `n_jobs` | `-1` |

---

### Modelagem — SVM

#### Pipeline sem Hiperparâmetros

```python
pipe_svm_simple = mu.criar_pipeline_svm(
    n_components=None,
    random_state=42
)
```

Avaliação com Cross Validation de 5 folds utilizando `roc_auc` como métrica.

#### Pipeline com Hiperparâmetros (GridSearchCV)

```python
param_grid_svm = [
    {
        'modelo__C'      : [10, 100],
        'modelo__kernel' : ['rbf'],
        'modelo__gamma'  : ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    },
]
```

| Configuração | Valor |
|---|---|
| `cv` | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| `scoring` | `roc_auc` |
| `n_jobs` | `1` |

---

### Submissão ao Kaggle

O melhor modelo (`melhor_modelo_xg`) foi utilizado para gerar previsões na base de teste do Kaggle e salvar o arquivo `submission.csv`:

```python
y_pred = melhor_modelo_xg.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred.astype(int)
})
submission.to_csv('submission.csv', index=False)
```

---

## ⚖️ Comparação dos Modelos

| Modelo | Observação |
|--------|------------|
| **XGBoost com hiperparâmetros** | Melhor desempenho geral — modelo escolhido para submissão |
| **XGBoost simples** | Bom desempenho como baseline |
| **SVM com hiperparâmetros (RBF)** | Resultado competitivo, porém inferior ao XGBoost |
| **SVM simples** | Menor desempenho entre os quatro |

O XGBoost com hiperparâmetros superou o SVM nesta base, sendo o modelo selecionado para a submissão ao Kaggle.

---

## 🔍 Principais Insights

1. **Sex** é a variável mais preditiva da sobrevivência, confirmada pela análise de correlação
2. O desbalanceamento das classes de `Survived` é um fator relevante para a avaliação — ROC-AUC foi preferida à acurácia simples
3. A estratégia de imputação condicional por grupo (`Sex` + `Embarked`) preservou melhor a distribuição original de `Age`
4. Outliers detectados em `Age` e `Fare` foram mantidos após análise contextual, pois representam passageiros reais
5. A combinação de features com `Embarked_S` (sem `Embarked_C`) resultou no melhor F1-score
6. O uso de Pipeline + StratifiedKFold garantiu consistência e evitou data leakage na avaliação

---

## 👩‍💻 Autora

**Bruna S. R. Santos**

* 🔗 LinkedIn: [www.linkedin.com/in/brunasrsantos](https://www.linkedin.com/in/brunasrsantos)
* 📧 Email: brunasrsantos@gmail.com

---

## 📝 Licença

Este projeto está licenciado sob a **MIT License**.
