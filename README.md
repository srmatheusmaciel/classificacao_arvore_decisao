# Algoritmo de Classificação por Árvores de Decisão para Segmentação de Empresas

## Descrição do Projeto

Este projeto implementa um modelo de aprendizado de máquina utilizando árvores de decisão para classificar empresas em diferentes segmentos de clientes (Starter, Bronze, Silver e Gold) com base em características operacionais e financeiras. O algoritmo analisa dados como faturamento mensal, número de funcionários, localização, idade da empresa e índice de inovação para prever o segmento mais adequado para cada empresa.

## Objetivo

O principal objetivo é automatizar o processo de segmentação de clientes corporativos, permitindo uma estratégia de atendimento e oferta de produtos personalizada para cada segmento. O algoritmo classifica empresas em quatro categorias:

- **Starter**: Empresas em estágio inicial ou de pequeno porte
- **Bronze**: Empresas de porte pequeno a médio com potencial de crescimento
- **Silver**: Empresas de médio porte com bom desempenho financeiro
- **Gold**: Empresas de grande porte e alto valor

## Requisitos Técnicos e Bibliotecas

Para executar este algoritmo, você precisará das seguintes bibliotecas Python:

```python
# Manipulação de dados
import pandas as pd
import numpy as np

# Modelagem e avaliação
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Visualização
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay

# Persistência do modelo
import joblib

# Interface de implementação (opcional)
import gradio as gr
```

Requisitos de ambiente:
```
Python 3.11+
pip ou pipenv para gerenciamento de dependências
```

## Estrutura de Dados

O algoritmo espera um conjunto de dados com as seguintes variáveis:

| Coluna                 | Tipo     | Descrição                                      |
|------------------------|----------|------------------------------------------------|
| atividade_economica    | object   | Setor de atuação da empresa (ex: Comércio, Indústria) |
| faturamento_mensal     | float64  | Valor do faturamento mensal em reais           |
| numero_de_funcionarios | int64    | Quantidade de funcionários                      |
| localizacao            | object   | Cidade/Estado onde a empresa está localizada    |
| idade                  | int64    | Tempo de existência da empresa em anos          |
| inovacao               | int64    | Índice de inovação (escala de 0 a 10)          |
| segmento_de_cliente    | object   | Target - Categoria de segmentação (Starter, Bronze, Silver, Gold) |

Como visto nos dados, o dataset completo possui 500 entradas e todos os campos são obrigatórios (não há valores nulos).

## Instalação e Configuração

1. Clone o repositório:
```bash
git clone https://github.com/srmatheusmaciel/classificacao_arvore_decisao.git
cd classificacao_arvore_decisao
```

2. Crie um ambiente virtual e instale as dependências:
```bash
# Usando pip
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt

# Ou usando pipenv
pipenv install
pipenv shell
```

3. Estrutura do projeto:
```
classificacao_arvore_decisao/
├── .gradio/
│   ├── arquivo/
│   └── output/
├── datasets/
│   ├── dataset_segmento_cliente.csv
│   └── novas_empresas.csv
├── modelo_classificacao_decision_tree.pkl
├── predicoes.csv
├── README.md
├── requirements.txt
└── pipfile
```

## Uso do Algoritmo

### Treinamento do Modelo

Para treinar o modelo com seus próprios dados:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Carregar os dados
df_segmento = pd.read_csv('datasets/dataset_segmento_cliente.csv')

# Separar features e target
X = df_segmento.drop('segmento_de_cliente', axis=1)
y = df_segmento['segmento_de_cliente']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lista de variáveis categóricas
categorical_features = ['atividade_economica', 'localizacao']

# Criar pipeline para transformação de variáveis categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessamento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline completo com preprocessamento e modelo
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Treinar o modelo
dt_model.fit(X_train, y_train)

# Salvar o modelo treinado
joblib.dump(dt_model, 'modelo_classificacao_decision_tree.pkl')
```

### Fazendo Predições

Para usar o modelo para classificar novas empresas:

```python
import pandas as pd
import joblib

# Carregar o modelo treinado
modelo = joblib.load('modelo_classificacao_decision_tree.pkl')

# Carregar novos dados
df_empresas = pd.read_csv('datasets/novas_empresas.csv')

# Fazer predições
predicoes = modelo.predict(df_empresas)

# Adicionar as predições ao dataframe
df_predicoes = pd.DataFrame(predicoes, columns=['segmento_de_cliente'])
df_final = pd.concat([df_empresas, df_predicoes], axis=1)

# Salvar as predições
df_final.to_csv('predicoes.csv', index=False)

print("Predições realizadas com sucesso!")
```

### Interface Gradio para Predições em Lote

O projeto inclui uma interface Gradio para facilitar o uso:

```python
import gradio as gr
import joblib
import pandas as pd

# Carregar o modelo
modelo = joblib.load('./modelo_classificacao_decision_tree.pkl')

def predict(arquivo):
    # Ler arquivo CSV
    df_empresas = pd.read_csv(arquivo.name)
    
    # Fazer predições
    y_pred = modelo.predict(df_empresas)
    
    # Criar dataframe com resultados
    df_segmentos = pd.DataFrame(y_pred, columns=['segmento_de_cliente'])
    df_predicoes = pd.concat([df_empresas, df_segmentos], axis=1)
    
    # Salvar as predições
    output_path = './predicoes.csv'
    df_predicoes.to_csv(output_path, index=False)
    
    return output_path

# Interface Gradio
demo = gr.Interface(
    predict,
    gr.File(file_types=['.csv']),
    'file'
)

# Iniciar a aplicação
demo.launch()
```

## Como o Algoritmo Funciona

### Árvore de Decisão para Segmentação

O algoritmo utiliza uma árvore de decisão para classificação, que funciona da seguinte forma:

1. **Divisão dos dados**: O algoritmo analisa todas as features e encontra os pontos de corte que melhor separam os segmentos.

2. **Critério de separação**: Utiliza o índice Gini para medir a pureza dos nós (quanto menor o valor, mais homogêneo é o grupo).

3. **Estrutura hierárquica**: Como visto nas imagens da árvore, o algoritmo toma decisões em cascata:
   - Primeiro avalia se o índice de inovação é ≤ 2.5
   - Em seguida, analisa faixas de faturamento mensal
   - O processo continua criando ramificações até chegar a uma classificação final

A partir da visualização da árvore, podemos ver que:
- Empresas com baixa inovação (≤ 2.5) e baixo faturamento (≤ 425959.422) tendem a ser classificadas como Starter
- Empresas com baixa inovação e faturamento moderado a alto tendem a ser Silver
- Existem caminhos específicos na árvore que levam à classificação Gold, envolvendo combinações específicas de atributos

### Pré-processamento

O pipeline de pré-processamento:
1. Trata valores ausentes com SimpleImputer (substituindo pela moda)
2. Converte variáveis categóricas (atividade_economica, localizacao) em numéricas usando OneHotEncoder
3. Preserva variáveis numéricas em seu formato original

## Métricas de Avaliação

A performance do modelo foi avaliada usando:

### Matriz de Confusão
Como visto na imagem da matriz de confusão, o modelo apresenta algumas confusões entre classes, especialmente:
- Confusão entre Bronze e Silver
- Dificuldade em identificar corretamente empresas Gold e Starter

### Relatório de Classificação
```
              precision    recall  f1-score   support
      Bronze       0.41      0.36      0.38       202
        Gold       0.00      0.00      0.00        16
      Silver       0.51      0.63      0.57       260
     Starter       0.00      0.00      0.00        22
    accuracy                           0.47       500
   macro avg       0.23      0.25      0.24       500
weighted avg       0.43      0.47      0.45       500
```

### Cross-Validation
A acurácia média obtida via cross-validation foi de aproximadamente 47%, indicando que há espaço para melhorias no modelo.

## Limitações e Possíveis Melhorias

### Limitações Identificadas:
1. **Desbalanceamento de classes**: O conjunto de dados possui poucas empresas nas categorias Gold e Starter
2. **Baixa performance preditiva**: Acurácia de 47% indica que o modelo erra mais da metade das classificações
3. **Incapacidade de prever Gold e Starter**: O modelo apresenta dificuldade em classificar corretamente estas categorias

### Melhorias Sugeridas:
1. **Balanceamento de classes**: Técnicas como SMOTE ou under/oversampling para equilibrar o dataset
2. **Feature Engineering**: Criação de novas variáveis derivadas das existentes
3. **Hiperparâmetros**: Otimização dos parâmetros da árvore de decisão (profundidade máxima, mínimo de amostras por folha)
4. **Algoritmos alternativos**: Testar Random Forest, Gradient Boosting ou modelos ensemble
5. **Dados adicionais**: Incorporar mais variáveis preditivas relevantes para a segmentação

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.

## Contato

Para dúvidas, sugestões ou contribuições,abra uma issue ou entre em contato:
- Email: eumatheusmaciel@gmail.com

## Como Contribuir

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do projeto
2. Crie sua branch de feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

### Áreas para Contribuição
- Melhoria na performance do modelo
- Interface de usuário mais amigável
- Documentação adicional
- Testes unitários e de integração
- Novas features como visualização interativa da árvore de decisão
