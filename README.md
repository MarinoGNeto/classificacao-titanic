# Titanic Dataset Analysis and k-NN Classification

Este projeto realiza **análise exploratória e limpeza de dados** em um conjunto de dados do Titanic, seguido pela aplicação de um modelo de classificação utilizando o algoritmo **k-Nearest Neighbors (k-NN)**. 

O código foi projetado para ser executado em um ambiente como **Google Colab** e utiliza bibliotecas comuns em ciência de dados, como **pandas** e **scikit-learn**.
---

## Etapas do Projeto

### Parte 1: Pré-processamento dos Dados

1. **Carregar a Base de Dados:**
   - Utilizamos **pandas** para carregar e visualizar os dados:
     ```python
     import pandas as pd  # Utilize a biblioteca pandas para carregar o arquivo CSV.

     genderSubmissionFile = pd.read_csv('gender_submission.csv')
     testFile = pd.read_csv('test.csv')
     trainFile = pd.read_csv('train.csv')

     # Explore os dados usando comandos como df.head(), df.info() e df.describe().
     print('----- Gender Submission File data -----\n')
     print(genderSubmissionFile.head())
     print(genderSubmissionFile.info())
     print(genderSubmissionFile.describe())

     print('\n----- Test File data -----\n')
     print(testFile.head())
     print(testFile.info())
     print(testFile.describe())

     print('\n----- Train File data -----\n')
     print(trainFile.head())
     print(trainFile.info())
     print(trainFile.describe())
     ```

2. **Limpeza de Dados:**
   - Identificar valores ausentes:
     ```python
     print('\nIdentificando valores ausentes em arquivo gender_submission.csv... ')
     print(genderSubmissionFile.isnull().sum())

     print('\nIdentificando valores ausentes em arquivo test.csv... ')
     print(testFile.isnull().sum())

     print('\nIdentificando valores ausentes em arquivo train.csv... ')
     print(trainFile.isnull().sum())

     # Preencher valores ausentes na coluna 'Age' com a mediana
     testFile['Age'] = testFile['Age'].fillna(testFile['Age'].median())
     testFile['Fare'] = testFile['Fare'].fillna(testFile['Fare'].median())
     testFile.dropna(subset=['Cabin'], inplace=True)

     trainFile['Age'] = trainFile['Age'].fillna(trainFile['Age'].median())
     trainFile.dropna(subset=['Cabin'], inplace=True)
     trainFile.dropna(subset=['Embarked'], inplace=True)

     print('\nSegue os valores que se encontram ausentes durante o tratamento do arquivo:')
     print(trainFile.isnull().sum())
     ```

3. **Seleção de Variáveis:**
   - Converter 'Sex' para variáveis numéricas:
     ```python
     testFile = pd.get_dummies(testFile, columns=['Sex'], drop_first=True)
     trainFile = pd.get_dummies(trainFile, columns=['Sex'], drop_first=True)
     ```

4. **Divisão dos Dados:**
   - As variáveis de entrada (X) e saída (y) são selecionadas:
     ```python
     X = trainFile[['Pclass', 'Sex_male', 'Age', 'Fare']]
     y = trainFile['Survived']
     ```

---

### Parte 2: Implementação do Modelo k-NN

1. **Treinamento do Modelo:**
   - O conjunto de dados é dividido em treino e validação (70/30):
     ```python
     from sklearn.model_selection import train_test_split

     # Dividir os dados de treino em treino e validação (70/30)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

     # Treinar o modelo com k=3
     from sklearn.neighbors import KNeighborsClassifier
     knn = KNeighborsClassifier(n_neighbors=3)
     knn.fit(X_train, y_train)
     ```

2. **Previsões:**
   - Realiza previsões com os dados de teste:
     ```python
     y_pred = knn.predict(X_test)
     print('Previsão:', y_pred)
     ```

---

### Parte 3: Avaliação de Desempenho

1. **Acurácia:**
   - Avaliação do desempenho usando a métrica de **acurácia**:
     ```python
     from sklearn.metrics import accuracy_score

     # Avaliar desempenho
     accuracy = accuracy_score(y_test, y_pred)
     print(f'Acurácia: {accuracy:.2f}')
     ```

2. **Matriz de Confusão:**
   - Visualização dos erros com a **matriz de confusão**:
     ```python
     from sklearn.metrics import confusion_matrix

     cm = confusion_matrix(y_test, y_pred)
     print("Matriz de Confusão:")
     print(cm)
     ```

---

## Dependências

Certifique-se de que as seguintes bibliotecas estão instaladas:
```bash
pip install pandas scikit-learn