# Importando bibliotecas necessárias
import pandas as pd

# Lendo o conjunto de dados
dados = pd.read_csv("creditcard.csv")

# Exibindo as primeiras 10 linhas do conjunto de dados
dados.head(10)

# Calculando o número de transações, fraudes e transações normais
n_transacoes = dados['Class'].count()
n_fraude = dados['Class'].sum()
n_normais = n_transacoes - n_fraude
fraudes_porc = n_fraude / n_transacoes
normais_porc = n_normais / n_transacoes

# Exibindo estatísticas das transações
print("Número de transações", n_transacoes)
print("Número de fraudes", n_fraude, "%.2f" %(fraudes_porc*100))
print("Número de transações normais: ", n_normais, "%.2f" %(normais_porc*100))

# Importando StratifiedShuffleSplit para divisão treino-teste
from sklearn.model_selection import StratifiedShuffleSplit

# Função para realizar a divisão dos dados para treino e teste
def executar_validador(x, y):
  validador = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
  for treino_id, teste_id in validador.split(x, y):
    x_train, x_test = x[treino_id], x[teste_id]
    y_train, y_test = y[treino_id], y[teste_id]
  return x_train, x_test, y_train, y_test

# Medindo o tempo de execução
%%time

# Importando o Classificador de Árvore de Decisão
from sklearn import tree

# Função para executar o Classificador de Árvore de Decisão
def executar_classificador(classificador, x_train, x_test, y_train):

  # Gerando a árvore de decisão
  arvore = classificador.fit(x_train, y_train)

  # Prevendo se as classes são fraude ou não
  y_pred = arvore.predict(x_test)
  return y_pred

# Importando o matplotlib para visualização da árvore
import matplotlib.pyplot as plt

# Função para salvar o gráfico da árvore de decisão como uma imagem
def salvar_arvore(classificador, nome):
  plt.figure(figsize=(200,100))
  tree.plot_tree(classificador, filled=True, fontsize=14)
  plt.savefig(nome)
  plt.close()

# Importando accuracy_score para avaliação do modelo
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Função para validar a árvore de decisão
def validar_arvore(y_test, y_pred):
  print("Acuracia: " , accuracy_score(y_test, y_pred))
  print("Matriz de Confusao: ", confusion_matrix(y_test, y_pred))
  print("Precisao: ", precision_score(y_test, y_pred))
  print("Recall: ", recall_score(y_test, y_pred))

# Executando o divisor de dados
x = dados.drop('Class', axis=1).values
y = dados['Class'].values
x_train, x_test, y_train, y_test = executar_validador(x, y)

# Executando o Classificador de Árvore de Decisão
classificador_arvore_decisao = tree.DecisionTreeClassifier()
y_pred_arvore_decisao = executar_classificador(classificador_arvore_decisao, x_train, x_test, y_train)

# Criando o gráfico da árvore de decisão
salvar_arvore(classificador_arvore_decisao, "arvore_decisao1.png")

# Validando a árvore de decisão
validar_arvore(y_test, y_pred_arvore_decisao)

print(classificador_arvore_decisao)
print(classificador_arvore_decisao.get_depth())

# Execução do classificador DecisionTreeClassifier
classificador_arvore_decisao = tree.DecisionTreeClassifier(max_depth=10, random_state=0)
y_pred_arvore_decisao = executar_classificador(classificador_arvore_decisao, x_train, x_test, y_train)

validar_arvore(y_test,y_pred_arvore_decisao)

# Execução do classificador DecisionTreeClassifier
classificador_arvore_decisao = tree.DecisionTreeClassifier(max_depth=10, random_state=0, min_samples_leaf=10)
y_pred_arvore_decisao = executar_classificador(classificador_arvore_decisao, x_train, x_test, y_train)

validar_arvore(y_test,y_pred_arvore_decisao)