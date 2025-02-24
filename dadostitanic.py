import pandas as pd
import numpy as np
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
states = pd.read_csv('train.csv') #ou dessa forma
#csv = pd.read_csv(r'C:\Users\User\Desktop\titanic\train.csv') #Colocar o "R" para o python entender que é uma raw string, normalmente ele tende a barra assim: /
#pandas le o arquivo csv e carrega como um dataframe(estrutura de dados)
#print(states) #Visão geral da tabela
#states.info() #Informações da tabela

#calculo da media
media = states['Age'].mean()

#calculando a mediana
mediana = states['Age'].median()

#calculando o desvio padrão
desviopadrao = states['Age'].std()

#Calculando moda
moda = states['Age'].mode()


#proporção entre homens e mulheres embarcados
proporcoes = states['Sex'].value_counts (normalize= True)
#print(proporcoes)


#DUVIDA!!!!!!!!!!!!!!!!!
#Quartil de homens e mulheres
states['SexNumeric'] = states['Sex'].map({'male': 0, 'female': 1}) #Utiliza a func de dicionario map, convertendo str em int
quartis = states['SexNumeric'].quantile([0.25, 0.5, 0.75, 1.0]) #Numero já em int, quartis
#print(quartis) #Mostra que tinham mais mulheres que homens.

quartildade = states.groupby(['Sex', 'Survived'])['Age'].quantile([0.25, 0.5, 0.75, 1.0]) #Prioridade de escolha para botes de evacuação

modaidade = states.groupby(['Sex', 'Survived'])['Age'].apply(lambda x: x.mode().iloc[0]) #Para garantir que o resultado seja valor unico, primeiro valor da moda calculada com iloc[0] e Se houver empate (mais de um valor com a mesma frequência), ele seleciona o primeiro valor.
#print(quartildade)#Taxa de sobrevivência por gênero e idade.
#print('Moda:', modaidade)#Moda de sobrevivência de pessoas por genero e idade.

fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Primeiro gráfico: Gráfico de barras para proporções entre homens e mulheres
proporcoes = states['Sex'].value_counts(normalize=True)
proporcoes.plot(kind='bar', alpha=0.7, ax=axs[0, 0])  # Plotando no primeiro subgráfico
axs[0, 0].set_title("Proporção entre Homens e Mulheres Embarcados")
axs[0, 0].set_xlabel("Gênero")
axs[0, 0].set_ylabel("Proporção")
axs[0, 0].set_xticklabels(['Masculino', 'Feminino'], rotation=0)

# Segundo gráfico: Histograma para distribuição de homens e mulheres
states['SexNumeric'] = states['Sex'].map({'male': 0, 'female': 1})
states['SexNumeric'].plot(kind='hist', bins=2, edgecolor='black', alpha=0.7, ax=axs[0, 1])  # Plotando no segundo subgráfico
axs[0, 1].set_title('Distribuição de Homens e Mulheres')
axs[0, 1].set_xlabel('Sexo (0 = Masculino, 1 = Feminino)')
axs[0, 1].set_ylabel('Frequência')
axs[0, 1].set_xticks([0, 1])
axs[0, 1].set_xticklabels(['Masculino', 'Feminino'])

# Terceiro gráfico: Histograma de Quartis de Idade por Gênero e Sobrevivência
quartildade = states.groupby(['Sex', 'Survived'])['Age'].quantile([0.25, 0.5, 0.75, 1.0])  # Cálculo dos quartis
quartildade = quartildade.unstack()  # Reorganizando os dados para facilitar o plot
quartildade.plot(kind='bar', alpha=0.7, ax=axs[1, 0])  # Plotando no terceiro subgráfico
axs[1, 0].set_title('Quartis de Idade por Gênero e Sobrevivência')
axs[1, 0].set_xlabel('Gênero e Sobrevivência')
axs[1, 0].set_ylabel('Idade')
axs[1, 0].set_xticklabels(['Homem Não Sobreviveu', 'Homem Sobreviveu', 'Mulher Não Sobreviveu', 'Mulher Sobreviveu'], rotation=0)

# Quarto gráfico: Histograma da Moda de Idade por Gênero e Sobrevivência
modaidade = states.groupby(['Sex', 'Survived'])['Age'].apply(lambda x: x.mode().iloc[0])  # Moda de idade
modaidade.plot(kind='hist', bins=10, edgecolor='black', alpha=0.7, ax=axs[1, 1])  # Plotando no quarto subgráfico
axs[1, 1].set_title('Distribuição da Moda de Idade por Gênero e Sobrevivência')
axs[1, 1].set_xlabel('Idade (Moda)')
axs[1, 1].set_ylabel('Frequência')

# Ajustando o layout para que os gráficos não se sobreponham
plt.tight_layout()

# Exibindo os gráficos
plt.show()
