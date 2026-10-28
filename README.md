# Titanic — Previsão de Sobrevivência com Random Forest

Modelo de classificação binária para a competição [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) do Kaggle. O objetivo é prever se um passageiro sobreviveu ao naufrágio a partir de dados como sexo, idade, classe e tarifa.

Mais do que maximizar o score, o projeto documenta **o porquê** de cada decisão: a escolha das features foi guiada por uma análise de **informação mútua (Mutual Information)**, e o desempenho foi estimado com **validação cruzada estratificada**.

O Teste é dividido entre dois arquivos CSV, um para treino e outro para validação, assim, evitando overfitting e uma checagem do modelo mais válida.

**Resultado:** acurácia de **83,05% ± 2,71%** em validação cruzada estratificada de 5 folds.

---

## Estrutura do projeto

```
.
├── app.py            # pipeline principal: feature engineering, validação cruzada, treino e submissão
├── mutual_info.py    # análise exploratória baseada em informação mútua
├── data/
│   ├── train.csv     # 891 passageiros com rótulo (Survived)
│   └── test.csv      # 418 passageiros sem rótulo
└── submission.csv    # gerado pelo app.py
```


## 1. Análise com Mutual Information

### Por que informação mútua?

A **informação mútua**  mede quanto saber o valor de uma feature `X` reduz a incerteza sobre o alvo `Y`. Ela vale 0 quando as variáveis são independentes e cresce conforme a dependência aumenta.

Comparada à correlação, ela tem duas vantagens importantes para este problema:

- **Captura relações não lineares.** A sobrevivência cai para quem viajou sozinho *e* para famílias muito grandes, mas é maior em famílias de 2 a 4 pessoas. Uma correlação linear não enxerga esse padrão em "U invertido"; a MI enxerga.
- **Funciona com variáveis categóricas** (título, porto de embarque, deck) sem precisar impor uma ordem artificial entre as categorias.

### Metodologia (`mutual_info.py`)

- **Estimador:** `mutual_info_classif` do scikit-learn.
- **Teste de permutação:** a MI foi recalculada 20 vezes com o alvo embaralhado. O percentil 95 dessa distribuição nula é o valor que uma feature sem relação nenhuma com o alvo atingiria por acaso. Só é relevante a feature que fica bem acima dele.
- **Normalização:** cada MI também aparece como fração da entropia de `Survived` . Isso responde de forma intuitiva "quanto da incerteza essa feature explica sozinha?".

### Relevância de cada feature para `Survived`

| Feature | MI (nats) | % de H(Survived) | Ruído p95 |
|---|---:|---:|---:|
| **Title** | 0,168 | **25,3%** | 0,005 |
| **Sex** | 0,151 | 22,7% | 0,001 |
| **Fare** | 0,130 | 19,5% | 0,018 |
| TicketGroup | 0,059 | 8,8% | 0,009 |
| Pclass | 0,058 | 8,7% | 0,002 |
| Deck | 0,055 | 8,3% | 0,008 |
| Age | 0,054 | 8,1% | 0,011 |
| HasCabin | 0,049 | 7,4% | 0,001 |
| FamilySize | 0,048 | 7,2% | 0,010 |
| SibSp | 0,023 | 3,5% | 0,007 |
| IsAlone | 0,021 | 3,1% | 0,002 |
| Parch | 0,016 | 2,5% | 0,006 |
| Embarked | 0,014 | 2,1% | 0,004 |
| AgeMissing | 0,004 | 0,7% | 0,003 |

### Principais conclusões

**`Title` é a feature mais informativa do dataset.** Sozinha, ela explica cerca de 25% da incerteza sobre a sobrevivência, mais que o próprio sexo. O título carrega a informação de sexo e, além disso, uma informação de **idade e status social**: "Master" identifica meninos, que tiveram prioridade nos botes, enquanto "Mr" identifica homens adultos, o grupo com a menor taxa de sobrevivência. Ou seja, o título codifica de forma compacta a norma *"mulheres e crianças primeiro"*.

**Há features que se complementam (sinergia).** A informação conjunta de duas features pode ser maior que a soma das informações individuais. Isso indica uma **interação** entre elas:

| Combinação | MI com Survived |
|---|---:|
| Sex sozinho | 0,151 |
| Pclass sozinho | 0,058 |
| Soma das individuais | 0,209 |
| **Sex + Pclass (conjunta)** | **0,218** |
| **Title + Pclass (conjunta)** | **0,249** |

O efeito do sexo depende da classe: mulheres da 3ª classe sobreviveram bem menos que mulheres da 1ª e 2ª. Juntos, `Title` e `Pclass` explicam cerca de 37% da incerteza. Modelos baseados em árvores capturam esse tipo de interação naturalmente, por meio de splits sucessivos, sem precisar criar termos de interação manualmente.

**`Fare` é mais informativa que `Pclass`** (0,130 contra 0,058). A tarifa é o valor do bilhete inteiro, compartilhado pelo grupo. Por isso ela agrega numa única variável contínua a classe, o tamanho do grupo e diferenças de status *dentro* da mesma classe.

**`Age` tem relevância moderada isoladamente** (0,054). O efeito se concentra nas crianças, e boa parte dele o `Title` ("Master") já captura.

**Distribuição treino × teste.** A MI de cada feature com a variável "pertence ao `test.csv`" ficou abaixo de 0,004 para todas. Isso equivale ao viés esperado do estimador para ~1300 amostras e indica **ausência de covariate shift**. Na prática, a estimativa da validação cruzada deve ser representativa do desempenho no conjunto de teste.

### Redundância entre features

A **informação mútua normalizada (NMI)** entre pares de features mede o quanto elas carregam a mesma informação (0 = independentes, 1 = equivalentes):

| Par | NMI |
|---|---:|
| HasCabin × Deck | 0,73 |
| **Sex × Title** | **0,71** |
| FamilySize × IsAlone | 0,70 |
| SibSp × FamilySize | 0,56 |

`Sex` e `Title` são **altamente redundantes**: todo "Mr" e "Master" é homem, e todo "Mrs" e "Miss" é mulher. Mesmo assim, **as duas foram mantidas**, de propósito:

- **Random Forest é robusta a multicolinearidade.** Diferente de uma regressão linear ou logística, cujos coeficientes ficam instáveis e difíceis de interpretar quando há features correlacionadas, uma árvore escolhe a cada split a feature que mais reduz a impureza (Gini). Uma feature redundante, no pior caso, simplesmente não é escolhida, e o desempenho preditivo não é prejudicado.
- **O ganho de informação não é nulo.** `Title` acrescenta a dimensão de idade/status que `Sex` não tem. Isso é confirmado empiricamente: adicionar `Title` aumentou a acurácia em validação cruzada (ver seção 4).

---

## 2. Feature engineering

| Feature | Tipo | Transformação |
|---|---|---|
| `Sex` | Categórica binária | Codificação binária: `male → 0`, `female → 1` |
| `Age` | Numérica | Imputação pela mediana condicional ao título |
| `Pclass` | Ordinal | Mantida como inteiro (1, 2, 3) |
| `SibSp`, `Parch` | Numéricas | Mantidas |
| `Fare` | Numéricam| Imputação pela mediana do treino (1 valor faltante no teste) |
| `Title` | Categórica | Extraída do nome via regex, categorias raras agrupadas, **one-hot encoding** |
| `FamilySize` | Numérica |

### `Sex` — codificação binária (label encoding de variável dicotômica)

Algoritmos do scikit-learn exigem entrada numérica. Como `Sex` tem apenas duas categorias, a codificação binária `{male: 0, female: 1}` é suficiente. Para variáveis dicotômicas ela é equivalente a um **one-hot encoding com `drop_first`**: gera uma única coluna, não cria uma ordem espúria e não introduz colunas redundantes.

### `Title` — extração do título a partir do nome

A coluna `Name` é texto livre e, crua, não tem valor preditivo. Mas ela contém um **título honorífico** ("Braund, **Mr.** Owen Harris") que concentra muita informação: sexo, faixa etária aproximada, estado civil e status social. A extração foi feita com a expressão regular ` ([A-Za-z]+)\.` e seguida de dois tratamentos:

1. **Normalização de sinônimos:** `Mlle` e `Ms` viram `Miss`, e `Mme` vira `Mrs`.
2. **Agrupamento de categorias raras (rare label encoding):** títulos com pouquíssimas ocorrências (`Dr`, `Rev`, `Col`, `Major`, `Countess`, `Jonkheer`...) foram agrupados em `Rare`. Categorias com 1 ou 2 exemplos levam a **overfitting**, porque o modelo memoriza indivíduos, e podem nem aparecer no teste.

O resultado tem 5 categorias (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`), codificadas com **one-hot encoding**. Por ser uma variável **nominal** (não existe ordem entre "Miss" e "Master"), um label encoding inteiro imporia uma ordem artificial. As colunas são alinhadas entre treino e teste com `reindex`, o que garante o mesmo espaço de features nos dois conjuntos mesmo se uma categoria faltar em um deles.

### `FamilySize` — feature derivada

`FamilySize = SibSp + Parch + 1` representa o tamanho total do grupo familiar a bordo. A análise de MI mostrou que ela é mais informativa (0,048) que `SibSp` (0,023) ou `Parch` (0,016) isoladamente, porque a relação com a sobrevivência depende do **tamanho do grupo**, não do tipo de parentesco. Essa relação é **não linear**: passageiros sozinhos e famílias com 5 ou mais pessoas sobreviveram menos que famílias pequenas. A Random Forest modela essa forma de "U invertido" com splits em limiares diferentes.

### Imputação de valores faltantes

- **`Age`** (~20% faltante): **imputação pela mediana condicional ao título**. Preencher com a mediana global atribuiria a mesma idade a um "Master" (menino, mediana ≈ 3,5 anos) e a um "Mr" (mediana ≈ 30 anos), o que distorceria justamente a informação de idade que se quer preservar. A **mediana** foi preferida à média por ser robusta a outliers.
- **`Fare`** (1 valor faltante no teste): mediana do treino.
- **Prevenção de data leakage:** as estatísticas de imputação (medianas) são calculadas **apenas no conjunto de treino** e reaproveitadas no teste. Usar informação do teste no pré-processamento vazaria dados que o modelo não deveria conhecer.

---

## 3. Modelo e validação cruzada

### Modelo

`RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)`

- **Random Forest:** um ensemble de árvores de decisão treinadas com *bagging* (amostras bootstrap) e subconjuntos aleatórios de features em cada split. Isso reduz a variância de uma árvore individual. É adequada para dados tabulares pequenos, com features mistas (numéricas e categóricas), interações e relações não lineares.
- **`max_depth=5`:** limita a profundidade das árvores como forma de **regularização**. Com apenas 891 amostras, árvores sem limite tendem a memorizar o treino (overfitting).

### Por que validação cruzada?

Com só **891 amostras rotuladas**, avaliar o modelo em um único split treino/validação (*holdout*) é pouco confiável. A acurácia muda bastante dependendo de quais passageiros caem na validação, ou seja, a **estimativa tem alta variância**. Além disso, separar 20% para validação desperdiça dados de treino que já são escassos.

A **validação cruzada k-fold** resolve os dois problemas. Os dados são divididos em *k* partes (folds), o modelo é treinado *k* vezes, cada vez validado em um fold diferente, e o desempenho final é a **média ± desvio padrão** das *k* execuções. Cada amostra é usada exatamente uma vez para validação, e o desvio padrão mostra a **estabilidade** do modelo.

Configuração usada: `StratifiedKFold(n_splits=5, shuffle=True, random_state=1)`

- **Estratificada:** as classes são desbalanceadas (38% sobreviveram, 62% não). A estratificação garante que cada fold mantenha essa proporção. Sem ela, um fold poderia ter proporções bem diferentes do conjunto completo, o que distorceria a métrica.
- **`shuffle=True`:** embaralha antes de dividir, evitando viés caso o CSV tenha alguma ordenação.
- **`random_state` fixo:** garante **reprodutibilidade**. Todos os conjuntos de features são comparados exatamente nos mesmos folds.

Depois da validação, o modelo final é retreinado com **100% do treino** para gerar a submissão. A validação cruzada serve para *estimar* o desempenho e *comparar* alternativas, não para produzir o modelo final.

---

## 4. Resultados

Comparação incremental dos conjuntos de features, todos avaliados nos mesmos 5 folds:

| Conjunto de features | Acurácia (CV) |
|---|---:|
| Baseline: Age, Sex, Pclass, SibSp, Parch | 81,82% ± 2,62% |
| + Fare | 82,15% ± 2,84% |
| + Fare + FamilySize | 81,93% ± 2,91% |
| + Fare + Title | 82,94% ± 2,80% |
| **Final: + Fare + Title + FamilySize** | **83,05% ± 2,71%** |

**Interpretação honesta dos números:** a ordem dos ganhos segue o que a análise de MI previa, e `Title` trouxe o maior salto (~+0,8 p.p. sobre "+ Fare"). Porém as diferenças entre conjuntos são **menores que o desvio padrão entre folds** (~2,7 p.p.). Com 891 amostras, não dá para afirmar com significância estatística que um conjunto é superior ao outro. Sozinha, `FamilySize` não melhorou a métrica (ela é uma combinação linear de `SibSp` e `Parch`, que já estão no modelo). Ela foi mantida no modelo final por ser mais informativa individualmente e mais interpretável.
