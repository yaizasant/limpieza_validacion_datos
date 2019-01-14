#!/usr/bin/env python
# -*- coding: utf-8 -*-

# importamos las librerias necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D


# [1] funcion para detectar outliers
def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)


# ------------------------ FUENTE DE DATOS, EXPLORACIÓN DE LOS DATOS -------------------- #
# convertimos los datos en un dataframe
df = pd.read_csv(r"C:\Users\Bomboncito\Documents\Master Ciencia de Datos\tipologia y ciclo de vida de los datos\Practica 2\winequality-red.csv")

# exploramos los datos para ver cuantos parametros tiene (12 columnas)
print(df.head())

# Controlando la cantidad de registros (1599)
print(df['quality'].count())

# para conocer las propiedades
print(df.columns)

# Ahora veamos algunas estadísticas de nuestros datos
print(df.describe())

# para conocer el tipo de las columnas
print(df.dtypes)

# Apartado 3. LIMPIEZA DE DATOS
# ------------------- SABER SI HAY QUE RELLENAR DATOS ---------------- #
# Controlando valores nulos; si resulta False el dataset esta limpio de valores faltantes
print(df.isnull().any().any())


# ---------------------------  SABER SI HAY RUIDO --------------------- #
# histograma [2]
plt.figure();
df.diff().hist(color='k', alpha=0.5, bins=50)
plt.show()

# outliers
for i in range(len(df.columns)):
    # descubrimiento de los outliers
    outliers = outliers_modified_z_score(df[df.columns[i]])
    print(len(outliers))
    print('Outliers de la columna ' + str(df.columns[i]) + ':')
    print(df.iloc[list(np.asarray(outliers)[0]),[i]])
    print(list(np.asarray(outliers)[0]))

    # diagramas de cajas para ver si son datos simétricos [3]
    boxplot = df.boxplot(column=[df.columns[i]])
    plt.show()


# Apartado 4. ANÁLISIS DE LOS DATOS
# a través del coeficiente de correlación y covarianza

# --------------- COMPROBACIÓN DE LA NORMALIDAD Y HOMOGENEIDAD DE LA VARIANZA------------ #

# pruebas de normalidad [4]
for i in range(len(df.columns)):

    print(df.columns[i])
    anderson_results = scipy.stats.anderson(df[df.columns[i]], dist='norm')
    print(anderson_results)
    if (anderson_results[1][2] > 0.05): # [5]
        print('No podemos rechazar la hipótesis nula de que la variable ' + str(df.columns[i])
              + ' proviene de una distribución normal')
    else:
        print('Podemos rechazar la hipótesis nula de que la variable ' + str(
            df.columns[i]) + ' proviene de una distribución normal')


# prueba de varianzas homogéneas [6]
# print(stats.levene(df['fixed acidity'],df['quality'])) # otra forma

for i in range(len(df.columns)-1):
    fligner_results = scipy.stats.fligner(df['quality'],df[df.columns[i]])
    print(fligner_results)
    print(fligner_results[1])
    if (fligner_results[1] > 0.05):
        print('La varianza de la variable ' + str(df.columns[i]) + ' y la variable quality son iguales')
    else:
        print('La varianza de la variable ' + str(df.columns[i]) + ' y la variable quality no son iguales')


# como las varianzas son distintas se aplica una modificación del test de Student
# t, p = stats.ttest_ind(data.Control.dropna(), data.Treatment.dropna(), equal_var = False) # [7]


# -------------------- ESTUDIO DE DEPENDENCIAS -------------------- #
# como las variables se han comprobado que siguen distribuciones normales[8]

for i in range(len(df.columns)-1):
    print('La correlación de la variable ' + str(df.columns[i]) + ' con la variable quality es')
    pearsonr_results = scipy.stats.pearsonr(df['quality'],df[df.columns[i]])
    print(pearsonr_results)


# -------------------- SELECCIÓN DE DATOS ------------------- #
# vamos a hacer grupos de datos en tres clases dependiendo de la calidad del vino
vino_bueno = df[df['quality'] >= 7.0]
vino_no_bueno = df[df['quality'] < 7.0]

print('Hay ' + str(len(vino_no_bueno)) + ' vinos no buenos')
print('Hay ' + str(len(vino_bueno)) + ' vinos buenos')

# ----------------------CONTRASTE DE HIPÓTESIS [9] ------------------- #
t, p = scipy.stats.ttest_ind(vino_bueno['alcohol'], vino_no_bueno['alcohol'], equal_var = False)
print(t, p)

# 4
# ------------------------------ TRANSFORMACIÓN DE LOS DATOS ----------------------------- #
normalized_df = (df - df.mean()) / df.std()
print(normalized_df.head())


# ------------- PREDICCIÓN CON REGRESIÓN MÚLTIPLE --------------- # [10]
df_lm = pd.DataFrame()
df_lm["alcohol"] = df["alcohol"]
df_lm["sulphates"] = df['sulphates']
df_lm['citric acid'] = df['citric acid']
df_lm['volatile acidity'] = df['volatile acidity']
df_lm['free sulfur dioxide'] = df['free sulfur dioxide']

# Se divide en training/testing sets
X_train = df_lm.iloc[:1280,]
X_test = df_lm.iloc[1280:,]


# La variable explicativa se divide en training/testing sets
y_train = df['quality'][:1280]
y_test = df['quality'][1280:]
print(len(y_train))
print(len(y_test))

# creamos el objeto para la regresión lineal
regr = linear_model.LinearRegression()

# Entrenamos el modelo
resultados_reg = regr.fit(X_train, y_train)

# Predecimos con el conjunto de test
y_pred = regr.predict(X_test)

# The coefficients
print('Coeficientes: \n', regr.coef_)
# The mean squared error [12]
print("Error cuadrático medio: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# Modelo 2
df_lm2 = pd.DataFrame()
df_lm2["alcohol"] = df["alcohol"]
df_lm2["sulphates"] = df['sulphates']
df_lm2['citric acid'] = df['citric acid']
df_lm2['free sulfur dioxide'] = df['free sulfur dioxide']

# Se divide en training/testing sets
X_train2 = df_lm2.iloc[:1280,]
X_test2 = df_lm2.iloc[1280:,]


# La variable explicativa se divide en training/testing sets
y_train2 = df['quality'][:1280]
y_test2 = df['quality'][1280:]
print(len(y_train2))
print(len(y_test2))

# creamos el objeto para la regresión lineal
regr2 = linear_model.LinearRegression()

# Entrenamos el modelo
resultados_reg2 = regr2.fit(X_train2, y_train2)

# Predecimos con el conjunto de test
y_pred2 = regr2.predict(X_test2)

# The coefficients
print('Coeficientes: \n', regr2.coef_)
# The mean squared error [12]
print("Error cuadrático medio: %.2f"
      % mean_squared_error(y_test2, y_pred2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test2, y_pred2))

# guardamos los resultados
df_resultados = pd.DataFrame()
df_resultados['y'] = y_test2
df_resultados['ypred'] = y_pred
df_resultados['ypred_2'] = y_pred2

writer = pd.ExcelWriter(r'C:\Users\Bomboncito\Documents\Master Ciencia de Datos\tipologia y ciclo de vida de los datos\Practica 2\resultados_modelo.xlsx')
df_resultados.to_excel(writer,'Sheet1')
writer.save()



# Bibliografia
# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
# https://www.datacamp.com/community/tutorials/pandas-read-csv
# [1] http://colingorrie.github.io/outlier-detection.html
# [2] https://pandas.pydata.org/pandas-docs/stable/visualization.html
# [3] https://pandas.pydata.org/pandas-docs/version/0.23.3/generated/pandas.DataFrame.boxplot.html
# [4] https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
# [4] https://eprints.ucm.es/45043/1/ProyectoDocente211.pdf
# [5] https://plot.ly/python/normality-test/
# [6] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.fligner.html#scipy.stats.fligner
# [7] http://pytolearn.csd.auth.gr/d1-hyptest/12/ttest-indep.html
# [8] https://www.youtube.com/watch?v=_uXQgDi7S3Q&feature=youtu.be [MINUTO 23]
# [9] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
# [10] http://www.aprendemachinelearning.com/regresion-lineal-en-espanol-con-python/
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# [12] http://ligdigonzalez.com/evaluando-el-error-en-los-modelos-de-regresion/