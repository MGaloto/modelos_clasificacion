import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np

#%%


input_file = "winemag-data_first150k.csv"
df_pandas = pd.read_csv(input_file , index_col=0)
df_pandas['id'] = df_pandas.index
df_pandas




#%%

# Reemplazo NaN por sin descripcion -> valores nulos
df_pandas = df_pandas[df_pandas['price'] > 0]
df_pandas = df_pandas[df_pandas['points'] > 0]
df_pandas = df_pandas.fillna('sin_descripcion')



#%%


df_pandas.describe()



#%%

#Grafico histograma para validar la distribución de los puntos
fig = px.histogram(df_pandas, x="points",
                   title='Histograma Puntaje',
                   opacity=0.8,
                   log_y=False, # represent bars with log scale
                   color_discrete_sequence=['indianred'] # color of histogram bars
                   )
fig.show()

#%%

# Ubicación de los vinos con mayor cantidad de calificaciones
# Agrupo el dataset por Ubicacion (país, provincia y región) + Suma de reviews y score promedio. Ordenados de manera descendente por suma de reviews
df_pandas[['id','country', 'province', 'region_1', 'points']].groupby(by=["country", "province", "region_1"]).agg(
    reviews_count = pd.NamedAgg(column = "id", aggfunc = "count"),
    max_score = pd.NamedAgg(column = "points", aggfunc=max),
    avg_score = pd.NamedAgg(column = "points", aggfunc=np.mean)).sort_values(by=['reviews_count'], ascending=False).head(50)



#%%

# Ubicación de los vinos con mayor score promedio
# Agrupo el dataset por Ubicacion (país, provincia y región) + Suma de reviews y score promedio. Ordenados de manera descendente por score promedio
df_pandas[['id','country', 'province', 'region_1', 'points']].groupby(by=["country", "province", "region_1"]).agg(
    reviews_count = pd.NamedAgg(column = "id", aggfunc = "count"),
    max_score = pd.NamedAgg(column = "points", aggfunc=max),
    avg_score = pd.NamedAgg(column = "points", aggfunc=np.mean)).sort_values(by=['avg_score'], ascending=False).head(50)



#%%

### Correlación
sns.heatmap(df_pandas.corr(), annot = True)

#%%

Machine Learning - Gradient Boosted Trees (GBT)
Gradient boosting es una técnica de machine learning orientada tanto a problemas de regresión como de clasificación. Produce un modelo de predicción del tipo "ensable", típicamente utilizando un árbol de decisión.

#%%

### Es necesario convertir columnas categoricas a numéricas
df_pandas_pred = df_pandas 
df_pandas_pred['country_num'] = df_pandas_pred['country'].astype('category').cat.codes
df_pandas_pred['designation_num'] = df_pandas_pred['designation'].astype('category').cat.codes
df_pandas_pred['province_num'] = df_pandas_pred['province'].astype('category').cat.codes
df_pandas_pred['region_1_num'] = df_pandas_pred['region_1'].astype('category').cat.codes
df_pandas_pred['variety_num'] = df_pandas_pred['variety'].astype('category').cat.codes
df_pandas_pred['winery_num'] = df_pandas_pred['winery'].astype('category').cat.codes




#%%

df_pandas_pred.describe()


#%%

df_pandas_pred = df_pandas_pred[["country_num", "designation_num", "province_num", "region_1_num", "variety_num", "winery_num", "price", "points"]]



from sklearn.model_selection import train_test_split

train, test = train_test_split(df_pandas_pred, test_size=0.2)

print ("Tenemos %d observaciones para el entrenamiento y %d para prueba/testing." %(train.count(), test.count()))


#%%

from pyspark.ml.regression import GBTRegressor

# Definimos el modelo -> Gradient Boosted Trees (GBT).
gbt = GBTRegressor(featuresCol='features',
                   labelCol='points',
                   predictionCol="Prediction_points",
                   seed=43)



#%%


#%%