'''
Nombre de Grupo: alc boys
Participantes: Clement Julian Ignacio, Ludueña Dalmiro Paz, Ser Gonzalo Nicolas
Trabajo Práctico 02 Laboratorio de Datos Verano 2025
'''
#%%
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import numpy as np
import pandas as pd
import duckdb as dd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import tree
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram , cut_tree

#%% Funciones
def sacar_promedio(numero_int):
    numero = dd.sql(f"""
                          SELECT * EXCLUDE('Unnamed: 0', labels)
                          FROM mnist
                          WHERE labels = {numero_int}
                          """).df()
    promedio_numero = list()

    for i in range(784):
        promedio_numero.append([i,np.mean(numero.iloc[:, i])])
    
    return promedio_numero

def comparar_promedios(promedio1: list[int], promedio2: list[int], umbral: int) -> int:
    promedio_res = list()
    for i in range(784):
        if promedio1[i] > umbral and promedio2[i] > umbral:
            if promedio1[i] > promedio2[i]:
                promedio_res.append(promedio2[i]/promedio1[i])
            else:
                promedio_res.append(promedio1[i]/promedio2[i])            
        
    return np.mean(promedio_res)

def devolver_sin_repetidos(lista1,lista2):
    res = list()
    for elem in lista1:
        if elem not in lista2:
            res.append(elem)
    return res

def devolver_repetidos(lista1,lista2):
    res = list()
    for elem in lista1:
        if elem in lista2:
            res.append(elem)
    return res

def elimina_intensidades(promedios, umbral):
    for tupla in promedios:
        if tupla[1] < umbral:
            tupla[1]=0
            
def devolver_tuplas_sin_repetidos(lista1,lista2):
    res = list()
    for elem in lista1:
        if elem[0] not in lista2:
            res.append(elem)
    return res

def pixeles_brillosos(promedios, rango, umbral):
    pixeles_brillosos = dict()
    
    for numero in rango:
        pixeles_brillosos[numero] = list()
        for pixel in promedios[numero]:
            if not pixel[1] < umbral:
                pixeles_brillosos[numero].append(pixel[0])
    
    for clave,valor in promedios.items():
        img = np.array([promedios[clave][i][1] if promedios[clave][i][1] >= umbral else 0 for i in range(784)]).reshape((28,28))
        plt.imshow(img, cmap='gray', vmin= 0, vmax=255)
        plt.show()
        
    return pixeles_brillosos

def pixeles_unicos(pixeles_brillosos):    
    pixeles_unicos = dict()

    for clave, valor in pixeles_brillosos.items():
        pixeles_unicos[clave] = valor
        for clave2, valor2 in pixeles_brillosos.items():
            if clave != clave2:
                pixeles_unicos[clave] = devolver_sin_repetidos(pixeles_unicos[clave], valor2)
                
    return pixeles_unicos

def pixeles_compartidos(pixeles_brillosos):    
    pixeles_compartidos = dict()

    for clave, valor in pixeles_brillosos.items():
        pixeles_compartidos[clave] = valor
        for clave2, valor2 in pixeles_brillosos.items():
            if clave != clave2:
                pixeles_compartidos[clave] = devolver_repetidos(pixeles_compartidos[clave], valor2)
        return pixeles_compartidos[clave]
    
#%% Copiar la ruta de acceso de su archivo del mnist.csv

mnist = pd.read_csv("C:/Users/julia/OneDrive/Escritorio/Facultad/Labo de Datos/mnist_c_fog_tp.csv")

#%%

# Análisis exploratorio

cantidad_datos = len(mnist) 
print(f"En total, el dataframe MNIST-C tiene {cantidad_datos} datos.")
cantidad_atributos = len(mnist.iloc[0])
print(f"En total, el dataframe MNIST-C tiene {cantidad_atributos} atributos. Entre los que se encuentran píxeles del 0 al 783, 'label' indicando el número y un índice bajo el nombre de 'Unnamed: 0'.")
print("Y, como clases, nos encontramos con los dìgitos del 0 al 9.")


#%%

# 1.a Atributos relevantes ¿Alguno se puede descartar? Más allá de los labels que identifican cada dibujo con un número
#     queremos ver qué píxeles nos interesan y cuales no.

mnist_img = mnist.drop(columns=['labels', 'Unnamed: 0'])
media_pixel = np.zeros(784)
for i in range(784):
    media_pixel[i] = np.mean(mnist_img.iloc[:, i])


# Veamos qué imagen forma el promedio de todos las imágenes a la vez

img = np.array(media_pixel).reshape((28,28))
plt.imshow(img, cmap='gray')
plt.title("Promedio por píxel de todas las clases")
plt.show()

# Claramente nos interesan aquellos píxeles más blancos ¿qué píxeles descartamos?
# Seleccionamos arbitrariamente los píxeles que no nos interesan

umbral = 100

descarte = list()
for i in range(len(media_pixel)):
    if media_pixel[i] < umbral:
        descarte.append(media_pixel[i])
    else:
        descarte.append(0)


img = np.array(descarte).reshape((28,28))
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Píxeles a descartar (gris)')
plt.show()

# Ahora tenemos los que sí nos interesan

relevante = list()
for i in range(len(media_pixel)):
    if media_pixel[i] > umbral:
        relevante.append(media_pixel[i])
    else:
        relevante.append(0)
        
img = np.array(relevante).reshape((28,28))
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Píxeles relevantes (gris)')
plt.show()

# Por lo tanto, podemos enfocarnos en la mitad de los píxeles
print("Por lo que descartamos un " + str(round((len([i for i in descarte if i != 0])/784)*100)) + " porciento del total y nos quedamos con tan solo el " + str(round((len([i for i in relevante if i != 0])/784)*100)) + " porciento.")


#%%

"""
1.b
Efectivamente hay números que a priori tienen un mayor parecido con unos que con otros. Entre ellos identificamos las 
siguientes relaciones:
    - 1 con 7
    - 8 con 3 (pudiendo incluir al 0)
    - 5 con 6
    - 4 con 9
Procedemos a hacer el promedio de cada número y compararlos el uno con su parecido para ver el porcentaje de parecido.
Decidimos usar la  media debido a que los rangos de los valores no son muy grandes pero si sensibles. 
"""

# Vamos a buscar el promedio de cada número

promedios = dict()

for i in [1,3,8]:
    promedios[i] = sacar_promedio(i)

# Vemos que mantenemos la forma escencial de cada número

for clave, valor in promedios.items():
    for tupla in valor:
        if tupla[1] < 120:
            tupla[1] = 0
    img = np.array([promedios[clave][i][1] for i in range(784)]).reshape((28,28))
    plt.imshow(img, cmap='gray', vmin= 0, vmax=255)
    plt.show()    
    

# Agarramos todos los pixeles de interes

pixeles_mayores_120_1_3 = pixeles_brillosos(promedios, [1,3], 120)
pixeles_mayores_120_3_8 = pixeles_brillosos(promedios, [3,8], 120)


# Ahora filtramos por los pixeles que solo estan en un numero en particular

pixeles_no_compartidos_1_3 = pixeles_unicos(pixeles_mayores_120_1_3)
pixeles_no_compartidos_3_8 = pixeles_unicos(pixeles_mayores_120_3_8)
pixeles_compartidos_1_3 = pixeles_compartidos(pixeles_mayores_120_1_3)
pixeles_compartidos_3_8 = pixeles_compartidos(pixeles_mayores_120_3_8)

print(" ")
print("El 1 y 3 tienen " + str(len(pixeles_no_compartidos_1_3[1])) + " y " +  str(len(pixeles_no_compartidos_1_3[3])) +  " píxeles únicos, respectivamente. Mientras que comparten "+ str(len(pixeles_compartidos_1_3)) + " píxeles")
print("El 3 y 8 tienen " + str(len(pixeles_no_compartidos_3_8[3])) + " y " +  str(len(pixeles_no_compartidos_3_8[8])) +  " píxeles únicos, respectivamente. Mientras que comparten "+ str(len(pixeles_compartidos_3_8)) + " píxeles")
                    

                    


pixeles_1_3 = dict()
for clave,valor in pixeles_no_compartidos_1_3.items():
    pixeles_1_3[clave] = [0 for i in range(784)]
    for i in valor:
        pixeles_1_3[clave][i] = 150
    img = np.array(pixeles_1_3[clave]).reshape((28,28))
    plt.imshow(img, cmap='gray', vmin= 0, vmax=255)
    plt.show()
    

pixeles_3_8 = dict()
for clave,valor in pixeles_no_compartidos_3_8.items():
    pixeles_3_8[clave] = [0 for i in range(784)]
    for i in valor:
        pixeles_3_8[clave][i] = 150
    img = np.array(pixeles_3_8[clave]).reshape((28,28))
    plt.imshow(img, cmap='gray', vmin= 0, vmax=255)
    plt.show()
    
    
#%%

"""
1.c
"""

ceros =  dd.sql("""
                SELECT * EXCLUDE('Unnamed: 0', 'labels')
                FROM mnist
                WHERE labels = 0
                """).df()

# Sacamos el promedio de cada pixel de la calse 0
p0 = [i[1] for i in sacar_promedio(0)]

# Lo hacemos array para pasarlo a graficos
ceros = np.array(ceros)

# Seleccionamos píxeles con promedios de intensidad más altos
intensidades = [ceros[:, 627], ceros[:, 213]]

# Boxplot
plt.boxplot(intensidades,showmeans=True, labels=['627', '213'])
plt.title("Distribución de intensidades sobre píxeles relevantes")
plt.ylabel("Intensidad")
plt.yticks(range(0,255,25))
plt.show()

# Histograma
sns.histplot(data=intensidades[0], color="blue", label="627", kde=True, alpha=0.5)
sns.histplot(data=intensidades[1], color="green", label="213", kde=True, alpha=0.5)
plt.ylabel("Cantidad")
plt.xlabel("Intensidad")
plt.title("Distribución de intensidades sobre píxeles relevantes")
plt.legend()
plt.show()
#%%
#2)a)

df_ejerc2 = mnist.loc[(mnist['labels'] == 0) | (mnist['labels'] == 1)]
muestras_0 = len(df_ejerc2.loc[mnist["labels"] == 0])
muestras_1 = len(df_ejerc2.loc[mnist["labels"] == 1])

print(f'La clase 0 tiene {muestras_0} muestras')
print(f'La clase 1 tiene {muestras_1} muestras')

"""
Es apreciable como la cantidad de muestras no está balanceada, aunque su diferencia es mínima
"""
proporcion_0 = round(muestras_0 * 100 / len(df_ejerc2), 2)
proporcion_1 = round(muestras_1 * 100 / len(df_ejerc2), 2)

print(f'\nEsta es la proporción: \nClase 0: {proporcion_0}% \nClase 1: {proporcion_1}%')

plt.bar(['Clase 0'], muestras_0, label='Clase 0', color='#98FF98', edgecolor='black')
plt.bar(['Clase 1'], muestras_1, label='Clase 1', color='#00BFFF', edgecolor='black')
plt.bar(['Total'], len(df_ejerc2), label='Total', color='#FFA07A', edgecolor='black')

plt.title('Proporión de muestras entre clase 0, 1 y total', fontsize=14, fontweight="bold")
plt.ylabel('Cantidad')
plt.grid(True, linestyle=':', color='gray', alpha=0.6)
plt.text('Clase 0', muestras_0 / 2, f'{proporcion_0:.1f}%', ha='center', va='center', color='black', fontweight="bold")
plt.text('Clase 1', muestras_1 / 2, f'{proporcion_1:.1f}%', ha='center', va='center', color='black', fontweight="bold")
plt.text('Total', len(df_ejerc2) / 2, f'{100:.1f}%', ha='center', va='center', color='black', fontweight="bold")

plt.show()

#%%
#2)b)

X = df_ejerc2.drop(columns=['labels'])
y = df_ejerc2['labels']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, stratify=y, random_state = 20)

#%%
#2)c)

"""
Para elegir atributos posibles vamos a computar los promedios de intensidad de cada pixel para la clase 1 y 0.
Con ellos, eliminaremos los pixeles que tengan promedio mayor a 140 y, a partir de los que quedaron, buscaremos 
cuales son los que están en la clase 1 y no en la del 0 y viceversa. De esta forma quedará claro cuales son los
pixeles que en promedio diferencia al 1 del 0 y al revés
"""

promedios_por_pixel_0 = sacar_promedio(0)
promedios_por_pixel_1 = sacar_promedio(1)

elimina_intensidades(promedios_por_pixel_0, 140)
elimina_intensidades(promedios_por_pixel_1, 140)

#Elimino las intensidades con 0 de las dos listas
for tupla in promedios_por_pixel_0[:]:
    if(tupla[1] == 0):
        promedios_por_pixel_0.remove(tupla)
        
for tupla in promedios_por_pixel_1[:]:
    if(tupla[1]==0):
        promedios_por_pixel_1.remove(tupla)
          
pixeles_en_0_no_en_1 = devolver_sin_repetidos(promedios_por_pixel_0, promedios_por_pixel_1)

pixeles_en_1_no_en_0 = devolver_sin_repetidos(promedios_por_pixel_1, promedios_por_pixel_0)

#Imprimimos en su respectiva intensidad los pixeles que diferencian al 1 del 0 y viceversa
plot0 = np.zeros(784)
plot1 = np.zeros(784)
for tupla in pixeles_en_0_no_en_1:
    indice = tupla[0]
    intensidad = tupla[1]
    plot0[indice] = intensidad

for tupla in pixeles_en_1_no_en_0:
    indice = tupla[0]
    intensidad = tupla[1]
    plot1[indice] = intensidad
   
img0 = plot0.reshape((28,28))
img1 = plot1.reshape((28,28))

plt.imshow(img0, cmap='gray', vmin=0, vmax=255)
plt.title("Pixeles con intensidad promedio mayor a 140 en clase 0 y no en 1")
plt.show()

plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
plt.title("Pixeles con intensidad promedio mayor a 140 en clase 1 y no en 0")
plt.show()

#Ahora vamos a entrenar el modelo con los distintos atributos que diferencian las clases
#Elegimos arbitrariamente 15 conjuntos de 3 atributos y también que el hiperparametro
#asociado al número de vecinos sea 3
exactitudes = []
precisiones = []
recalls = []
f1s = []
for i in range(15):
    atributo1 = str(pixeles_en_0_no_en_1[i][0])
    atributo2 = str(pixeles_en_0_no_en_1[i+1][0])
    atributo3 = str(pixeles_en_0_no_en_1[i+2][0])
    
    X_train_3atributos = X_train[[atributo1, atributo2, atributo3]]
    X_test_3atributos = X_test[[atributo1, atributo2, atributo3]]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_3atributos, y_train)
        
    y_pred_3atributos = knn.predict(X_test_3atributos)   
     
    exactitud = accuracy_score(y_test, y_pred_3atributos) 
    precision = precision_score(y_test, y_pred_3atributos)
    recall = recall_score(y_test, y_pred_3atributos)
    f1 = f1_score(y_test, y_pred_3atributos)
    
    exactitudes.append(exactitud)
    precisiones.append(precision)
    recalls.append(recall)
    f1s.append(f1)

#Volvemos a entrenar el modelo pero esta vez con 9 atributos 
for i in range(2):
    atributo1 = str(pixeles_en_0_no_en_1[i+9][0])
    atributo2 = str(pixeles_en_0_no_en_1[i+10][0])
    atributo3 = str(pixeles_en_0_no_en_1[i+11][0])
    atributo4 = str(pixeles_en_0_no_en_1[i+12][0])
    atributo5 = str(pixeles_en_0_no_en_1[i+13][0])
    atributo6 = str(pixeles_en_0_no_en_1[i+14][0])
    atributo7 = str(pixeles_en_0_no_en_1[i+15][0])
    atributo8 = str(pixeles_en_0_no_en_1[i+16][0])
    atributo9 = str(pixeles_en_0_no_en_1[i+17][0])
   
    X_train_9atributos = X_train[[atributo1, atributo2, atributo3, atributo4, atributo5, atributo6, atributo7, atributo8, atributo9]]
    X_test_9atributos = X_test[[atributo1, atributo2, atributo3, atributo4, atributo5, atributo6, atributo7, atributo8, atributo9]]
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_9atributos, y_train)
        
    y_pred_9atributos = knn.predict(X_test_9atributos)   
     
    exactitud = accuracy_score(y_test, y_pred_9atributos) 
    precision = precision_score(y_test, y_pred_9atributos)
    recall = recall_score(y_test, y_pred_9atributos)
    f1 = f1_score(y_test, y_pred_9atributos)
        
    exactitudes.append(exactitud)
    precisiones.append(precision)
    recalls.append(recall)
    f1s.append(f1)

#Ahora vamos a generar gráficos para poder comparar los modelos 
categorias = [f'Modelo {i+1}' for i in range(17)]

plt.bar(categorias, exactitudes, color='royalblue', edgecolor='black')
plt.title('Métrica de exactitud por modelo', fontsize=14, fontweight="bold")
plt.xticks(rotation=90)
plt.ylabel('Exactitud')
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0, 1)
plt.show()

plt.bar(categorias, precisiones, color='royalblue', edgecolor='black')
plt.title('Métrica de precisión por modelo', fontsize=14, fontweight="bold")
plt.xticks(rotation=90)
plt.ylabel('Precisión')
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0, 1)          
plt.show()

plt.bar(categorias, recalls, color='royalblue', edgecolor='black')
plt.title('Métrica de recall por modelo', fontsize=14, fontweight="bold")
plt.xticks(rotation=90)
plt.ylabel('Recall')
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0, 1)
plt.show()

#%%
#2)c)
#Vamos a hacer las comparaciones para 3 cantidades distintas de vecinos y para 3 cantiddades de atributos 

random.seed(42)
exactitudes = []
precisiones = []
recalls = []
for k in range(3):
    for i in range(3):
        if(k==0):
            vecinos = 1
        elif(k==1):
            vecinos = 10
        elif(k==2):
            vecinos = 30
        
        if(i==0):
            cant_atributos = 1 
            #elige 1 atributo al azar en pixeles_en_0_no_en_1
            atributos = random.sample(pixeles_en_0_no_en_1, cant_atributos)
            atributos_str = [str(x[0]) for x in atributos]
            
            X_train_1 = X_train[atributos_str]
            X_test_1 = X_test[atributos_str]
            
            knn = KNeighborsClassifier(n_neighbors= vecinos)
            knn.fit(X_train_1, y_train)
                
            y_pred_1 = knn.predict(X_test_1)   
             
            exactitud = accuracy_score(y_test, y_pred_1) 
            precision = precision_score(y_test, y_pred_1)
            recall = recall_score(y_test, y_pred_1)
            
            exactitudes.append(exactitud)
            precisiones.append(precision)
            recalls.append(recall)
          
        elif(i==1):
            cant_atributos = 10
            atributos = random.sample(pixeles_en_0_no_en_1, cant_atributos)
            atributos_str = [str(x[0]) for x in atributos]
            
            X_train_1 = X_train[atributos_str]
            X_test_1 = X_test[atributos_str]
            
            knn = KNeighborsClassifier(n_neighbors= vecinos)
            knn.fit(X_train_1, y_train)
                
            y_pred_1 = knn.predict(X_test_1)   
             
            exactitud = accuracy_score(y_test, y_pred_1) 
            precision = precision_score(y_test, y_pred_1)
            recall = recall_score(y_test, y_pred_1)
            
            exactitudes.append(exactitud)
            precisiones.append(precision)
            recalls.append(recall)
            
        elif(i==2):
            cant_atributos = 19
            atributos = random.sample(pixeles_en_0_no_en_1, cant_atributos)
            atributos_str = [str(x[0]) for x in atributos]
            
            X_train_1 = X_train[atributos_str]
            X_test_1 = X_test[atributos_str]
            
            knn = KNeighborsClassifier(n_neighbors= vecinos)
            knn.fit(X_train_1, y_train)
                
            y_pred_1 = knn.predict(X_test_1)   
             
            exactitud = accuracy_score(y_test, y_pred_1) 
            precision = precision_score(y_test, y_pred_1)
            recall = recall_score(y_test, y_pred_1)
            
            exactitudes.append(exactitud)
            precisiones.append(precision)
            recalls.append(recall)
      
#Generamos gráficos para poder comparar los modelos 

#Exactitudes:
categorias = ['Modelo 1 \n (k=1)','Modelo 2 \n (k=10)','Modelo 3 \n (k=30)']
plt.bar(categorias, exactitudes[0:3], label='a=1', color='#A7C7E7', edgecolor='black')

categorias = ['Modelo 4 \n (k=1)','Modelo 5 \n (k=10)','Modelo 6 \n (k=30)']
plt.bar(categorias, exactitudes[3:6], label='a=10', color='#F5F5DC', edgecolor='black')

categorias = ['Modelo 7 \n (k=1)','Modelo 8 \n (k=10)','Modelo 9 \n (k=30)']
plt.bar(categorias, exactitudes[6:9], label='a=19', color='#696969', edgecolor='black')

plt.title('Métrica de exactitud por modelo separados por \n cantidad de vecinos (k) y cantidad de atributos (a) \n de entrenamiento', fontsize=14, fontweight="bold")
plt.xticks(rotation=45, fontsize=7.9)
plt.ylabel('Exactitud')
plt.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)
plt.legend(fontsize=7.8)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()

#Precisiones:
categorias = ['Modelo 1 \n (k=1)','Modelo 2 \n (k=10)','Modelo 3 \n (k=30)']
plt.bar(categorias, precisiones[0:3], label='a=1', color='#A7C7E7', edgecolor='black')

categorias = ['Modelo 4 \n (k=1)','Modelo 5 \n (k=10)','Modelo 6 \n (k=30)']
plt.bar(categorias, precisiones[3:6], label='a=10', color='#F5F5DC', edgecolor='black')

categorias = ['Modelo 7 \n (k=1)','Modelo 8 \n (k=10)','Modelo 9 \n (k=30)']
plt.bar(categorias, precisiones[6:9], label='a=19', color='#696969', edgecolor='black')

plt.title('Métrica de precisión por modelo separados por \n cantidad de vecinos (k) y cantidad de atributos (a) \n de entrenamiento', fontsize=14, fontweight="bold")
plt.xticks(rotation=45, fontsize=7.9)
plt.ylabel('Precisión')
plt.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)
plt.legend(fontsize=7.8)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()

#Recalls:
categorias = ['Modelo 1 \n (k=1)','Modelo 2 \n (k=10)','Modelo 3 \n (k=30)']
plt.bar(categorias, recalls[0:3], label='a=1', color='#A7C7E7', edgecolor='black')

categorias = ['Modelo 4 \n (k=1)','Modelo 5 \n (k=10)','Modelo 6 \n (k=30)']
plt.bar(categorias, recalls[3:6], label='a=10', color='#F5F5DC', edgecolor='black')

categorias = ['Modelo 7 \n (k=1)','Modelo 8 \n (k=10)','Modelo 9 \n (k=30)']
plt.bar(categorias, recalls[6:9], label='a=19', color='#696969', edgecolor='black')

plt.title('Métrica de exhaustividad por modelo separados por \n cantidad de vecinos (k) y cantidad de atributos (a) \n de entrenamiento', fontsize=14, fontweight="bold")
plt.xticks(rotation=45, fontsize=7.9)
plt.ylabel('Exhaustividad')
plt.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)
plt.legend(fontsize=7.8)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()

#%% Ejercicio 3
'''
Debido a lo mencionado en el punto 1 el 59% de los pixeles totales los descartaremos
a la hora de hacer la prediccion ya que por un lado elimnarlos aumenta muy considerablemente
la eficiencia de nuestro arbol al darle menos de la mitad de los datos originales a considerar
y tambien porque muchos de estos pixeles prodian no permitir crear patrones en ciertas 
profundidades del arbol que podria ser capaz de hacerlo o inclusivemente afectar la prediccion
al punto de hacerlo equivocarse creando un patron incorrecto habiendo dejado influenciarse
por los pixeles que no tienen peso a la hora de distinguir distintos diguitos
'''
#%%

pixeles_3 = list()
for i in range(len(relevante)):
    if relevante[i] == 0:
        pixeles_3.append(i + 1)
        
mnist_relevante = mnist.drop(mnist.columns[pixeles_3], axis = 1)
#%% 3_a
X = mnist_relevante.drop(columns=['labels', 'Unnamed: 0'])
y = mnist_relevante['labels']

X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,test_size = 0.20, stratify=y, random_state = 20)

#%% 3_b

profundidad = [1,2,3,4,5,6,7,8,9,10]
nsplits = 10

# Se usa StratifiedKFold en vez de KFold ya que como mencionamos previamente hay un
# desbalance de clases y KFold no lo considera al contrario de StratifiedKFold 
# aumentando levemente el porcentaje de prediccion
kf = StratifiedKFold(n_splits = nsplits)

#creo la lista que va a tener cada promedio de cada arbol en cada altura
exactitud = np.zeros((nsplits, len(profundidad)))
precision = np.zeros((nsplits, len(profundidad)))
recall = np.zeros((nsplits, len(profundidad)))

# una fila por cada fold, una columna por cada modelo con su hiperparametro

for i, (train_index, test_index) in enumerate(kf.split(X_dev, y_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    # Ahora con este for voy a poder ver cual es el mejor hiperparametro de mi  modelo
    
    for j, hmax in enumerate(profundidad):

        arbol = tree.DecisionTreeClassifier(max_depth = hmax)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        score_exactitud = accuracy_score(kf_y_test, pred)
        score_precision = precision_score(kf_y_test, pred, average='macro')
        score_recall = recall_score(kf_y_test, pred, average='macro')

        exactitud[i, j] = score_exactitud
        precision[i, j] = score_precision
        recall[i, j] = score_recall

#%% 3_c
# promedio scores sobre los folds para cada metrica

scores_promedio_exactitud = exactitud.mean(axis = 0)
scores_promedio_precision = precision.mean(axis = 0)
scores_promedio_recall = recall.mean(axis = 0)

for i,e in enumerate(profundidad):
    print(f'Exactitud promedio del modelo con la altura {e}: {scores_promedio_exactitud[i]:.4f}')
    print(f'Precision promedio del modelo con la altura {e}: {scores_promedio_precision[i]:.4f}')
    print(f'Recall promedio del modelo con la altura {e}: {scores_promedio_recall[i]:.4f}\n')

#%% grafico de cada profundida del arbol junto a sus metricas

metricas = ['Exactitud', 'Precision', 'Recall']

fig, ax = plt.subplots()

plt.rcParams['font.family'] = 'sans-serif'


#Grafica de Exactitud
ax.plot(profundidad, scores_promedio_exactitud, marker = '.',
        linestyle = '-',
        linewidth = 0.5,
        label = 'Exactitud')

#Grafica de Precision
ax.plot(profundidad, scores_promedio_precision, marker = '.',
        linestyle = '-',
        linewidth = 0.5,
        label = 'Precision')

#Grafica de Recall
ax.plot(profundidad, scores_promedio_recall, marker = '.',
        linestyle = '-',
        linewidth = 0.5,
        label = 'Recall')

ax.set_title('Valor de cada Metrica por profundidad del Arbol de Testeo')
ax.set_xlabel('Profundidad', fontsize = 'medium')
ax.set_ylabel('Valor Metrica', fontsize = 'medium')
ax.legend(title = metricas)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(0, 11, 1))

plt.savefig('Valor de cada Metrica por profundidad del Arbol de Testeo',dpi=300, bbox_inches="tight")

plt.show()
    
#%% 

#entreno el modelo elegido en el conjunto dev entero
def get_max(l):
    max = 1
    for i in range(len(l)):
        if l[0] < l[i]:
            max = i + 1
    return max
    
    
arbol_elegido = tree.DecisionTreeClassifier(max_depth = get_max(scores_promedio_exactitud))
arbol_elegido.fit(X_dev, y_dev)

# con el arbol entrenado con x_dev e y_dev (con una altura adecuada) observamos como predice
y_pred = arbol_elegido.predict(X_dev)

# encontramos el valor de exactitud,precision y recall de que tan bien predice el arbol con x_dev e y_dev
score_exactitud_arbol_elegido_dev = accuracy_score(y_dev, y_pred)
score_precision_arbol_elegido_dev = precision_score(y_dev, y_pred, average='macro')
score_recall_arbol_elegido_dev = recall_score(y_dev, y_pred, average='macro')

print(f'Exactidud del arbol con profundidad {get_max(scores_promedio_exactitud)} en cuanto al dev: {score_exactitud_arbol_elegido_dev:.4f}')
print(f'Precision del arbol con profundidad {get_max(scores_promedio_exactitud)} en cuanto al dev: {score_precision_arbol_elegido_dev:.4f}')
print(f'Recall del arbol con profundidad {get_max(scores_promedio_exactitud)} en cuanto al dev: {score_recall_arbol_elegido_dev:.4f}')
#%% 3_d

# probamos el modelo con el hiperparametro elegido y entrenado en el conjunto dev
# procedemos a ver que tan bien predice comparandolo con el held_out

y_pred_eval = arbol_elegido.predict(X_eval)

score_exactitud_arbol_elegido_eval = accuracy_score(y_eval, y_pred_eval)
score_precision_arbol_elegido_eval = precision_score(y_eval, y_pred_eval, average='macro')
score_recall_arbol_elegido_eval = recall_score(y_eval, y_pred_eval, average='macro')

print(f'Exactitud del arbol en comparacion al held_out: {score_exactitud_arbol_elegido_eval:.4f}')
print(f'Precision del arbol en comparacion al held_out: {score_precision_arbol_elegido_eval:.4f}')
print(f'Recall del arbol en comparacion al held_out: {score_recall_arbol_elegido_eval:.4f}')

#%% Matriz de confusion

# Calcular la matriz de confusión
cm = confusion_matrix(y_eval, y_pred_eval)

y_labels = np.unique(y)

# Visualizacion
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels = y_labels, yticklabels = y_labels)
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')

plt.savefig('Matriz de Confusión',dpi=300, bbox_inches="tight")

plt.show()

#%% 
#Creamos la matriz de confusion como data frame


l0 = list()
l1 = list()
l2 = list()
l3 = list()
l4 = list()
l5 = list()
l6 = list()
l7 = list()
l8 = list()
l9 = list()


for i in range(len(cm[0])):
    l0.append(cm[i,0])
    l1.append(cm[i,1]) 
    l2.append(cm[i,2]) 
    l3.append(cm[i,3]) 
    l4.append(cm[i,4]) 
    l5.append(cm[i,5]) 
    l6.append(cm[i,6]) 
    l7.append(cm[i,7]) 
    l8.append(cm[i,8]) 
    l9.append(cm[i,9]) 
    
cm_data = {
    "0": l0,
    "1": l1,
    "2": l2,
    "3": l3,
    "4": l4,
    "5": l5,
    "6": l6,
    "7": l7,
    "8": l8,
    "9": l9
}    
    

cm_df = pd.DataFrame(cm_data)

#Ahora creamos la misma pero traspuesta
l00 = list()
l11 = list()
l22 = list()
l33 = list()
l44 = list()
l55 = list()
l66 = list()
l77 = list()
l88 = list()
l99 = list()


for i in range(len(cm[0])):
    l00.append(cm[0,i])
    l11.append(cm[1,i]) 
    l22.append(cm[2,i]) 
    l33.append(cm[3,i]) 
    l44.append(cm[4,i]) 
    l55.append(cm[5,i]) 
    l66.append(cm[6,i]) 
    l77.append(cm[7,i]) 
    l88.append(cm[8,i]) 
    l99.append(cm[9,i]) 
    
cm_data_t = {
    "d0": l00,
    "d1": l11,
    "d2": l22,
    "d3": l33,
    "d4": l44,
    "d5": l55,
    "d6": l66,
    "d7": l77,
    "d8": l88,
    "d9": l99
}    

cm_df_traspuesta = pd.DataFrame(cm_data_t)

# Buscamos las muestras de cada clase

cm_muestras = dd.sql("""
                     SELECT SUM(CASE WHEN d0 IS NOT NULL THEN d0 END ) AS '0',
                            SUM(CASE WHEN d1 IS NOT NULL THEN d1 END ) AS '1',
                            SUM(CASE WHEN d2 IS NOT NULL THEN d2 END ) AS '2',
                            SUM(CASE WHEN d3 IS NOT NULL THEN d3 END ) AS '3',
                            SUM(CASE WHEN d4 IS NOT NULL THEN d4 END ) AS '4',
                            SUM(CASE WHEN d5 IS NOT NULL THEN d5 END ) AS '5',
                            SUM(CASE WHEN d6 IS NOT NULL THEN d6 END ) AS '6',
                            SUM(CASE WHEN d7 IS NOT NULL THEN d7 END ) AS '7',
                            SUM(CASE WHEN d8 IS NOT NULL THEN d8 END ) AS '8',
                            SUM(CASE WHEN d9 IS NOT NULL THEN d9 END ) AS '9',
                     FROM cm_df_traspuesta
                    """).df()

series_m_ev = cm_muestras.iloc[0]

plt.figure(figsize=(8, 5))
plt.bar(series_m_ev.index, series_m_ev.values)

plt.xlabel("Digito")
plt.ylabel("Muestras")
plt.title("Cantidad de muestras por digito evaluado")

plt.savefig('Cantidad de muestras por digito evaluado',dpi=300, bbox_inches="tight")

plt.show()

#%%
# Buscamos el exito de cada clase

tasa_exito = list()
for j in range(10):
    tasa_exito.append(cm_df_traspuesta.iloc[j, j] / cm_muestras.iloc[0,j])

tasa_exito_redondeada = [round(val, 2) for val in tasa_exito]

tasa_exito_dc = {'0': [tasa_exito_redondeada[0]], 
                 '1': [tasa_exito_redondeada[1]],
                 '2': [tasa_exito_redondeada[2]],
                 '3': [tasa_exito_redondeada[3]],
                 '4': [tasa_exito_redondeada[4]],
                 '5': [tasa_exito_redondeada[5]],
                 '6': [tasa_exito_redondeada[6]],
                 '7': [tasa_exito_redondeada[7]],
                 '8': [tasa_exito_redondeada[8]],
                 '9': [tasa_exito_redondeada[9]]}

tasa_exito_df = pd.DataFrame(tasa_exito_dc)

series_t_ex = tasa_exito_df.iloc[0]

plt.figure(figsize=(8, 5))
plt.bar(series_t_ex.index, series_t_ex.values)

for y in np.arange(0.5, 0.9, 0.1):  
    plt.axhline(y, linestyle="--", color="red", alpha=0.7)

plt.xlabel("Digito")
plt.ylabel("Tasa de exito")
plt.title("Tasa de exito por digito")
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.savefig('Tasa de exito por digito',dpi=300, bbox_inches="tight")

plt.show()
