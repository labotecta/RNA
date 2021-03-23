# Aplicación que implementa un OCR básico para el reconocimiento de dígitos
# individuales.
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as pyplot
import sys
import os

# Tamaño de la consola de windows
os.system('mode 100,40')

def GraficoImagen(vg, imagen, senda, nombre):
    pyplot.figure()
    pyplot.imshow(imagen)
    pyplot.colorbar()
    pyplot.grid(False)
    pyplot.savefig(senda + nombre + '.png', dpi = 96, facecolor = 'w', edgecolor = 'w', orientation = 'portrait', format = 'png', transparent = False, bbox_inches = None, pad_inches = 0.1, metadata = None)
    if vg == 1: pyplot.show()
    pyplot.close()

def GraficoImagenes(vg, imagenes, lado, senda, nombre):
    ancho = 8
    alto = int(imagenes.shape[0] / ancho)
    fig, axs = pyplot.subplots(alto, ancho)
    n = 0
    for k in range(alto):
        for i in range(ancho):
            pyplot.setp(axs[k,i].get_xticklabels(), visible=False)
            pyplot.setp(axs[k,i].get_yticklabels(), visible=False)
            axs[k,i].grid(False)
            axs[k,i].imshow(imagenes[n].reshape((lado, lado)))
            n += 1
    pyplot.savefig(senda + nombre + '.png', dpi = 96, facecolor = 'w', edgecolor = 'w', orientation = 'portrait', format = 'png', transparent = False, bbox_inches = None, pad_inches = 0.1, metadata = None)
    if vg == 1: pyplot.show()
    pyplot.close()

def GraficoDistribucion(vg, minimo, maximo, titulo, valores, corte, senda, nombre):
    pyplot.rcParams["figure.figsize"] = (12,7)
    pyplot.xlim(minimo, maximo)
    pyplot.hist(valores, bins = 25)
    pyplot.axvline(x=corte, color='C1', linewidth=2)
    pyplot.title(titulo, fontsize=18)
    pyplot.xlabel("Distribución de valores", fontsize=16)
    pyplot.ylabel("Frecuencia", fontsize=16)
    pyplot.savefig(senda + nombre + '.png', dpi = 96, facecolor = 'w', edgecolor = 'w', orientation = 'portrait', format = 'png', transparent = False, bbox_inches = None, pad_inches = 0.1, metadata = None)
    if vg == 1: pyplot.show()
    pyplot.close()

def PintaUnCero(metrica, probabilidad, corte, lado, puntos_iniciales, intentos):
    pixeles = lado * lado
    img_mejor = np.zeros((lado, lado), dtype = np.float32)
    img_prueba = np.zeros((lado, lado), dtype = np.float32)
    for i in range(puntos_iniciales):
        x = random.randrange(0, lado)
        y = random.randrange(0, lado)
        img_prueba[x, y] = 1.0
    if metrica == 0:
        # minimizar
        mejor_curso = 99999.9
    else:
        # maximizar
        mejor_curso = -99999.9
    
    # Recorrido sistemático añadiendo puntos que mejoran
    puntos_fin, img_mejor, val_mejor, mas = AjustaMas(metrica, lado, img_prueba, mejor_curso)        
    
    nazar = 0
    if ((metrica == 0) and (val_mejor > corte)) or ((metrica == 1) and (val_mejor < corte)):    
        
        # Es mejorable, modificaciones al azar
        mejor_curso = val_mejor
        img_prueba = img_mejor.copy()
        nazar = intentos
        for i in range(intentos):
        
            # Cambia 1 pixel en la imagen de prueba, alternando su valor
            x = random.randrange(0, lado)
            y = random.randrange(0, lado)
            img_prueba[x, y] = 1.0 - img_prueba[x, y]

            if metrica == 0:
                distancia = img_prueba - probabilidad
                distancia = distancia * distancia
            else:
                distancia = img_prueba * probabilidad
            prediccion = distancia.sum()
            if metrica == 0:
                # minimizar
                mejorado = prediccion < mejor_curso
            else:
                # maximizar
                mejorado = prediccion > mejor_curso
        
            # Se aceptan retrocesos en el 5% de los casos
            if mejorado == True or (random.randrange(0, 100) < 5):
            
                # Consolida el cambio
                mejor_curso = prediccion
            
                if ((metrica == 0) and (mejor_curso < val_mejor)) or ((metrica == 1) and (mejor_curso > val_mejor)):
                    val_mejor = prediccion
                    img_mejor = img_prueba.copy()
            
                if ((metrica == 0) and (mejor_curso < corte)) or ((metrica == 1) and (mejor_curso > corte)):
                    nazar = i + 1
                    break
            else:
                
                # Deshace el cambio
                img_prueba[x, y] = 1.0 - img_prueba[x, y]
        
    # Recorrido sistemático quitando puntos que no mejoran
    puntos_fin, img_mejor, val_mejor, menos = AjustaMenos(metrica, lado, img_mejor, val_mejor)        

    imprime('Proceso de ajuste: [Dimensión {:,}] {:,} puntos más y {:,} puntos menos. Puntos totales {:,}'.format(pixeles, mas, menos, puntos_fin))
    if ((metrica == 0) and (val_mejor < corte)) or ((metrica == 1) and (val_mejor > corte)):
        imprime('Objetivo alcanzado           [{:10.3f}] tras {:,} intentos MC'.format(corte, nazar))
    else:
        imprime('NO se alcanzó el objetivo de [{:10.3f}] tras {:,} intentos MC'.format(corte, nazar))
    imprime('distancia                     {:10.3f}'.format(val_mejor))
    GraficoImagen(1, img_mejor, sendaG, 'mejor_' + str(metrica))

def AjustaMas(metrica, lado, img_mejor, mejorada):
    
    # Agregar puntos que mejoran el resultado
    print('Ajustando (+) ...')
    mas = 0
    puntos_fin = 0
    for y in range(lado):
        for x in range(lado):
            if img_mejor[x, y] == 0.0:
                img_mejor[x, y] = 1.0
                if metrica == 0:
                    distancia = img_mejor - probabilidad
                    distancia = distancia * distancia
                else:
                    distancia = img_mejor * probabilidad
                prediccion = distancia.sum()
                if metrica == 0:
                    # minimizar
                    mejorado = prediccion < mejorada
                else:
                    # maximizar
                    mejorado = prediccion > mejorada
                if mejorado == True:
                    mejorada = prediccion
                    mas += 1
                    puntos_fin += 1
                else:
                    img_mejor[x, y] = 0.0
            else:
                puntos_fin += 1
    return puntos_fin, img_mejor, mejorada, mas

def AjustaMenos(metrica, lado, img_mejor, mejorada):
    
    # Eliminar puntos superfluos, que no empeoran el resultado
    print('Ajustando (-) ...')
    menos = 0
    puntos_fin = 0
    for y in range(lado):
        for x in range(lado):
            if img_mejor[x, y] == 1.0:
                img_mejor[x, y] = 0.0
                if metrica == 0:
                    distancia = img_mejor - probabilidad
                    distancia = distancia * distancia
                else:
                    distancia = img_mejor * probabilidad
                prediccion = distancia.sum()
                if metrica == 0:
                    # minimizar
                    mejorado = prediccion < mejorada
                else:
                    # maximizar
                    mejorado = prediccion > mejorada
                if mejorado == True:
                    mejorada = prediccion
                    menos += 1
                else:
                    img_mejor[x, y] = 1.0
                    puntos_fin += 1
    return puntos_fin, img_mejor, mejorada, menos

def imprime(texto):
    print(texto)
    log.write(texto + '\n')

# Mostrar gráficos en pantalla.  0 = No, 1 = Si
vg = 0

# Donde están los datos
sendaD = 'F:/Articulos/IA/Datos/'
fientrena = 'mnist_train'
fiprueba = 'mnist_test'

# Donde se guardarán los resultados, asegurando que las caprpetas existen
senda = 'F:/Articulos/IA/OCR/Patron/'
if not os.path.exists(senda): os.makedirs(senda)
sendaG = senda + 'Imagenes/'
if not os.path.exists(sendaG): os.makedirs(sendaG)

# Vamos a guardar los resultados en el fichero 'resumen.log'
# escribiremos en él lo mismo que mostremos en pantalla
log = open(senda + 'resumen.log', 'w')

# Leer el fichero con la muestra de entrenamiento y poner los datos en el
# 'dataframe' 'entrena', cada fila tiene (1 + lado * lado) valores: la etiqueta
# que
# identifica el dígito y la imagen de lado * lado pixeles
entrena = pd.read_csv(sendaD + fientrena + '.csv', sep=',', header=None)

# Reordenamos ('sample') las imágenes al azar.  Inicializamos el generador con
# una semilla concreta para que sea reproducible
random.seed(a=2947, version=2)
entrena = entrena.sample(frac=1).reset_index(drop=True)

# Asignar un nombre (del 0 al lado * lado-1) a las cabeceras de las columnas,
# necesario para indicarle la columna a la sentencia 'pop'
entrena.columns = list(range(0, entrena.shape[1]))

# Extraer la columna de las etiquetas antes de quitar esta coluna en el
# 'dataframe' 'entrena'
eti_entrena = entrena.iloc[:,0] 

# Extraer las imagenes que corresponden al dígito '0'
ceros = entrena[entrena.iloc[:,0] == 0] 

# Quitar la columna de la etiqueta, así cada fila queda con los lado * lado
# pixeles de la imagen
entrena.pop(0)
ceros.pop(0)

pixeles = entrena.shape[1]
lado = int(math.sqrt(pixeles))

# La transformamos en números reales (coma flotante) ya que antes son enteros
entrena = np.array(entrena, dtype = np.float32)
ceros = np.array(ceros, dtype = np.float32)

# Mostrar 32 imagenes
numimg = 32
imagenesej = np.zeros((numimg, pixeles), dtype = np.float32)
for i in range(numimg):
    for j in range(pixeles):
        imagenesej[i,j] = entrena[i,j]
GraficoImagenes(vg, imagenesej, lado, sendaG, 'ejemplos')

# Dibujamos las tres primeras imagenes de la muestra de ceros
GraficoImagen(vg, ceros[0].reshape((lado, lado)), sendaG, 'cero1')
GraficoImagen(vg, ceros[1].reshape((lado, lado)), sendaG, 'cero2')
GraficoImagen(vg, ceros[2].reshape((lado, lado)), sendaG, 'cero3')

imprime('Muestra de entrenamiento. dígitos 0 : {:7,} dígitos del 1 al 9: {:7,} total {:7,}'.format(len(ceros), len(entrena) - len(ceros), len(entrena)))

# Obtenemos, para cada punto, la media de ese punto en todas las imagenes,
# obteniendo lo que podemos interpretar como la probabilidad de que una imagen
# del dígito 0 tenga ocupado ese punto
probabilidad = ceros.mean(axis = 0)

# Normalizar a 1.0 ya que los valores van de 0 a 255
probabilidad = probabilidad / 255

# Pasar a dos dimensiones, para dibujarla
probabilidad = probabilidad.reshape((lado, lado))
GraficoImagen(vg, probabilidad,sendaG, 'semejanza')

# Penalizamos los pixeles con baja probabilidad.  Hasta aquí habiamos puesto el
# enfasis en lo que hace iguales entre si a las imagenes del 0, ahora añadimos
# lo que las hace diferentes del resto
probabilidad = np.where(probabilidad < 0.1,-1,probabilidad)
GraficoImagen(vg, probabilidad, sendaG, 'semejanzas_discrepancias')

# Ya tenemos la referencia (patrón) para comparar, ahora tenemos que establecer
# un criterio (métrica) para calcular cuanto se diferencia una imagen del
# patrón, lo que llamaremos la distancia entre imagen y patrón.

# Queremos obtener los valores mínimos y máximos que resultan para
# todas las imagenes de 0 que hay en la muestra de entrenamiento
# Vamos a utilizar dos métricas distintas:

# 1.  El tradicional error cuadrático medio:
# Dada una imagen, vamos a definir el error en un punto como la diferencia
# entre la probabilidad de que esté ocupado y la ocupación real en la imagen,
# elevada al cuadrado para evitar compensaciones entre positivos y negativos.
# Finalmente, obtenemos la distancia de la imagen sumando las diferencias de
# los 28x28 pixeles.  Cuanto menor sea la distancia más semejante es la imagen
# al patron (probabilidad)

# 2.  La poneración de la imagen con el patrón (covarianza).
# Dada una imagen, vamos a definir la distancia de la imagen como la suma de
# los productos, punto a punto, de la imagen por su probabilidad.  Cuanto mayor
# sea la distancia más semejante será la imagen al '0'

# Leer la muestra de prueba
prueba = pd.read_csv(sendaD + fiprueba + '.csv', sep=',', header=None)
random.seed(a=1001, version=2)
prueba = prueba.sample(frac=1).reset_index(drop=True)
prueba.columns = list(range(0, pixeles + 1))

# Extraer la columna de las etiquetas antes de quitar esta coluna en el
# 'dataframe' 'prueba'
eti_prueba = prueba.iloc[:,0] 
prueba.pop(0)
prueba = np.array(prueba, dtype = np.float32)
nreferentes = np.count_nonzero(eti_prueba == 0)
imprime('Muestra de prueba.        dígitos 0 : {:7,} dígitos del 1 al 9: {:7,} total {:7,}\n'.format(nreferentes, len(prueba) - nreferentes,len(prueba)))

nbmetrica = ['ECM', 'CC']
cortes = [0.0, 0.0]
for metrica in (0, 1):
    imprime('-----------')
    imprime('Métrica {}'.format(nbmetrica[metrica]))
    imprime('-----------\n')

    # Procesamos las imagenes de '0'
    historico_0 = []
    minimo_0 = pixeles
    maximo_0 = 0
    for imagen in ceros:
    
        # Pasamos la imagen a dos dimensiones porque la probabilidad la tenemos
        # así
        imagen = imagen.reshape((lado, lado))
        imagen = imagen / 255
        if metrica == 0:
            # Suma del cuadrado de las diferencias
            distancia = imagen - probabilidad
            distancia = distancia * distancia
        else:
            # Suma de los productos punto a punto
            distancia = imagen * probabilidad
        suma = distancia.sum()
        historico_0.append(suma)
        if suma < minimo_0: minimo_0 = suma
        elif suma > maximo_0: maximo_0 = suma
    imprime('Distancia mínima {:11.1f} {:11.3f} por punto'.format(minimo_0, minimo_0 / pixeles))
    imprime('Distancia máxima {:11.1f} {:11.3f} por punto\n'.format(maximo_0, maximo_0 / pixeles))

    if metrica == 0:
        # Vamos a exigir una distancia mayor que el mínimo obtenido, aunque
        # esto de lugar a que se escapen algunos ceros (falsos negativos),
        # perdemos sensibilidad para ganar acierto.  El corte se ha fijado por
        # tanteo
        corte = minimo_0 + (maximo_0 - minimo_0) / 4
    else:
        # Vamos a exigir una distancia mayor que el mínimo obtenido, aunque
        # esto de lugar a que se escapen algunos ceros (falsos negativos).  El
        # corte se ha fijado por tanteo
        corte = minimo_0 + (maximo_0 - minimo_0) / 2.5
    imprime('Punto de corte {:11.3f} {:11.3f}\n'.format(corte, corte / pixeles))
    cortes[metrica] = corte

    # Para dibujar la distribución de las distancias en las imágenes distintas
    # de 0
    minimo_t = pixeles
    maximo_t = 0
    historico_t = []
    for imagen, etiqueta in zip(entrena, eti_entrena):
        if etiqueta != 0:
            imagen = imagen.reshape((lado, lado))
            imagen = imagen / 255
            if metrica == 0:
                distancia = imagen - probabilidad
                distancia = distancia * distancia
            else:
                distancia = imagen * probabilidad
            suma = distancia.sum()
            historico_t.append(suma)
            if suma < minimo_t: minimo_t = suma
            elif suma > maximo_t: maximo_t = suma

    # Para igualar el eje x en ambos gráficos
    minimo = min(minimo_0, minimo_t)
    maximo = max(maximo_0, maximo_t)
    
    # Dibujamos la frecuencia de las distancias en las imágenes 0
    GraficoDistribucion(vg, minimo, maximo, 'Imágenes 0; Métrica: ' + nbmetrica[metrica] + '; valor: mínimo {:.0f};  máximo {:.0f}; corte {:.1f}'.format(minimo_0,maximo_0,corte), historico_0, corte, sendaG, 'distribucion_0_' + nbmetrica[metrica])
    
    # Dibujamos la frecuencia de las distancias en las imagenes distintas a 0
    GraficoDistribucion(vg, minimo, maximo, 'Imágenes distintas a 0; Métrica: ' + nbmetrica[metrica] + '; valor: mínimo {:.0f};  máximo {:.0f}; corte {:.1f}'.format(minimo_0,maximo_0,corte), historico_t, corte, sendaG, 'distribucion_X_' + nbmetrica[metrica])

    # Ahora vamos a ver que tal funciona la probabilidad obtenida para
    # detectar los 0 en la muestra de prueba, que no ha intervenido en la
    # construcción de dicha probabilidad

    # Recorrer todas las imagenes de la muestra de prueba
    nreferentes = aciertopositivo = aciertonegativo = falsopositivo = falsonegativo = 0
    for imagen, etiqueta in zip(prueba, eti_prueba):
        if etiqueta == 0: nreferentes +=1
        imagen = imagen.reshape((lado, lado))
        imagen = imagen / 255
        if metrica == 0:
            distancia = imagen - probabilidad
            distancia = distancia * distancia
        else:
            distancia = imagen * probabilidad
        suma = distancia.sum()
        if (metrica == 0 and suma < corte) or (metrica == 1 and suma > corte):
            # Como la distancia (error) es suficientemente pequeña (menor
            # que el corte) concluimos que es un 0
            if etiqueta == 0: aciertopositivo +=1
            else:             falsopositivo +=1
        else:
            # La distancia excede el máximo (corte) para que lo podamos
            # consideremos un 0
            if etiqueta == 0: falsonegativo +=1
            else:             aciertonegativo +=1
    
    deno1 = (len(prueba) - nreferentes)
    if deno1 == 0:
        pfp = 0
        pan = 0
    else:
        pfp = 100.0 * falsopositivo / deno1
        pan = 100.0 * aciertonegativo / deno1
    deno2 = nreferentes
    if deno2 == 0:
        pfn = 0
        pap = 0
    else:
        pfn = 100.0 * falsonegativo / deno2
        pap = 100.0 * aciertopositivo / deno2
    deno3 = len(prueba)
    if deno3 == 0:
        pat = 0
    else:
        pat = 100.0 * (aciertopositivo + aciertonegativo) / deno3
    imprime('Resultados para la muestra de prueba\n')
    imprime('Falsos positivos   (dígitos del 1 al 9 confundidos con 0)          : {:7,} {:4.1f}%'.format(falsopositivo, pfp))
    imprime('Falsos negativos   (dígitos 0 no reconocidos)                      : {:7,} {:4.1f}%' .format(falsonegativo, pfn))
    imprime('Aciertos positivos (dígitos 0 reconocidos como tal -sensibilidad-) : {:7,} {:4.1f}%'.format(aciertopositivo, pap))
    imprime('Aciertos negativos (dígitos 1 al 9 reconocidos como no 0)          : {:7,} {:4.1f}%'.format(aciertonegativo, pan))
    imprime('Aciertos totales                                                   : {:7,} {:4.1f}%'.format(aciertopositivo + aciertonegativo, pat))
    imprime('\n')
    log.flush()

random.seed(a=1355, version=2)
imprime('Dibujar un cero\n')
imprime('Métrica {}'.format(nbmetrica[0]))
PintaUnCero(0, probabilidad, cortes[0] / 2, lado, 1, 50000)
random.seed(a=7347, version=2)
imprime('Métrica {}'.format(nbmetrica[1]))
PintaUnCero(1, probabilidad, cortes[1] * 3, lado, 1, 50000)
log.close()
