import argparse
import math
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as pyplot
import os
os.system('mode 120,50')

# Opciones que se pueden incluir en la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--dird', default='F:/Articulos/IA/Datos/', type=str, help='fichero con la muestra de entrenamiento sin extensión qye debe ser .csv')
parser.add_argument('--dirr', default='F:/Articulos/IA/OCR/Formas_64x64_6000_DN_0_0_r0/', type=str, help='fichero con la muestra de entrenamiento sin extensión ya que debe ser .csv')
parser.add_argument('--objetos', default=1, type=int, help='0 = dígitos, 1 = formas')
parser.add_argument('--fientrena', default='formas_train_6000_64x64', type=str, help='fichero con la muestra de entrenamiento, sin extensión ya que debe ser .csv')
parser.add_argument('--fiprueba', default='formas_test_30000_64x64', type=str, help='fichero con la muestra de prueba, sin extensión qye debe ser .csv')
parser.add_argument('--fisorpresa', default='formas_sorpresa_30000_64x64', type=str, help='fichero con la muestra de prueba alternativa, sin extensión qye debe ser .csv')
parser.add_argument('--referente', default=0, type=int, help='0 = dígito 0/circulo, 1 = dígito 1/cuadrado, 2 = dígito/triangulo, n = dígito n')
parser.add_argument('--tipoRNA', default=0, type=int, help='0 = Densa, 1 = Convolución')

parser.add_argument('--capas_ocultas', default=0, type=int, help='0 = sin capa intermedia, 1 = con una capa intermedia de 2 neuronas')
parser.add_argument('--neuronas', default=0, type=int, help='neuronas en la capa intermedia')

parser.add_argument('--fil', default=32, type=int, help='Número de filtros')
parser.add_argument('--mcv', default=5, type=int, help='Marco de convolucion')
parser.add_argument('--ccv', default=2, type=int, help='Concentrador de convolucion')
parser.add_argument('--scv', default=2, type=int, help='Salida de convolucion')
parser.add_argument('--vg', default=0, type=int, help='0 = No mostrar gráficos en pantalla; 1 = Mostrar gráficos en pantalla')

parser.add_argument('--barridoini', default=1, type=int, help='0 = No realizar el barrido inicial (ajuste +); 1 = realizarlo')
parser.add_argument('--montecarlo', default=3, type=int, help='Número de imagenes a crear del referente')
parser.add_argument('--sorteos', default=[5000, 5000, 5000], nargs='*', type=int, help='Número de sorteos para generar las imágenes del referente')
parser.add_argument('--pini', default=[0, 0, 0], nargs='*', type=int, help='Número de puntos iniciales al azar para generar las imágenes del referente')
parser.add_argument('--cortemc', default=[0.99999, 0.99999, 0.99999], nargs='*', type=float, help='Valor de corte para generar las imágenes del referente')
parser.add_argument('--margenmc', default=[4, 4, 4], nargs='*', type=int, help='Margen para aceptar un retroceso en la generación de las imágenes del referente')

parser.add_argument('--pases', default=10, type=int, help='Número de veces que se pasan todas las observaciones de la muestra entrenar')

# Extrae los argumento de la línea de comandos
args = parser.parse_args()

# Funciones que se llamarán en algún momento
# ---------------------------------------------------------------
def GraficoImagen(vg, imagen, senda, nombre, suf):
    pyplot.figure()
    pyplot.imshow(imagen)
    pyplot.colorbar()
    pyplot.grid(False)
    if senda != '' and nombre != '':
        pyplot.savefig(senda + nombre + suf + '.png', dpi = 96, facecolor = 'w', edgecolor = 'w', orientation = 'portrait', format = 'png', transparent = False, bbox_inches = None, pad_inches = 0.1, metadata = None)
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

def GraficoEvolucion(vg, historico, senda, nombre):
    pyplot.rcParams["figure.figsize"] = (12,7)
    pyplot.title('Curva de aprendizaje', fontsize=18)
    pyplot.ylim(0, 1.1)
    pyplot.xlabel('Pases', fontsize=16)
    pyplot.ylabel('Error', fontsize=16)
    pyplot.plot(historico.history['loss'], label = 'error entrenar. ' + '%.3f' % min(historico.history['loss']))
    pyplot.plot(historico.history['accuracy'], label='acierto entrenar. ' + '%.3f' % max(historico.history['accuracy']))
    pyplot.plot(historico.history['val_accuracy'], label='acierto validar. ' + "%.3f" % max(historico.history['val_accuracy']))
    pyplot.legend(fontsize=16)
    pyplot.savefig(senda + nombre + '.png', dpi = 96, facecolor = 'w', edgecolor = 'w', orientation = 'portrait', format = 'png',
                    transparent = False, bbox_inches = None, pad_inches = 0.1, metadata = None)
    if vg == 1: pyplot.show()
    pyplot.close()

def imprime(texto, rc=True):
    if rc:
        print(texto)
        log.write(texto + '\n')
    else:
        print(texto, end='')
        log.write(texto)

def SalidasPorCapas(model, prueba):
    # De la capa 1 hasta la última (la 0 es la de entrada) 'model.layers[1:]'
    obtener_salidas = tf.keras.backend.function([model.layers[0].input], [l.output for l in model.layers[1:]])
    salidas = obtener_salidas(prueba)
    imprime('Listas de nparray anidados (una lista por capa)  {}'.format(len(salidas)))
    # Para cada una de las capas
    for i1 in range(len(salidas)):
        for i2 in range(len(salidas[i1])):
            if (salidas[i1]).ndim == 1:
                imprime(' {:4.2f}'.format(salidas[i1][i2]), False)
            else:
                for i3 in range(len(salidas[i1][i2])):
                    if (salidas[i1][i2]).ndim == 1:
                        imprime(' {:4.2f}'.format(salidas[i1][i2][i3]), False)
                    else:
                        for i4 in range(len(salidas[i1][i2][i3])):
                            if (salidas[i1][i2][i3]).ndim == 1:
                                imprime(' {:4.2f}'.format(salidas[i1][i2][i3][i4]), False)
                            else:
                                for i5 in range(len(salidas[i1][i2][i3][i4])):
                                    if (salidas[i1][i2][i3][i4]).ndim == 1:
                                        imprime(' {:4.2f}'.format(salidas[i1][i2][i3][i4][i5]), False)
                                    else:
                                        imprime('')
                    imprime('')
                imprime('')
            imprime('')
        imprime('')
    log.flush()

def PruebaModelo(args, model, sendaD, fiprueba, lado):
    prueba = pd.read_csv(sendaD + fiprueba + '.csv', sep=',', header=None)
    random.seed(a=1001, version=2)
    prueba = prueba.sample(frac=1).reset_index(drop=True)
    prueba.columns = list(range(0, columnas))
    eti_prueba = prueba.iloc[:,0] 
    prueba.pop(0)
    prueba = np.array(prueba, dtype = np.float32)
    prueba = prueba / 255
    # Transformar las etiquetas del referente (dígito/forma) en 1 y las de los
    # restantes dígitos/formas en 0, para que los resultados tengan el mismo
    # sentido que el OCR
    eti_prueba = np.where(eti_prueba == args.referente, 1, 0)
    if args.tipoRNA == 1:
        prueba = prueba.reshape(prueba.shape[0], lado, lado, 1)
        eti_prueba = tf.keras.utils.to_categorical(eti_prueba, num_classes=2, dtype='float32')
    
    imprime('Resultados para la muestra de prueba {} con {:,} observaciones\n'.format(sendaD + fiprueba, len(prueba)))
    loss, acierto = model.evaluate(prueba,  eti_prueba, verbose=2)
    imprime('Error   {}'.format(loss))
    imprime('Acierto {}\n'.format(acierto))
    
    # Evalua las predicciones para la muestra de prueba.
    # El resultado es una lista de listas, la primera con tantos elementos como
    # observaciones de prueba, la segunda con tantos elementos como neuronas de
    # salida
    prediccion = model.predict(prueba)

    # Matriz de confusión
    nreferentes = aciertopositivo = aciertonegativo = falsopositivo = falsonegativo = 0
    for i in range(len(prueba)):
        if args.tipoRNA == 0:
            etiqueta = eti_prueba[i]
            predice = prediccion[i][0]
        else:
            if args.scv == 1:
                etiqueta = eti_prueba[i]
                predice = prediccion[i][0]
            else:
                # Hemos cambiado las etiquetas del referente a 1, luego son
                # referentes los que tiene un 1 en la segunda columna de
                # 'eti_prueba', la 1
                etiqueta = eti_prueba[i][1]
                # Nos quedamos con la predicción de la clase 1, la segunda
                # columna de 'predice', la 1
                predice = prediccion[i][1]

        # Hemos cambiado las etiquetas del referente a 1
        if etiqueta == 1: nreferentes +=1
        
        if predice > 0.5:
            if etiqueta == 1: aciertopositivo +=1
            else:             falsopositivo +=1
        else:
            if etiqueta == 1: falsonegativo +=1
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

    if args.objetos == 0:
        imprime('Dígitos 0 : {:7,} dígitos del 1 al 9: {:7,} total {:7,}\n'.format(nreferentes, deno3 - nreferentes, deno3))
        imprime('Falsos positivos   (dígitos del 1 al 9 confundidos con 0)          : {:7,} {:4.1f}%'.format(falsopositivo, pfp))
        imprime('Falsos negativos   (dígitos 0 no reconocidos)                      : {:7,} {:4.1f}%' .format(falsonegativo, pfn))
        imprime('Aciertos positivos (dígitos 0 reconocidos como tal -sensibilidad-) : {:7,} {:4.1f}%'.format(aciertopositivo, pap))
        imprime('Aciertos negativos (dígitos 1 al 9 reconocidos como distintos a 0) : {:7,} {:4.1f}%'.format(aciertonegativo, pan))
    else:
        imprime('{}: {:7,} resto de formas   : {:7,} total {:7,}\n'.format(rotulo[args.referente], nreferentes, deno3 - nreferentes, deno3))
        imprime('Falsos positivos   (formas confundidas con {} sin serlo)   : {:7,} {:4.1f}%'.format(rotulo[args.referente], falsopositivo, pfp))
        imprime('Falsos negativos   ({} no reconocidos)                     : {:7,} {:4.1f}%' .format(rotulo[args.referente], falsonegativo, pfn))
        imprime('Aciertos positivos ({} reconocidos como tal -sensibilidad-): {:7,} {:4.1f}%'.format(rotulo[args.referente], aciertopositivo, pap))
        imprime('Aciertos negativos ("no {}" reconocido que no lo son)      : {:7,} {:4.1f}%'.format(rotulo[args.referente], aciertonegativo, pan))
    imprime('Aciertos totales                                                   : {:7,} {:4.1f}%'.format(aciertopositivo + aciertonegativo, pat))
    imprime('\n')

    # Distribución de las predicciones, separadas para referente y resto
    # El referente es de la clase 1
    if args.objetos == 0:
        titulos = ['Imágenes que no son un 0', 'Imágenes del 0']
    else:
        titulos = ['Formas distintas a ' + rotulo[args.referente], rotulo[args.referente]]
    
    prediccion_clase = []
    if eti_prueba.ndim == 1 or eti_prueba.shape[1] == 1:
        clases_prueba = tf.keras.utils.to_categorical(eti_prueba, num_classes=None, dtype='int')
        if clases_prueba.ndim == 1:
            nclases = 1
        else:
            nclases = clases_prueba.shape[1]
        for i in range(nclases):
            # 'boolpre' valdrá True en todas las filas en las que la columna i
            # de 'clases_prueba' sea 1 que son las observaciones de la clase
            # 'i'
            if nclases == 1:
                boolpre = clases_prueba == 1
            else:
                boolpre = clases_prueba[:, i] == 1
            # 'prediccion_clase' extraerá los valores de la columna i de
            # 'predicciones', para las filas en las que 'boolpre' es True
            prediccion_clase.append(prediccion[boolpre])
            GraficoDistribucion(args.vg, 0, 1, titulos[i], prediccion_clase[i], 0.5, sendaG, prefijo + fiprueba + '_dis_' + str(i))
    else:
        for i in range(eti_prueba.ndim):
            # 'boolpre' valdrá True en todas las filas en las que la columna i
            # de 'eti_prueba' sea 1 que son las filas de la clase 'i'
            boolpre = eti_prueba[:, i] == 1
            # 'prediccion_clase' extraerá los valores de la columna i de
            # 'predicciones', para las filas en las que 'boolpre' es True
            prediccion_clase.append(prediccion[boolpre, i])
            GraficoDistribucion(args.vg, 0, 1, titulos[i], prediccion_clase[i], 0.5, sendaG, prefijo + fiprueba + '_dis_' + str(i))

def PintaReferente(args, model, lado, intentos, puntos, corte, margen, prefijo, caso):
    pixeles = lado * lado
    ladom1 = lado - 1
    nazar = 0
    if args.tipoRNA == 0 or (args.tipoRNA == 1 and args.scv == 1):
        img_mejor = np.zeros((1, lado, lado), dtype = np.float32)
        if puntos == -1:
            # todos los píxeles
            for y in range(lado):
                for x in range(lado):
                    img_mejor[0, x, y] = 1.0
        else:
            for i in range(puntos):
                x = random.randrange(0, lado)
                y = random.randrange(0, lado)
                img_mejor[0, x, y] = 1.0
        prediccion = model.predict(img_mejor.reshape(1, pixeles))
        val_mejor = prediccion[0][0]
        imprime('Inicio          [{:8.5f}]'.format(val_mejor))
        
        if args.barridoini == 1:
            
            # Recorrido sistemático añadiendo puntos que mejoran
            puntos_fin, img_mejor, val_mejor, mas = AjustaMas(args, model, lado, img_mejor, val_mejor)
            imprime('Ajuste +         {:8.5f}'.format(val_mejor))
        else:
            puntos_fin = 0
            mas = 0
        if val_mejor < corte:
            
            # Es mejorable, modificaciones al azar
            val_prueba = val_mejor 
            img_prueba = img_mejor.copy()
            nazar = intentos
            for i in range(intentos):
                if (i % 1000) == 0:
                    print('{:7,}'.format(i))

                # Cambia 1 pixel en la imagen de prueba, alternando su valor
                x = random.randrange(0, lado)
                y = random.randrange(0, lado)
                if puntos == -1:
                    # Solo quitar en el 95% de las ocasiones
                    if img_prueba[0, x, y] == 1.0 or (random.randrange(0, 100) < 5):
                        img_prueba[0, x, y] = 0.0
                        prediccion = model.predict(img_prueba.reshape(1, pixeles))
                else:
                    img_prueba[0, x, y] = 1.0 - img_prueba[0, x, y]
                    prediccion = model.predict(img_prueba.reshape(1, pixeles))
            
                # Se aceptan retrocesos en el 'margen' por cien de los casos
                if (prediccion[0][0] > val_prueba) or (random.randrange(0, 100) < margen):
            
                    # Consolida el cambio
                    val_prueba = prediccion[0][0]
                
                    if val_prueba > val_mejor:
                        print('{:7,} {:10.7f} {:8.5f}'.format(i, val_prueba - val_mejor, val_prueba))
                        val_mejor = val_prueba
                        img_mejor = img_prueba.copy()
                
                    # Con un valor superior al corte se termina
                    if val_prueba > corte:
                        nazar = i + 1
                        break
                else:
                
                    # Deshace el cambio
                    img_prueba[0, x, y] = 1.0 - img_prueba[0, x, y]
            imprime('MC               {:8.5f}'.format(val_mejor))
    else:    
        img_mejor = np.zeros((1, lado, lado, 1), dtype = np.float32)
        if puntos == -1:
            # todos los píxeles
            for y in range(lado):
                for x in range(lado):
                    img_mejor[0, x, y, 0] = 1.0
        else:
            for i in range(puntos):
                x = random.randrange(0, lado)
                y = random.randrange(0, lado)
                img_mejor[0, x, y, 0] = 1.0
        prediccion = model.predict(img_mejor)
        val_mejor = prediccion[0][1]
        imprime('Inicio  {:8.5f} [{:8.5f}]'.format(prediccion[0][0], val_mejor))

        if args.barridoini == 1:
            
            # Recorrido sistemático añadiendo puntos que mejoran
            puntos_fin, img_mejor, val_mejor, mas = AjustaMas(args, model, lado, img_mejor, val_mejor)
            imprime('Ajuste +         {:8.5f}'.format(val_mejor))
        else:
            puntos_fin = 0
            mas = 0
        if val_mejor < corte:
            
            # Es mejorable, modificaciones al azar
            val_prueba = val_mejor 
            img_prueba = img_mejor.copy()
            nazar = intentos
            for i in range(intentos):
                if (i % 1000) == 0:
                    print('{:7,}'.format(i))
        
                # Cambia 1 pixel en la imagen de prueba, alternando su valor
                x = random.randrange(0, lado)
                y = random.randrange(0, lado)
                if puntos == -1:
                    # Solo quitar en el 95% de las ocasiones
                    if img_prueba[0, x, y, 0] == 1.0 or (random.randrange(0, 100) < 5):
                        img_prueba[0, x, y, 0] = 0.0
                        prediccion = model.predict(img_prueba)
                else:
                    img_prueba[0, x, y, 0] = 1.0 - img_prueba[0, x, y, 0]
                    prediccion = model.predict(img_prueba)
            
                # Se aceptan retrocesos en el 'margen' por cien de los casos
                if (prediccion[0][1] > val_prueba) or (random.randrange(0, 100) < margen):
            
                    # Consolida el cambio
                    val_prueba = prediccion[0][1]
                
                    if val_prueba > val_mejor:
                        print('{:7,} {:10.7f} {:8.5f}'.format(i, val_prueba - val_mejor, val_prueba))
                        val_mejor = val_prueba
                        img_mejor = img_prueba.copy()
                
                    # Con un valor superior al corte se termina
                    if val_prueba > corte:
                        nazar = i + 1
                        break
                else:

                    # Deshace el cambio
                    img_prueba[0, x, y, 0] = 1.0 - img_prueba[0, x, y, 0]
            imprime('MC               {:8.5f}'.format(val_mejor))

    # Recorrido sistemático quitando puntos que no mejoran
    puntos_fin, img_mejor, val_mejor, menos = AjustaMenos(args, model, lado, img_mejor, val_mejor)        
    imprime('Ajuste -         {:8.5f}'.format(val_mejor))
    
    if val_mejor < corte:
        # Una última oportunidad
        puntos_fin, img_mejor, val_mejor, remas = AjustaMas(args, model, lado, img_mejor, val_mejor)
        mas += remas
        imprime('Ajuste + final   {:8.5f}'.format(val_mejor))

    imprime('Proceso de ajuste: [Dimensión {:,}] {:,} puntos más y {:,} puntos menos. Puntos totales {:,}'.format(pixeles, mas, menos, puntos_fin))
    if val_mejor > corte:
        imprime('{:9,} {:11,} {:9.5f} {:8,} Alcanzado objetivo de       [{:10.3f}%] tras {:,} intentos MC'.format(intentos, puntos, corte, margen, 100.0 * corte, nazar))
    else:
        imprime('{:9,} {:11,} {:9.5f} {:8,} NO se alcanzó el objetivo de {:10.3f}% tras {:,} intentos MC'.format(intentos, puntos, corte, margen, 100.0 * corte, nazar))
    imprime('                                         certeza                      {:10.3f}%'.format(100.0 * val_mejor))
    GraficoImagen(args.vg, img_mejor.reshape((lado, lado)), sendaG, prefijo + 'mejor_', str(caso + 1))

def AjustaMas(args, model, lado, img_mejor, mejorada):
    # Agregar puntos que mejoran el resultado

    print('Ajustando (+) ...')
    mas = 0
    puntos_fin = 0
    if args.tipoRNA == 0 or (args.tipoRNA == 1 and args.scv == 1):
        for y in range(lado):
            for x in range(lado):
                if img_mejor[0, x, y] == 0.0:
                    img_mejor[0, x, y] = 1.0
                    prediccion = model.predict(img_mejor.reshape(1, pixeles))
                    if prediccion[0][0] > mejorada:
                        mejorada = prediccion[0][0]
                        mas += 1
                        puntos_fin += 1
                    else:
                        img_mejor[0, x, y] = 0.0
                else:
                    puntos_fin += 1
    else:
        for y in range(lado):
            for x in range(lado):
                if img_mejor[0, x, y, 0] == 0.0:
                    img_mejor[0, x, y, 0] = 1.0
                    prediccion = model.predict(img_mejor)
                    if prediccion[0][1] > mejorada:
                        mejorada = prediccion[0][1]
                        mas += 1
                        puntos_fin += 1
                    else:
                        img_mejor[0, x, y] = 0.0
                else:
                    puntos_fin += 1
    return puntos_fin, img_mejor, mejorada, mas

def AjustaMenos(args, model, lado, img_mejor, mejorada):

    # Eliminar puntos superfluos, que no empeoran el resultado
    print('Ajustando (-) ...')
    menos = 0
    puntos_fin = 0
    if args.tipoRNA == 0 or (args.tipoRNA == 1 and args.scv == 1):
        for y in range(lado):
            for x in range(lado):
                if img_mejor[0, x, y] == 1.0:
                    img_mejor[0, x, y] = 0.0
                    prediccion = model.predict(img_mejor.reshape(1, pixeles))
                    if prediccion[0][0] >= mejorada:
                        mejorada = prediccion[0][0]
                        menos += 1
                    else:
                        img_mejor[0, x, y] = 1.0
                        puntos_fin += 1
    else:
        for y in range(lado):
            for x in range(lado):
                if img_mejor[0, x, y, 0] == 1.0:
                    img_mejor[0, x, y, 0] = 0.0
                    prediccion = model.predict(img_mejor)
                    if prediccion[0][1] >= mejorada:
                        mejorada = prediccion[0][1]
                        menos += 1
                    else:
                        img_mejor[0, x, y] = 1.0
                        puntos_fin += 1
    return puntos_fin, img_mejor, mejorada, menos

# ---------------------------------------------------------------
# Fin de la definición de funciones
#
if (args.montecarlo > 0) and (len(args.sorteos) != args.montecarlo or len(args.pini) != args.montecarlo or len(args.cortemc) != args.montecarlo or len(args.margenmc) != args.montecarlo):
    print('Datos incoherentes. Numero de imagenes del referente {:,} sorteos {:,} puntos iniciales {:,} corte {:,} margen {:,}'.format(args.montecarlo, len(args.sorteos), len(args.pini), len(args.corte), len(args.margenmc)))
    sys.exit()

rotulo = ['Círculos  ','Cuadrados ','Triangulos']

# Donde están los datos
sendaD = args.dird

# Donde se guardarán los resultados, asegurando que las caprpetas existen
sendaR = args.dirr
sendaG = sendaR + 'Imagenes/'
if not os.path.exists(args.dirr): os.makedirs(args.dirr)
if not os.path.exists(sendaR): os.makedirs(sendaR)
if not os.path.exists(sendaG): os.makedirs(sendaG)

# 'prefijo' servirá para que los ficheros con salidas sean distintos según las
# opciones y no se sobreescriban al ejecutar con unas u otras
if args.tipoRNA == 0:
    if args.capas_ocultas == 0:
        prefijo = args.fientrena + '_' + args.fiprueba + '_ds_0_' + '__' + str(args.referente)
    else:
        prefijo = args.fientrena + '_' + args.fiprueba + '_ds_1_' + str(args.objetos) + '_' + str(args.neuronas) + '_' + str(args.referente)
else:
    prefijo = args.fientrena + '_' + args.fiprueba + '_cv_' + str(args.objetos) + '_' + str(args.fil) + '_' + str(args.mcv) + '_' + str(args.ccv) + '_' + str(args.scv) + '_' + str(args.referente)

# Fichero de resultados
log = open(sendaR + prefijo + '.log', 'w')

# Muestra de entrenamiento.  Leemos el fichero
entrena = pd.read_csv(sendaD + args.fientrena + '.csv', sep=',', header=None)
random.seed(a=2947, version=2)
# Reordenamos ('sample') las imágenes al azar
entrena = entrena.sample(frac=1).reset_index(drop=True)

# los ficheros con los datos no tienen cabeceras que se puedan usar para
# identificar las columnas, por lo que las creamos con números del 0 al número
# de columnas menos 1
columnas = entrena.shape[1]
entrena.columns = list(range(0, columnas))
# Las etiquetas, que nos dicen que contiene la imagen, están en la primera
# columna, la columna '0'.  Las extraemos en 'eti_entrena' y la eliminamos de
# 'entrena' con 'pop'
eti_entrena = entrena.iloc[:,0] 
entrena.pop(0)

# Convertimos 'entrena' en una matriz de números de coma flotante y
# normalizamos los valores entre 0.0 y 1.0, al dividir por 255.
entrena = np.array(entrena, dtype = np.float32)
entrena = entrena / 255

pixeles = entrena.shape[1]
lado = int(math.sqrt(pixeles))

# Mostrar 32 imagenes
numimg = 32
imagenesej = np.zeros((numimg, pixeles), dtype = np.float32)
for i in range(numimg):
    for j in range(pixeles):
        imagenesej[i,j] = entrena[i,j]
GraficoImagenes(args.vg, imagenesej, lado, sendaG, prefijo + '_Imagenes entrenar')

# Muestra de prueba.  La tratamos igual que a la de entrenamiento
prueba = pd.read_csv(sendaD + args.fiprueba + '.csv', sep=',', header=None)
random.seed(a=1001, version=2)
prueba = prueba.sample(frac=1).reset_index(drop=True)
prueba.columns = list(range(0, columnas))
eti_prueba = prueba.iloc[:,0] 
prueba.pop(0)
prueba = np.array(prueba, dtype = np.float32)
prueba = prueba / 255

# Mostrar 32 imagenes
numimg = 32
imagenesej = np.zeros((numimg, pixeles), dtype = np.float32)
for i in range(numimg):
    for j in range(pixeles):
        imagenesej[i,j] = prueba[i,j]
GraficoImagenes(args.vg, imagenesej, lado, sendaG, prefijo + '_Filtros')

# Transformar las etiquetas del referente (dígito/forma) en 1 y las de los
# restantes dígitos/formas en 0, para que los resultados tengan el mismo
# sentido que el OCR
eti_entrena = np.where(eti_entrena == args.referente, 1, 0)
eti_prueba = np.where(eti_prueba == args.referente, 1, 0)

# Esto servira para que se ir salvando el mejor modelo que se vaya encontrando
# durante el entrenamiento, el que tenga el mejor 'val_accuracy'
check_point = tf.keras.callbacks.ModelCheckpoint(sendaR + 'modelo_' + prefijo + '.tf', monitor='val_accuracy', verbose=0, save_best_only=True, mode='max', save_format='tf')

# Define la RNA
random.seed(a=701, version=2)
if args.tipoRNA == 0:
    # RNA Densa
    if args.capas_ocultas == 0:
    
        # Sin capa intermedia de neuronas
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(pixeles,)),
                                            tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Zeros(), activation='sigmoid')])
    else:
    
        # Con una capa intermedia de 'args.neuronas' neuronas
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(pixeles,)),
                                            tf.keras.layers.Dense(args.neuronas, kernel_initializer=tf.keras.initializers.Zeros(), activation='sigmoid'),
                                            tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Zeros(), activation='sigmoid')])
    # Como sólo hay un valor de salida podemos utilizar como función de error
    # el tradicional error cuadrático medio
    ferror = 'mean_squared_error'
else:
    # RNA convolucional
    # El tensor de entrada hay que transformarlo a uno con 4 ejes, hay que
    # añadir el cuarto:
    #   el número de imagenes: entrena.shape[0]
    #   la dimensión de la imagen lado x lado
    #   el canal de color, que en este caso es 1 por ser en blanco y negro
    entrena = entrena.reshape(entrena.shape[0], lado, lado, 1)
    prueba = prueba.reshape(prueba.shape[0], lado, lado, 1)
    # Para usar la función de error 'categorical_crossentropy' debemos
    # transformar las etiquetas en un vector de ceros con tantos elementos como
    # clases y un 1 en la clase correcta
    eti_entrena = tf.keras.utils.to_categorical(eti_entrena, num_classes=2, dtype='float32')
    eti_prueba = tf.keras.utils.to_categorical(eti_prueba, num_classes=2, dtype='float32')

    # La forma de los elementos de entrada es (lado, lado, 1)
    # La matriz de pesos a optimizar es 'kernel_size=(args.mcv, args.mcv)' y
    # hay tantas matrices como filtros 'args.fil'
    # Por lo tanto el número de parametros a optimizar será:
    # args.fil * (args.mcv * args.mcv + 1) el '+ 1' es para el sesgo
    # la capa de 'MaxPooling2D' recibira como entradas los 'args.fil' donde
    # cada uno de ellos tendrá tantos valores como desplazamientos se hayan
    # podido realizar sobre la imagen con el marco de convolución, es decir
    # (lado - args.mcv + 1) desplazamientos a la derecha por otros tantos hacia
    # abajo.  La salida de esta capa reducirá esos valores por un factor de
    # (args.ccv * args.ccv) que será aplanada en la capa siguiente 'Flatten()'
    # para que sirva de entrada a la capa de salida que es una capa densa con
    # tantas neurona como clases.  Todo esto se mostrará con 'model.summary'

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(args.fil, kernel_size=(args.mcv, args.mcv), activation='relu', input_shape=(lado, lado, 1)))
    if args.ccv > 0: model.add(tf.keras.layers.MaxPooling2D(pool_size=(args.ccv, args.ccv)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(args.scv, activation='softmax'))
    
    # Mediremos el error con la entropia cruzada, una extension al error medio
    # cuando el resultado que se compara no es un sólo valor, si no varios,
    # aquí son dos: las probabilidades de que sea el rerente (clase 1) y de que
    # no lo sea (clase 2)
    ferror = 'categorical_crossentropy'

# imprimimos el resumen del modelo
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
resumen = "\n".join(stringlist)
imprime(resumen)

# Compilamos el modelo
model.compile(optimizer='adam', loss=ferror, metrics=['accuracy'])
# Entrenamos la RNA
historico = model.fit(entrena, eti_entrena, epochs=args.pases, verbose=2, validation_data=(prueba, eti_prueba), callbacks=[check_point])

# Dibujamos la evolución
GraficoEvolucion(args.vg, historico, sendaG, prefijo + 'evolucion_ajuste')

# Cargar el mejor modelo encontrado
model = tf.keras.models.load_model(sendaR + 'modelo_' + prefijo + '.tf')

#SalidasPorCapas(model, prueba[:25])

# Extraer los pesos de la primera capa
if args.tipoRNA == 0:
    # Dibujar la matriz de pesos
    weights = model.layers[1].get_weights()
    pesos = weights[0]
    sesgo = weights[1]
    if args.capas_ocultas == 0:
        # Transformar los pesos con valor 0 en -1 para hacer la imagen similar
        # al OCR.  Los pesos 0 son los que la entrada es irelevante: los puntos
        # en los bordes que están vacios en todas las imagenes
        pesos = np.where(pesos == 0, -1, pesos)
        pesos1 = pesos[:,0].reshape((lado, lado))
        GraficoImagen(args.vg, pesos1, sendaG, 'pesos_', prefijo + '__')
    else:
        pesosc2 = model.layers[2].get_weights()[0]
        sesgoc2 = model.layers[2].get_weights()[1]
        for i in range(args.neuronas):
            pesosa = pesos[:,i].reshape((lado, lado))
            # Multiplicamos los pesos por el peso de la capa intermedia hacia
            # la de salida
            pesosa = pesosa * pesosc2[i,0]
            # Transformar los pesos con valor 0 en -1 para hacer la imagen
            # similar al OCR, pero después de haber multiplicado por el peso de
            # la capa 2
            pesosa = np.where(pesosa == 0, -1, pesosa)
            GraficoImagen(args.vg, pesosa, sendaG, 'pesos_', prefijo + '_' + str(i))
else:
    weights = model.layers[0].get_weights()
    pesos = weights[0]
    sesgo = weights[1]
    numimg = pesos.shape[3]
    imagenes = np.zeros((numimg, pesos.shape[0] * pesos.shape[1]), dtype = np.float32)
    for n in range(numimg):
        i = 0
        for j in range(pesos.shape[0]):
            for k in range(pesos.shape[1]):
                imagenes[n,i] = pesos[j, k, 0, n]
                i += 1
    GraficoImagenes(args.vg, imagenes, pesos.shape[0], sendaG, prefijo + '_Imagenes probar')
imprime('Muestra de entrenamiento {} con {:,} observaciones\n'.format(sendaD + args.fientrena, len(entrena)))
log.flush()

if args.tipoRNA == 0:
    if args.capas_ocultas == 0:
        imprime('Sesgo         {:7.4f}\n'.format(sesgo[0]))
    else:
        imprime('Sesgos        {:7.4f} {:7.4f}\n'.format(sesgo[0], sesgoc2[0]))
        imprime('Pesos  capa 2')
        for i in range(args.neuronas):
            imprime(' {:7.4f}'.format(pesosc2[i,0]))
        imprime('\n')
PruebaModelo(args, model, sendaD, args.fiprueba, lado)
log.flush()
if len(args.fisorpresa) > 0:
    PruebaModelo(args, model, sendaD, args.fisorpresa, lado)
    log.flush()

if args.montecarlo > 0:
    if args.objetos == 0:
        imprime('Dibujar ceros\n')
    else:
        imprime('Dibujar {}\n'.format(rotulo[args.referente]))
    imprime('Número de imagenes a crear del referente {:6,}\n'.format(args.montecarlo))
    imprime('  Sorteos P.iniciales     Corte % Margen')
    imprime('--------- ----------- --------- -------- ---------------------------------------------------------------')
    random.seed(a=1355, version=2)
    for i in range(args.montecarlo):
        PintaReferente(args, model, lado, args.sorteos[i], args.pini[i], args.cortemc[i], args.margenmc[i], prefijo, i)
        imprime('--------- ----------- --------- -------- ---------------------------------------------------------------')
        log.flush()
    imprime('\n')

log.close()
