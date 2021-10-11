import numpy as np
import math as math

def Histograma(Imagen):
  M = Imagen.shape[0]
  N = Imagen.shape[1]
  Histograma = np.zeros((1,256))

  for i in range(M):
    for j in range(N):
      tono = Imagen[i,j]
      Histograma[0,tono] += 1

  return Histograma

def Otsu(imagen):
    # Paso 1: Calculamos el histograma normalizado (dividido por la cantidad de píxeles)
    p = Histograma(imagen) / np.size(imagen)

    # Paso 2: Definimos un vector (np.zeros) de varianza inter-clase (dimensión 256)
    L = 256  # por ser uint8
    Varianza = np.zeros(L)

    # Paso 3: Armo fórmulas para calcular la varianza 256 veces, con t=1...256
    for t in range(1, L):
        w0 = mu0 = mu1 = varianza = 0

        for i in range(t):
            w0 = w0 + p[0, i]
            mu0 = mu0 + i * p[0, i]

        for i in range(t, L):
            mu1 = mu1 + i * p[0, i]

        w1 = 1 - w0
        mu0 = mu0 / w0
        mu1 = mu1 / w1
        Varianza[t - 1] = w1 * w0 * (mu0 - mu1) ** 2

    # Paso 4: Me quedo con el nivel de gris que maximiza la varianza (busco el máximo del vector de varianzas y me fijo la posición en la que está el máximo, ese va a ser el t óptimo)

    t_optimo = np.argmax(Varianza)

    # Paso 5: Binarizamos la imagen con el umbral t óptimo

    M = imagen.shape[0]
    N = imagen.shape[1]
    imagen_resultante = np.zeros(imagen.shape)

    for i in range(M):
        for j in range(N):
            if imagen[i, j] > t_optimo:
                imagen_resultante[i, j] = 255

    return imagen_resultante, t_optimo

def Region_growing(Semilla_inicial, umbral_min, umbral_max, imagen):

    i = Semilla_inicial[0]
    j = Semilla_inicial[1]

    if umbral_max > imagen[i, j] > umbral_min:  # Propago sólamente si la semilla inicial cumple con el rango de umbrales.
        Lista_semillas = []
        Lista_semillas.append(Semilla_inicial)

        Imagen_Resultante = np.zeros(imagen.shape)
        Imagen_rastreo = np.zeros(imagen.shape)  # La imágen de rastreo se usa para registrar los puntos por los que ya pasé y sea más fácil de rastrearlos.

        # Registro la semilla inicial.
        Imagen_Resultante[i, j] = 50
        Imagen_rastreo[i, j] = 1

        # Obtengo las dimensiones de la imágen.
        M = imagen.shape[0]
        N = imagen.shape[1]

        while Lista_semillas != []:

            Semilla = Lista_semillas.pop()  # Extraigo la primer semilla dentro de Lista_semillas y a su vez la quito. Como si fuese un dispenser de semillas.
            i = Semilla[0]
            j = Semilla[1]

            # Una solución al problema de los bordes
            imin = i - 1
            imax = i + 2
            jmin = j - 1
            jmax = j + 2

            # Estos son los límites de la vecindad de 8 de mi semilla(límite máx excluyente).
            # A medida que me voy encontrando con los bordes voy modificando estos límites para no tener problemas con las dimensiones.

            if i == 0:
                imin = i
            elif i == M - 1:
                imax = i + 1
            if j == 0:
                jmin = j
            elif j == N - 1:
                jmax = j + 1

            for m in range(imin, imax):  # Recorro la vecindad de la semilla con los límites correspondientes.
                for n in range(jmin, jmax):

                    if (m, n) != (i, j):  # El centro ya es la semilla.

                        if Imagen_rastreo[m, n] != 1:  # Verifico que no haya pasado por este punto previamente.

                            if umbral_max >= imagen[m, n] >= umbral_min:  # Si está dentro del rango lo pinto de blanco
                                                                          # en la imágen resultante y lo agrego como
                                                                          # nueva semilla en Lista_semillas (refill del
                                                                          # dispenser).
                                Lista_semillas.append((m, n))
                                Imagen_Resultante[m, n] = 50

                        Imagen_rastreo[m, n] = 1  # indico con un 1 que ya pasé por este punto en la imágen de rastreo.

        return Imagen_Resultante

    else:
        print("La semilla inicial no cumple con la regla de asignación fijada.")

def Kmeans_de3(imagen, error):
    # Paso 1: Defino centroides iniciales

    """
    c1 = 100
    c2 = 185
    c3 = 255
    """
    c1 = 100
    c2 = 185
    c3 = 255

    M = imagen.shape[0]
    N = imagen.shape[1]
    # Paso 2: Recorremos píxel a píxel la imagen y calculamos la distancia de cada píxel (nivel de gris) al centroide
    # Nivel de gris del píxel es n
    cond = 0
    while (cond == 0):
        co1 = c1
        co2 = c2
        co3 = c3
        Imagen_aux = np.zeros((M, N))

        for i in range(M):
            for j in range(N):

                d = np.zeros(3)
                n = imagen[i, j]

                d[0] = abs(n - co1)
                d[1] = abs(n - co2)
                d[2] = abs(n - co3)

                # Cuál es la mínima distancia?
                minima_dist = d[0]
                cluster = 1
                for k in range(1, len(d)):
                    if (d[k] <= minima_dist):
                        minima_dist = d[k]
                        cluster = k + 1

                # Tengo que incluír ese píxel en el primer cluster por ejemplo.. Cómo?
                # Una opción

                Imagen_aux[i, j] = cluster  # Asigno 1, si esta en el cluster 1, Si está en el segundo cluster le asigno otro valor, y en el tercero otro valor

        # Paso 3: Una vez que ya recorrí toda la imagen tengo que agarrar los píxeles que forman parte de cada uno de los clusters y hacer un promedio que va a resultar en el valor del nuevo centroide del clúster 1
        # Debo transformar c1, c2 y c3 a números enteros
        # Ejemplo para el clúster 1

        c1 = int(np.mean(imagen[(Imagen_aux == 1)]))
        c2 = int(np.mean(imagen[(Imagen_aux == 2)]))
        c3 = int(np.mean(imagen[(Imagen_aux == 3)]))

        # Paso 4: Una vez recalculados los centroides, repetir el proceso (desde Paso 2), hasta que el valor de los centroides no cambie más o tengan una diferencia muy chica (definir un error permitido)
        if (abs(c1 - co1) <= error and abs(c2 - co2) <= error and abs(c3 - co3) <= error):
            cond = 1

    # Paso 5: La imagen resultante va a tener la pinta de Imagen_aux pero con 3 valores correspondientes a los 3 valores de los últimos centroides calculados
    return Imagen_aux

def Region_growing_4_lados(Semilla_inicial, umbral_min, umbral_max, imagen):

    # Paso 1: Defino las coordenadas del punto semilla
    Lista_semillas = []
    Lista_semillas.append(Semilla_inicial)
    print(Lista_semillas)
    print(Lista_semillas != [])

    Imagen_Resultante = np.zeros(imagen.shape)
    Imagen_rastreo = np.zeros(imagen.shape)

    i = Semilla_inicial[0]
    j = Semilla_inicial[1]
    Imagen_Resultante[i, j] = 1
    Imagen_rastreo[i, j] = 1

    M = imagen.shape[0]
    N = imagen.shape[1]

    # Paso 2: Defino la regla de asignación. Defino un valor (ej: q) que me indique +- cuanto puede valer el píxel vecino para incluirlo en la región
    # Ejemplo. Nivel de gris del píxel semilla es 140. q=10
    # la regla de asignación sería con el rango 130-150
    while Lista_semillas != []:
        Semilla = Lista_semillas.pop()
        i = Semilla[0]
        j = Semilla[1]

        # Vecinos (Paso 3)
        if Imagen_rastreo[i - 1, j] != 1:  # izquierda

            if umbral_max >= imagen[i - 1, j] >= umbral_min:
                Lista_semillas.append((i - 1, j))
                Imagen_Resultante[i - 1, j] = 50
            Imagen_rastreo[i - 1, j] = 1

        if Imagen_rastreo[i + 1, j] != 1:  # derecha

            if umbral_max >= imagen[i + 1, j] >= umbral_min:
                Lista_semillas.append((i + 1, j))
                Imagen_Resultante[i + 1, j] = 50
            Imagen_rastreo[i + 1, j] = 1

        if Imagen_rastreo[i, j - 1] != 1:  # arriba

            if umbral_max >= imagen[i, j - 1] >= umbral_min:
                Lista_semillas.append((i, j - 1))
                Imagen_Resultante[i, j - 1] = 50
            Imagen_rastreo[i, j - 1] = 1

        if Imagen_rastreo[i, j + 1] != 1:  # abajo

            if umbral_max >= imagen[i, j + 1] >= umbral_min:
                Lista_semillas.append((i, j + 1))
                Imagen_Resultante[i, j + 1] = 50
            Imagen_rastreo[i, j + 1] = 1

        # Arriba de la derecha

        if Imagen_rastreo[i + 1, j - 1] != 1:  # abajo

            if umbral_max >= imagen[i + 1, j - 1] >= umbral_min:
                Lista_semillas.append((i + 1, j - 1))
                Imagen_Resultante[i + 1, j - 1] = 50
            Imagen_rastreo[i + 1, j - 1] = 1

        # Arriba a la izquierda

        if Imagen_rastreo[i - 1, j - 1] != 1:  # abajo

            if umbral_max >= imagen[i - 1, j - 1] >= umbral_min:
                Lista_semillas.append((i - 1, j - 1))
                Imagen_Resultante[i - 1, j - 1] = 50
            Imagen_rastreo[i - 1, j - 1] = 1

        # Abajo a la derecha

        if Imagen_rastreo[i + 1, j + 1] != 1:  # abajo

            if umbral_max >= imagen[i + 1, j + 1] >= umbral_min:
                Lista_semillas.append((i + 1, j + 1))
                Imagen_Resultante[i + 1, j + 1] = 50
            Imagen_rastreo[i + 1, j + 1] = 1

        # Abajo a la izquierda

        if Imagen_rastreo[i + 1, j - 1] != 1:  # abajo

            if umbral_max >= imagen[i + 1, j - 1] >= umbral_min:
                Lista_semillas.append((i + 1, j - 1))
                Imagen_Resultante[i + 1, j - 1] = 50
            Imagen_rastreo[i + 1, j - 1] = 1

    return Imagen_Resultante

def watershed(img, img_pad):

    # img_pad es el resultado de aplicar un zeropadding (con kernel 3x3) a la imagen original img

    etiquetas = [0]  # Variable que se va a necesitar despues

    # ---------- Paso 1: Identificar el valor mínimo y máximo en la imagen ----------
    maximo = img.max()
    minimo = img.min()

    if minimo == 0:  # Condicion para el caso donde la imagen original tiene pixeles con valor 0
        minimo += 1

        while (np.where(img_pad == minimo) == []):
            minimo += 1

    # ---------- Paso 2: Defino la imagen resultante con np.zeros... ----------
    img_ws = np.zeros((img_pad.shape[0], img_pad.shape[1]))

    # ---------- Paso 3: Voy a realizar un ciclo con tantas iteraciones como niveles de gris que tiene la imagen ----------
    lista_coor_total = []
    for n in range(minimo, maximo):

        # ----- Paso 4: Identificar donde estan ubicados los píxeles que tienen el nivel de gris n -----
        coordenadas = np.where(img_pad == n)

        lista_coor = []
        for c in range(len(coordenadas[0])):
            lista_coor.append([coordenadas[0][c], coordenadas[1][c]])

        # ----- Paso 5: me voy parando en cada uno de los píxeles que tiene nivel de gris n -----
        # Tengo que analizar los vecinos (como en region growing)
        # Hay algún vecino etiquetado? Si la respuesta es Sí, a ese píxel le asigno la misma etiqueta
        # Si la respuesta en NO, le asigno una nueva etiqueta
        # Si dos vecinos tienen etiquetas diferentes le asigno la etiqueta correspondiente a la línea divisoria de agua.

        seguir = 1  # inicializo la variable de control

        while lista_coor != []:

            cant_anterior = len(lista_coor)  # inicializo la otra variable de control
            aux = []  # lista auxiliar

            while lista_coor != []:
                coor = lista_coor.pop()
                i = coor[0]
                j = coor[1]

                etiquetas_vecinos = []  # Lista donde se guardan las etiquetas de los vecinos

                # Considero los 4 vecinos principales:
                if img_ws[i - 1, j] != 0:
                    etiquetas_vecinos.append(img_ws[i - 1, j])

                if img_ws[i, j - 1] != 0:
                    etiquetas_vecinos.append(img_ws[i, j - 1])

                if img_ws[i, j + 1] != 0:
                    etiquetas_vecinos.append(img_ws[i, j + 1])

                if img_ws[i + 1, j] != 0:
                    etiquetas_vecinos.append(img_ws[i + 1, j])

                # Por si tambien quiero considerar los otros 4 vecinos diagonales, sino comentar este bloque:
                if img_ws[i - 1, j - 1] != 0:
                    etiquetas_vecinos.append(img_ws[i - 1, j - 1])

                if img_ws[i - 1, j + 1] != 0:
                    etiquetas_vecinos.append(img_ws[i - 1, j + 1])

                if img_ws[i + 1, j - 1] != 0:
                    etiquetas_vecinos.append(img_ws[i + 1, j - 1])

                if img_ws[i + 1, j + 1] != 0:
                    etiquetas_vecinos.append(img_ws[i + 1, j + 1])

                # Verifico si habia vecinos con etiquetas:
                if etiquetas_vecinos == []:
                    if seguir == 1:
                        aux.append(
                            coor)  # Pixel candidato a tener etiqueta nueva, pero por ahora lo vuelvo a meter al ciclo
                    else:
                        nueva_etiqueta = etiquetas[
                                             -1] + 1  # Tomo el ultimo valor en la lista global de etiquetas, y le sumo uno...
                        etiquetas.append(nueva_etiqueta)  # ...guardo ese nuevo valor en la lista...
                        img_ws[i, j] = nueva_etiqueta  # ... y se la asigno al pixel

                        # seguir = 1  #Descomentar si se quiere verificar aún más, haciendo un ciclo extra cada vez que se asigna una etiqueta nueva... pero ahora sí aumenta el tiempo de ejecución
                else:
                    etiquetas_vecinos_unicas = list(set(etiquetas_vecinos))  # Elimino las etiquetas duplicadas

                    if len(etiquetas_vecinos_unicas) == 1:
                        img_ws[i, j] = etiquetas_vecinos_unicas[0]  # Asigno la etiqueta del vecino
                    else:
                        img_ws[i, j] = 0  # Asigno un 0, un borde

            # Reintroduzco al loop a los pixeles candidatos a etiqueta nueva:
            lista_coor = aux

            # Chequeo variable de control para evitar un loop infinito:
            if len(lista_coor) == cant_anterior:
                seguir = 0

    img_ws = img_ws[1:img_ws.shape[0] - 1, 1:img_ws.shape[1] - 1]  # Corto los bordes extras debido al zeropadding

    return img_ws

def zeropadding(img_np, mask):

    D = mask.shape[0]  # Es cuadrada
    Cant = int((D - 1) / 2)
    M = img_np.shape[0]  # Filas
    N = img_np.shape[1]  # Columnas

    # Qué dimensiones va a tener mi imagen a la salida de de este algoritmo?
    A = M + (D - 1)
    B = N + (D - 1)

    img_pad = np.zeros([A, B])
    img_pad[Cant:A - Cant, Cant:B - Cant] = img_np

    return img_pad

def Etiquetado2(imagen_binarizada, umbral):
    M = imagen_binarizada.shape[0]
    N = imagen_binarizada.shape[1]

    # Zero padding para ahorrar el problema con los bordes
    imagen_pad = np.zeros((M + 2, N + 2))
    imagen_pad[1:M + 1, 1:N + 1] = imagen_binarizada

    # Identificar las coordenadas de los píxeles que son iguales a 1 o 255
    Coordenadas = np.where((imagen_pad == 1) | (imagen_pad == 255))  # Blanco en imágen binaria o 8 bits
    Coordenadasx = Coordenadas[0]
    Coordenadasy = Coordenadas[1]
    etiquetas = np.zeros(imagen_pad.shape)

    # Defino una variable para enumerar las etiquetas y una lista donde guardo las etiquetas usadas
    e = 1
    eti = []

    for i in range(len(Coordenadasx)):

        # Saco las coordenadas de un primer pixel blanco
        x = Coordenadasx[i]
        y = Coordenadasy[i]

        if imagen_pad[x, y] == 1 or imagen_pad[x, y] == 255:
            # Defino una lista para guardar la etiqueta a la que pertenecen los vecinos etiquetados
            vecino_etiquetado = []
            vecino_no_etiquetado = []

            # Recorro la vecindad de 8 y pregunto si alguno de estos pertenece a una etiqueta dentro del diccionario
            for m in range(x - 1, x + 2):
                for n in range(y - 1, y + 2):
                    if (imagen_pad[m, n] == 1 or imagen_pad[m, n] == 255) and ((m, n) != (x, y)):
                        etiqueta_vecino = etiquetas[m, n]

                        if etiqueta_vecino:  # Si pertenece lo agrego a vecino_etiquetado
                            vecino_etiquetado.append(etiqueta_vecino)

            vecino_etiquetado = list(set(vecino_etiquetado))  # Ordeno de menor a mayor y elimino duplicados
            L = len(vecino_etiquetado)

            # Si vecino_etiquetado está vacío creo una nueva etiqueta y aumento en 1 la numeración para la próxima iteración
            if not L:
                etiquetas[x, y] = e
                eti = np.append(eti, e)
                e += 1

            # Si hay un solo vecino etiquetado agrego el pixel a esa etiqueta
            elif L == 1:
                etiquetas[x, y] = vecino_etiquetado[0]

            # Si hay más de uno agrego el pixel actual y todos los pixeles que pertenecen a otras etiquetas a la etiqueta de menor índice.
            # Ej: si tengo un vecino que pertenece a etiqueta 1 y otro a etiqueta 2 meto mi pixel actual y todos los pixeles en etiqueta 2 a etiqueta 1.
            elif L > 1:
                etiquetas[x, y] = vecino_etiquetado[0]

                for z in vecino_etiquetado[1::]:
                    etiquetas[etiquetas == z] = vecino_etiquetado[0]

                eti = [n for n in eti if n not in vecino_etiquetado[1::]]

                eti_escalado = np.arange(1, len(eti) + 1)

                for (m, n) in zip(eti, eti_escalado):
                    etiquetas[etiquetas == m] = n

                e = eti_escalado[-1] + 1
                eti = eti_escalado

    # Creo la imágen a color
    colores_usados = []  # Lista para evitar crear colores repetidos
    Imagen_Color = np.zeros((M, N, 3), np.uint8)

    etiquetas = list(np.ravel(etiquetas))
    etiquetas = [i for i in etiquetas if i != 0]

    # Por cada etiqueta creo un color aleatorio y corroboro que no esté repetido
    for e in eti:
        Repetido = True
        while Repetido:
            R = np.random.randint(0, 256)
            G = np.random.randint(0, 256)
            B = np.random.randint(0, 256)
            if (R, G, B) not in colores_usados:
                Repetido = False

        colores_usados.append((R, G, B))
        x = np.array([i for (i, j) in zip(Coordenadasx, etiquetas) if j == e])
        y = np.array([i for (i, j) in zip(Coordenadasy, etiquetas) if j == e])

        if len(x) > umbral:
            Imagen_Color[x - 1, y - 1, :] = [R, G, B]

    D = {'COORX': Coordenadasx, 'COORY': Coordenadasy, 'Etiquetas': eti, 'COOR_Etiquetas': etiquetas}

    return Imagen_Color, D

def Bounding_Box(imagen_original, imagen_binarizada, DIC, grosor, umbral):
    M = imagen_original.shape[0]
    N = imagen_original.shape[1]

    Imagen_BB = np.zeros((M + grosor * 2, N + grosor * 2, 3), np.uint8)
    Imagen_BB[grosor:M + grosor, grosor:N + grosor, 0] = imagen_original
    Imagen_BB[grosor:M + grosor, grosor:N + grosor, 1] = imagen_original
    Imagen_BB[grosor:M + grosor, grosor:N + grosor, 2] = imagen_original

    Coordenadasx = DIC['COORX']
    Coordenadasy = DIC['COORY']
    Eti = DIC['Etiquetas']
    Etiquetas = DIC['COOR_Etiquetas']

    colores_usados = []

    for e in Eti:
        Repetido = True
        while Repetido:
            R = np.random.randint(0, 256)
            G = np.random.randint(0, 256)
            B = np.random.randint(0, 256)
            if (R, G, B) not in colores_usados:
                Repetido = False
        colores_usados.append((R, G, B))
        x = np.array([i for (i, j) in zip(Coordenadasx, Etiquetas) if j == e])
        y = np.array([i for (i, j) in zip(Coordenadasy, Etiquetas) if j == e])

        if len(x) > umbral:
            Borde_superior = grosor + min(x) - 1
            Borde_inferior = grosor + max(x) - 1
            Borde_izquierdo = grosor + min(y) - 1
            Borde_derecho = grosor + max(y) - 1

            # Bordes = [[Borde_superior]*A + [Borde_inferior]*A + list(range(Borde_superior,Borde_inferior+1)) * 2,
            #        list(range(Borde_izquierdo,Borde_derecho+1)) * 2 + [Borde_izquierdo]*B + [Borde_inferior]*B]

            # Imagen_BB[Borde[0],Bordes[1],:] = [R,G,B]

            # Borde vertical izquierdo
            Imagen_BB[Borde_superior - grosor:Borde_inferior + grosor + 1, Borde_izquierdo - grosor:Borde_izquierdo + 1,
            :] = [R, G, B]
            # Borde vertical derecho
            Imagen_BB[Borde_superior - grosor:Borde_inferior + grosor + 1, Borde_derecho:Borde_derecho + grosor + 1,
            :] = [R, G, B]
            # Borde horizontal superior
            Imagen_BB[Borde_superior - grosor:Borde_superior + 1, Borde_izquierdo:Borde_derecho + 1, :] = [R, G, B]
            # Borde horizontal inferior
            Imagen_BB[Borde_inferior:Borde_inferior + grosor + 1, Borde_izquierdo:Borde_derecho + 1, :] = [R, G, B]

    Imagen_BB = Imagen_BB[grosor:M + grosor, grosor:N + grosor, :]

    return Imagen_BB

def dice(img1,img2):
  img3=img1+img2
  interaccion = (img3 == 2).sum() #cantidad de veces que A y B coinciden en blanco
  unos_1 = (img1 == 1).sum() #cantidad de veces que A es 1
  unos_2 = (img2 == 1).sum() #cantidad de veces que B es 1
  dice = 2*interaccion/(unos_1 + unos_2)
  return dice

def jaccard(img1,img2):
  img3=img1+img2
  interaccion = (img3 == 2).sum() #cantidad de veces que A y B coinciden en blanco
  union = (img3 == 1).sum() + interaccion #cantidad de veces que A y b  son 1
  jaccard = interaccion/union

  return jaccard

def seleccion(imagen):

    result = np.zeros(imagen.shape)

    for i in range(len(imagen)):
        for k in range(len(imagen[i])):

            if (imagen[i][k] == 1 and 340 < i < 460 and 160 < k < 300 ) or (imagen[i][k] > 10):

                result[i][k] = 1

            else:
                result[i][k] = 0

    return result

def bin(im,umbral):
  M = im.shape[0]
  N = im.shape[1]
  res = np.zeros([M,N])

  for i in range(len(im)):
    for j in range(len(im[0])):
      if im[i,j] > umbral:
        res[i,j] = 1
      else:
        res[i,j] = 0

  return res

def m_coocurrencia(imagen, dir):
    # Paso 1: Creamos la matriz que va a ser el output.
    max = np.amax(imagen)
    min = np.amin(imagen)

    imagen_res = np.zeros([max - min + 1, max - min + 1])

    if dir == 0:
        eventos_total = 0
        for i in range(len(imagen)):
            for j in range(len(imagen[0]) - 1):
                imagen_res[imagen[i, j] - min, imagen[i, j + 1] - min] += 1
                eventos_total += 1

    if dir == 45:
        eventos_total = 0
        for i in range(len(imagen)):
            for j in range(len(imagen[0]) - 1):
                imagen_res[imagen[i, j] - min, imagen[i - 1, j + 1] - min] += 1
                eventos_total += 1

    if dir == 90:
        eventos_total = 0
        for i in range(len(imagen)):
            for j in range(len(imagen[0]) - 1):
                imagen_res[imagen[i, j] - min, imagen[i - 1, j] - min] += 1
                eventos_total += 1

    if dir == 135:
        eventos_total = 0
        for i in range(len(imagen)):
            for j in range(len(imagen[0]) - 1):
                imagen_res[imagen[i, j] - min, imagen[i - 1, j - 1] - min] += 1
                eventos_total += 1

    # Paso 3: Nos aseguramos que sea una matriz simétrica sumando su transpuesta
    imagen_res += np.transpose(imagen_res)

    return imagen_res, eventos_total

def medidas_estadisticas(matriz):
  M = matriz.shape[0]
  N = matriz.shape[1]

  media = 0
  for i in range(M):
    for j in range(N):
      media = media + (i*matriz[i,j])

  desv = 0
  for i in range(M):
    for j in range(N):
      desv = desv + ((i-media)**2 * matriz[i,j])

  energia = 0
  for i in range(M):
    for j in range(N):
      energia = energia + (matriz[i,j]**2)

  entropia = 0
  for i in range(M):
    for j in range(N):
      if matriz[i,j] !=0:
        entropia = entropia + (matriz[i,j]*math.log2(matriz[i,j]))


  contraste = 0
  for i in range(M):
    for j in range(N):
      contraste = contraste + ((i-j)**2 * matriz[i,j])


  return media, desv, energia, entropia, contraste

def correlacion(matriz, media, d_stand):
  A = matriz.shape[0]
  B = matriz.shape[1]
  correlacion = 0
  for i in range(A):
    for j in range(B):
      correlacion = correlacion + (((i-media)*(j-media) * matriz[i][j])/ (d_stand*2))
  return correlacion

def IDM(matriz):
  A = matriz.shape[0]
  B = matriz.shape[1]
  IDM = 0
  for i in range(A):
    for j in range(B):
      IDM = IDM + (matriz[i,j])/(1 + (i-j)**2)
  return IDM
