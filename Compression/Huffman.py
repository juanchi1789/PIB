import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Huffman Algorithm for image compression
"""

# Load the images:

image = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Compression/Img2.png', 0)


# comprimida,dicc,shape = Huffman(img) # Asi tenemos que aplicar el algoritmo
# descomprimida = HuffmanDec(comprimida,dicc,shape) # Asi lo vamos a descomprimir (Es una funcion nueva)

def findCode(tree, val):
    code, padre = tree[val]
    if padre != 'r':
        code = findCode(tree, padre) + code
    return code


def Huffman(img):
    # Paso 1, tamaño de la imagen
    shape = img.shape

    # paso 2, Calculo el histograma normalizado

    _, vector, _a = plt.hist(np.ravel(image), bins=256, range=(0, 255))

    # paso 3, Genero dos listas vacias para las frecuencias y para la intensidad correspondiente

    Lista_intensidad = []  # Se guardan los valores de intensidad de gris
    Lista_histograma = []  # Se guarda la frecuencia con el nivel de gris

    # paso 4, ingreso valores en las listas => Estaria bueno llenar solo los que tienen valores != 0

    for i in range(len(vector)):
        Lista_intensidad.append(vector[i])
        Lista_intensidad.append(i)

    # paso 5,Creo mi arbol y el contador para los nodos

    tree = {}
    k = 0

    # paso 6, While que genera el arbol
    while len(Lista_histograma) > 1:

        # Ordeno Histograma segun la frecuencia (hist(i)) ordenar de menor a mayor
        Lista_histograma = sorted(Lista_histograma)

        # Selecciono los primeros dos, saco la frecuencia conjunta y los vuelvo a agregar a la lista para que se reordenen en el while
        a1 = Lista_histograma.pop()
        a2 = Lista_histograma.pop()
        fconj = a1[0] + a2[0]
        k += 1
        kstr = str(k)
        nstr = 'n' + kstr
        n = [fconj, nstr]
        Lista_histograma.append(n)

        # Armo el arbol
        tree[a1[1]] = ['0', nstr]
        tree[a2[1]] = ['1', nstr]

    # paso 7, finaliza el árbol
    tree[nstr] = ['', 'r']

    # paso 8, Inicio el diccionario con la codificación
    dicc = {}

    # paso 9, codifico el árbol
    for a in Lista_intensidad:
        code = findCode(tree, a)
        dicc[a] = code

    print(dicc)

    # paso 10, Recorro la imagen y genero una lista de string con los valores que le corresponden

    # Si tenemos esto ya esta el algorithm

    # paso 11

    # return comprimida, dicc, shape


Huffman(image)
