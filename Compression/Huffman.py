import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import SimpleITK as sitk
import math

"""
Huffman Algorithm for image compression
"""

# Load the images:

image = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Compression/Img2.png', 0)

print(image)

vals = image.mean(axis=1).flatten()
# plot histogram with 255 bins
b, bins, patches = plt.hist(image, 255)
plt.xlim([0,255])
plt.show()

# comprimida,dicc,shape = Huffman(img) # Asi tenemos que aplicar el algoritmo
# descomprimida = HuffmanDec(comprimida,dicc,shape) # Asi lo vamos a descomprimir (Es una funcion nueva)

"""
def Huffman(img):
    # Paso 1, tamaño de la imagen
    
    shape_x = img.get_shape[0]
    shape_y = img.get_shape[1]

    # paso 2, Calculo el histograma normalizado
    
    plt.hist(np.ravel(image),bins=256, range=(0,255)), plt.title('Histograma de la imagen')
    plt.show()
    
    # paso 3, Genero dos listas vacias para las frecuencias y para la intensidad correspondiente

    # paso 4, ingreso valores en las listas

    # paso 5,Creo mi arbol y el contador para los nodos
    tree = {}
    k = 0

    # paso 6, While que genera el arbol
    while (len(Lista_histograma) > 1):
        # Ordeno Histograma segun la frecuencia (hist(i)) ordenar de menor a mayor

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

    # paso 10, Recorro la imagen y genero una lista de string con los valores que le corresponden

    # paso 11
    return comprimida, dicc, shape
"""