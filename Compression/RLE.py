import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import SimpleITK as sitk
import math

image = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Compression/Img2.png', 0)


def ComprimirLRE_HV(img, modo):
    # modos: 0 horizontal, 1 vertical

    # Horizontal
    if modo == 0:
        modo = "Horizontal"
        fil, col = img.shape
        dtype = img.dtype
        listaRLE = [[fil, col, modo, dtype],[]]

        value = img[0, 0]
        cant = 0

        for i in range(fil):
            for j in range(col):
                gris = img[i, j]
                if gris == value:
                    cant += 1
                else:

                    listaRLE[1].append([value,cant])

                    value = gris
                    cant = 1

    # Vertical
    if modo == 1:
        modo = "Vertical"
        fil, col = img.shape
        dtype = img.dtype
        listaRLE = [[fil, col, modo, dtype], []]

        value = img[0, 0]
        cant = 0

        for i in range(col):
            for j in range(fil):
                gris = img[i, j]
                if gris == value:
                    cant += 1
                else:

                    listaRLE[1].append([value, cant])

                    value = gris
                    cant = 1
        listaRLE[1].append([value, cant])

    return listaRLE

def comprimirRLE_ZZ(img):

    modo = "Zig-Zag"
    fil, col = img.shape
    dtype = img.dtype
    listaRLE = [[fil, col, modo, dtype], []]

    return listaRLE

def zig_zag_list(matrix):
    fil, col = matrix.shape
    solution = [[] for i in range(fil + col - 1)]

    for i in range(fil):
        for j in range(col):
            sum = i + j
            if (sum % 2 == 0):

                # add at beginning
                solution[sum].insert(0, matrix[i][j])
            else:
                # add at end of the list
                solution[sum].append(matrix[i][j])

    return solution

#comp = comprimirRLE_ZZ(image)

matrix = np.array([
    [1, 2, 5],
    [2, 4, 1],
    [4, 1, 9],
])

print("Tenemos la siguiente matrix\n")
print(matrix)

solution = zig_zag_list(matrix)

print("La solucion:",solution)
# print the solution as it as

# Desde aca opero sobre la solucion
modo = "Zig-Zag"
dtype = type(matrix)
lista_salida = [[fil, col, modo, dtype],[]]

cant = 0
for i in range(len(solution)):
    value = solution[i][0]
    componentes = ""
    print("Tenemos xx",solution[i])
    if len(solution[i]) == 1:
        lista_salida[1].append([solution[i][0],1])
    else:
        for j in range(len(solution[i])):
            print("Tenemos",solution[i])
            print("j:",j)
            gris = solution[i][j]
            if gris == value:
                cant += 1
            else:
                lista_salida[1].append([value, cant])
                value = gris
                cant = 1


print("La salida es:",lista_salida)

