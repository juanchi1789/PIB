import numpy as np
import cv2

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

def encoder_rle(lista):
    i = 0
    contador = 1
    lista_out = []
    while i < len(lista):
        value = lista[i]

        if i == len(lista) - 1 and contador ==1:
            lista_out.append([lista[i], contador])
            break

        if i == len(lista) - 1 and contador > 1:
            lista_out.append([lista[i], contador])
            break

        if value == lista[i + 1] and i < len(lista):
            contador += 1

        if value != lista[i + 1] or i == len(lista):
            lista_out.append([lista[i],contador])
            contador = 1

        i += 1

    return lista_out

def decoder_rle(lista):
    list_decoded = []
    for i in range(len(lista)):
        value = lista[i][0]
        times = lista[i][1]
        k = 0
        while k < times:
            list_decoded.append(value)
            k += 1
    return list_decoded

def RLE_encoder(matrix):

    solution = zig_zag_list(matrix) # Esta es la lista de listas, de las diagonales
    # Desde aca opero sobre la solucion
    fil, col = matrix.shape
    modo = "Zig-Zag"
    dtype = type(matrix)
    lista_salida = [[fil, col, modo, dtype], []]
    print("Teniendo una lista solution",solution)

    for i in range(len(solution)):
        lista_salida[1].append(encoder_rle(solution[i]))

    return lista_salida


# Una matrix simil imagen
matrix = np.array([
    [1, 2, 7],
    [4, 7, 6],
    [7, 8, 9],
])

print("Tenemos:",RLE_encoder(matrix))