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

#comp = comprimirRLE_ZZ(image)

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print("Tenemos la siguiente matrix\n")
print(matrix)

solution = zig_zag_list(matrix)

print("La solucion:",solution)

# Desde aca opero sobre la solucion
fil, col = matrix.shape
modo = "Zig-Zag"
dtype = type(matrix)
lista_salida = [[fil, col, modo, dtype],[]]

cant = 0
for i in range(len(solution)):# Recorro las listas formadas por el metodo de diagonalizacion

    cant = 0
    value = solution[i][0]# Me paro en el primer valor de la sublista

    print("Me paro en:", solution[i],"Con el valor:",value)

    if len(solution[i]) == 1:# En el caso que la sublista tenga tamaño 1 (ocurre en las puntas de la imagen) me devuelve
                             # una lista asi: [Valor de gris en cuestiion,1 (corresponde con la cantidad)]

        lista_salida[1].append([solution[i][0],1])
        print("Tiene longitud 1")

    else:
        for j in range(len(solution[i])):# Recorro los componentes de la sublista
            gris = solution[i][j]# Valor de gris

            if gris == value:# Si el valor es el mismo se suma en 1 la variable cantidad
                cant += 1
            else:
            #else:# Si son distintos tengo que pararme en ese nuevo valor, y appendear lo que venia sumando antes
                lista_salida[1].append([value, cant])
                value = gris# Me paro en otro valor de gris
                cant = 1 # La cantidad vuelve a ser 1


print("La zz es:",solution)
print("La salida es:",lista_salida)

