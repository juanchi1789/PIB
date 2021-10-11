# Ejercicio de Mejoramiento
# Dada la imagen de entrada, devolver la salida.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import SimpleITK as sitk
import math

# Cargamos la imagen original (1)
imagen1 = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/imagen.jpg', 0)

# plt.figure(figsize=(5,5))
# plt.imshow(imagen1, cmap="gray",vmin=0, vmax=255)
# plt.title("La imagen Original")
# plt.show()
# print(imagen1.shape)
print(imagen1)

mask = 1/1.5 * np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]))


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

def convolucion(img_np, img_pad, mask):
    # calculamos las dimensiones de todas las imágenes img_np(M,N) img_pad(A,B) mask(D)

    M = img_np.shape[0]
    N = img_np.shape[1]

    img_res = np.zeros([M, N])
    D = mask.shape[0]

    # Hacer ciclo para poder recorrer toda la imagen

    #############################

    for i in range(0, M):
        for j in range(0, N):
            img_aux = img_pad[i:i + D, j:j + D]
            conv = np.sum(np.multiply(img_aux, mask))
            img_res[i, j] = conv
    ###############################

    return img_res

def mediana(img_pad, kernel):
    # tomo como que paso la imagen padeada
    D = len(kernel)
    M = len(img_pad) + 1 - D  # filas
    N = len(img_pad[0]) + 1 - D  # columnas
    img_res = np.zeros([M, N])
    C = int((D - 1) / 2)

    for i in range(C, M):
        for j in range(C, N):
            img_aux = img_pad[i - C:i + D - C, j - C:j + D - C]
            media = np.median(np.multiply(img_aux, kernel))
            img_res[i, j] = media

    return img_res

def escalado(imagen, F1, F2):
    M = imagen.shape[0]  # Filas
    N = imagen.shape[1]  # Columnas
    fmax = 255
    imagen_escal = np.zeros([M, N])

    for i in range(0, M):
        for j in range(0, N):
            if F1 <= imagen[i, j] <= F2:
                imagen_escal[i, j] = ((imagen[i, j] - F1) / (F2 - F1)) * fmax
            else:
                imagen_escal[i, j] = 0
    return imagen_escal

def zerocrossing(imagen, umbral):

    # Parametro
    M = imagen.shape[0]
    N = imagen.shape[1]
    # kernels
    KSH = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])# Sobel Horizontal
    KV = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])# Sobel Vertical

    KO = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])# Laplaciano

    im_pad_H = zeropadding(imagen,KSH)
    im_pad_V = zeropadding(imagen, KV)

    imagenH = convolucion(imagen,im_pad_H,KSH)
    imagenV = convolucion(imagen,im_pad_V,KV)


    # Imagen de salida
    imagen_cross = np.zeros([M,N])

    # Imagen de gradiente
    grad = np.zeros([M,N])

    for i in range(0, M):
        for j in range(0, N):
            grad[i,j] = math.sqrt((imagenH[i,j]**2)+(imagenV[i,j]**2))

    grad_pad = zeropadding(grad,KO)
    deriv_seg = zeropadding(convolucion(grad,grad_pad,KO),KO)

    imagen_out = np.zeros([M,N])

    for i in range(1, M-1):
        for j in range(1, N-1):
            if grad[i,j] > umbral:# and 40<i<80 and 32<j<64:

                # Productos Cruzados

                prod1 = deriv_seg[i - 1, j] * deriv_seg[i + 1, j]
                prod2 = deriv_seg[i, j + 1] * deriv_seg[i,  j - 1]
                prod3 = deriv_seg[i - 1, j + 1] * deriv_seg[i + 1, j - 1]

                if prod1<0 or prod2<0 or prod3<0:
                    imagen_out[i,j] = 255 # Blanco ==> Borde
            else:
                imagen_out[i, j] = 0 # Negro ==> Fondo

    return imagen_out

median = mediana(imagen1, mask)

plt.figure(figsize=(5, 5))
plt.imshow(median, cmap="gray", vmin=0, vmax=255)
plt.title("La imagen con filtro de Mediana")
plt.show()

# Falta el (Recortar)

imagen_escalada = escalado(median, 50, 85)

plt.figure(figsize=(5, 5))
plt.imshow(imagen_escalada, cmap="gray", vmin=0, vmax=255)
plt.title("La imagen Escalada")
plt.show()

image_zc = zerocrossing(imagen_escalada,210)

plt.figure(figsize=(5, 5))
plt.imshow(image_zc, cmap="gray", vmin=0, vmax=255)
plt.title("La imagen Zero crossing")
plt.show()


def recorte(imagen,limx1,limx2,limy1,limy2):

    M = imagen.shape[0]
    N = imagen.shape[1]

    imagen_out = np.zeros([M,N])
    for i in range(0, M):
        for j in range(0, N):
            if limx1 < i < limx2:
                if limy1 < j < limy2:
                    imagen_out[i,j] = imagen[i,j]
                else:

                    imagen_out[i, j] = 0
            else:

                imagen_out[i, j] = 0

    return imagen_out

imagen_recortada = recorte(image_zc,33,70,40,90)

plt.figure(figsize=(5, 5))
plt.imshow(imagen_recortada, cmap="gray", vmin=0, vmax=255)
plt.title("La imagen Zero crossing")
plt.show()

