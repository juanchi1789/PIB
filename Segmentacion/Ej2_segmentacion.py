
"""
Parcialito de PIB

Ejercicios:

1) ¿Que metodos de segmentacion vamos a utilizar? (Teoricamete)
2) Se puede segmentar a color?
3) Segmentar por texturas
    3.1) Metricas
4) Indice de Dice and Jaccard
"""
############################################################################################
"""
Esquema para obtener la celula buscada

1) Binarizar
    Puede ser por 
        Otsu
        Umbralizado ?
2) Segmentar
    Puede ser por
        
3) Mejorar la segmentacion
4) ¿Remarcar la celula encontrada?
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import funciones as func


imagen1 = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Segmentacion/img20.jpg', 0)
imagen_resultado = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Segmentacion/img21.jpg', 0)

imagen_bin = func.bin(imagen_resultado,40)

"""
plt.figure(figsize=(5,5))
plt.imshow(imagen_resultado, cmap="gray",vmin=0, vmax=255)
plt.title("La imagen Original")
plt.show()
"""

###################################################################################################

# Otsu ==>

"""
otsu , t = func.Otsu(imagen1)
plt.figure(figsize=(5,5))
plt.imshow(otsu, cmap="gray",vmin=0, vmax=255)
plt.title("La imagen Por Otsu")
plt.show()
"""

# Sin indices

###################################################################################################

# Region Growing

pixel = [400,200]

imagen_rg = func.Region_growing_4_lados(pixel,69,160,imagen1)
plt.figure(figsize=(5,5))
plt.imshow(imagen_rg, cmap="gray",vmin=0, vmax=255)
plt.title("La imagen Por Region Growing")
plt.show()


imagen_rg_select = func.seleccion(imagen_rg)

# Indices


jaccard = func.jaccard(imagen_rg_select,imagen_bin)
dice = func.dice(imagen_rg_select,imagen_bin)
print("Indice Jaccard Region Growing",jaccard)
print("Indice dice Region Growing",dice)


###################################################################################################

# K-means

imagen_k_means = func.Kmeans_de3(imagen1,2)

plt.figure(figsize=(5,5))
plt.imshow(imagen_k_means)
plt.title("La imagen imagen_k_means")
plt.show()

imagen_k_means_segmentada = func.seleccion(imagen_k_means)

plt.figure(figsize=(5,5))
plt.imshow(imagen_k_means_segmentada)
plt.title("La imagen imagen_k_means segmentada")
plt.show()

mask = np.array(([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]))
dilatada = cv2.dilate(imagen_k_means_segmentada,mask,iterations=2)

plt.figure(figsize=(5,5))
plt.imshow(dilatada)
plt.title("La imagen imagen_k_means")
plt.show()

plt.subplot(131), plt.imshow(dilatada), plt.title('Imagen Segmentada')
plt.subplot(132), plt.imshow(imagen_bin), plt.title('Imagen Resultado')

# Indices
jaccard = func.jaccard(dilatada,imagen_bin)
dice = func.dice(dilatada,imagen_bin)
print("Indice Jaccard k-means",jaccard)
print("Indice dice k-means",dice)

###################################################################################################

# watershed

pixel = [400,200]

mask = np.array(([1, 1, 1], [1, 1, 1], [1, 1, 1]))
imagen_orig = imagen1

imagen_pad = func.zeropadding(imagen_orig,mask)

imagen_watershed = func.watershed(imagen_orig,imagen_pad)

imagen_region_grow = func.Region_growing(pixel,0,300,imagen_watershed)

plt.figure(figsize=(5,5))
plt.imshow(imagen_region_grow)
plt.title("La imagen Por Region Growing y watershed")
plt.show()

# El metodo no sirve

###################################################################################################

# Indices ==> van por metodo

###################################################################################################
print("\n")
# Segmentacion por color


imagen_org = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Segmentacion/img20.jpg',1) #con 1 la leermos a color

plt.figure(figsize=(5,5))
plt.imshow(imagen_org)
plt.show()

plt.figure(figsize=(5,5))
plt.subplot(131),plt.imshow(imagen_org[:,:,0]), plt.title('Capa Roja')
plt.subplot(132),plt.imshow(imagen_org[:,:,1]), plt.title('Capa Verde')
plt.subplot(133),plt.imshow(imagen_org[:,:,2]), plt.title('Capa Azul')
plt.show()

imagen_verde = imagen_org[:,:,1]

imagen_k_means = func.Kmeans_de3(imagen_verde,2)

plt.figure(figsize=(5,5))
plt.imshow(imagen_k_means)
plt.title("La imagen imagen_k_means")
plt.show()


imagen_k_means_segmentada = func.seleccion(imagen_k_means)

plt.figure(figsize=(5,5))
plt.imshow(imagen_k_means_segmentada)
plt.title("La imagen imagen_k_means segmentada")
plt.show()

mask = np.array(([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]))
dilatada = cv2.dilate(imagen_k_means_segmentada,mask,iterations=2)

plt.figure(figsize=(5,5))
plt.imshow(dilatada)
plt.title("La imagen imagen_k_means")
plt.show()

jaccard = func.jaccard(dilatada,imagen_bin)
dice = func.dice(dilatada,imagen_bin)
print("Indice Jaccard k-means en color",jaccard)
print("Indice dice k-means en color",dice)

pixel = [400,200]

imagen_color_rg = func.Region_growing(pixel,69,160,imagen_verde)

imagen_color_rg_seleccionada = func.seleccion(imagen_color_rg)

plt.figure(figsize=(5,5))
plt.imshow(imagen_color_rg_seleccionada)
plt.title("La imagen Por Region Growing")
plt.show()

jaccard = func.jaccard(imagen_color_rg_seleccionada,imagen_bin)
dice = func.dice(imagen_color_rg_seleccionada,imagen_bin)
print("Indice Jaccard r-Growing color",jaccard)
print("Indice dice r-Growing color",dice)


###################################################################################################


matriz_0, ev_0 = func.m_coocurrencia(imagen1,0)
matriz_45, ev_45 = func.m_coocurrencia(imagen1,45)
matriz_90, ev_90 = func.m_coocurrencia(imagen1,90)
matriz_135, ev_135 = func.m_coocurrencia(imagen1,135)

matriz_coocurrencia4 = matriz_0+matriz_45+matriz_90+matriz_135 #primero sumo en las 4 direcciones
cantidad_eventos = ev_0 + ev_45 + ev_90 + ev_135

matriz_coocurrencia4 = matriz_coocurrencia4/cantidad_eventos #normalizo una vez sumadas y obtengo la matriz final


#print(matriz_coocurrencia4)