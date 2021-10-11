import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import funciones as func



imagen1 = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Segmentacion/img20.jpg', 0)
imagen_resultado = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Segmentacion/img21.jpg', 0)

imagen_bin = func.bin(imagen_resultado,40)


imagen_target1 = imagen1[340:460,160:300]
plt.figure(figsize=(5,5))
plt.imshow(imagen_target1)
plt.title("Imagen 1")
plt.show()

imagen_target2 = imagen1[100:300,160:300]
plt.figure(figsize=(5,5))
plt.imshow(imagen_target2)
plt.title("Imagen 2")
plt.show()

###### ==> Para la primera imagen

matriz_0, ev_0 = func.m_coocurrencia(imagen_target1,0)
matriz_45, ev_45 = func.m_coocurrencia(imagen_target1,45)
matriz_90, ev_90 = func.m_coocurrencia(imagen_target1,90)
matriz_135, ev_135 = func.m_coocurrencia(imagen_target1,135)

matriz_coocurrencia_target = matriz_0 + matriz_45 + matriz_90 + matriz_135 #primero sumo en las 4 direcciones
cantidad_eventos = ev_0 + ev_45 + ev_90 + ev_135

matriz_coocurrencia_1 = matriz_coocurrencia_target/cantidad_eventos #normalizo

media, desv, energia, entropia, contraste = func.medidas_estadisticas(matriz_coocurrencia_1)
corr = func.correlacion(imagen_target1,media,desv)
idm = func.IDM(imagen_target1)

print("Para la primera imagen")
print("La media es:", media)
print("El desvio es:", desv)
print("La Energia es:", energia)
print("La Entropia es:", entropia)
print("El contraste es:", contraste)
print("El IDM es:", idm)
print("La correlacion es:", corr)
print("\n")

###### ==> Para la segunda imagen

matriz_0, ev_0 = func.m_coocurrencia(imagen_target2,0)
matriz_45, ev_45 = func.m_coocurrencia(imagen_target2,45)
matriz_90, ev_90 = func.m_coocurrencia(imagen_target2,90)
matriz_135, ev_135 = func.m_coocurrencia(imagen_target2,135)

matriz_coocurrencia_target = matriz_0 + matriz_45 + matriz_90 + matriz_135 #primero sumo en las 4 direcciones
cantidad_eventos = ev_0 + ev_45 + ev_90 + ev_135

matriz_coocurrencia_2 = matriz_coocurrencia_target/cantidad_eventos #normalizo
media, desv, energia, entropia, contraste = func.medidas_estadisticas(matriz_coocurrencia_2)
corr = func.correlacion(imagen_target2,media,desv)
idm = func.IDM(imagen_target2)

print("Para la segunda imagen")

print("La media es:", media)
print("El desvio es:", desv)
print("La Energia es:", energia)
print("La Entropia es:", entropia)
print("El contraste es:", contraste)
print("El IDM es:", idm)
print("La correlacion es:", corr)