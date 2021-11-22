from tkinter import *
import tkinter
import PIL
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


root = Tk()
# Definir un texto
myLabel = Label(root, text="Hello world")
myLabel.pack()

texto = Entry(root, width=50)  # Casilla para ingresar texto
texto.pack()


# Definir un boton
# Podemos cambiarle el tamaño
# Para que tenga alguna actividad tenes que definir una funcion!
def onClick():
    label = Label(root, text="Hola! " + texto.get())
    label.pack()


# El parametro command define la funcion que tiene asignada
myButton = Button(root, text="Enter your name", padx=50, pady=20, command=onClick)
myButton.pack()

exit_button = Button(root, text="Exit", command=root.quit)
exit_button.pack()

# Para cargar imagenes!


cv_img = cv2.imread('/Users/juanmedina1810/PycharmProjects/PIB/Final Exercise GUI/Images/ALL_1.bmp',0)

photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
#height, width = cv_img.shape
canvas = tkinter.Canvas(root)#, width = width, height = height)
canvas.pack()

# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)


root.mainloop()
