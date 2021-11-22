from tkinter import *
from tkinter import filedialog
import PIL
from PIL import ImageTk,Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from PIL import ImageTk
import cv2

"""
Cosas de practica:

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
"""

def select_image():
    # Los paneles son las imagenes
    global panelA, panelB

    # La funcion que nos permite seleccionar las imagenes (el path)
    path = filedialog.askopenfilename()

    if len(path) > 0:

        image_inicial = cv2.imread(path)
        image = cv2.resize(image_inicial, (400, 300))

        """
            La funcion que diseñamos va aqui!
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Se convierte la imagen a un formato que pueda ser leido
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)

        contador = 10

        """
            Desde aqui solo se posicionan las imagenes
        """
        imagen_1 = LabelFrame(root, text = "Imagen original")
        imagen_1.pack(side="left", padx=30, pady=30)
        imagen_2 = LabelFrame(root, text="Imagen resultante")
        imagen_2.pack(side="right", padx=30, pady=30)

        calulas_contadas = Label(root, text="Se contaron: " + str(contador) + " Celulas")
        calulas_contadas.pack(side="bottom")

        if panelA is None or panelB is None:
            # Imagen 1
            panelA = Label(imagen_1,image=image)
            panelA.image = image
            panelA.pack()
            # Imagen 2
            panelB = Label(imagen_2,image=edged)
            panelB.image = edged
            panelB.pack()

        else:

            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged



# Desde aca empieza la GUI
root = Tk()
root.title("Trabajo final de PIB")
root.geometry("800x500")

myLabel = Label(root, text="Trabajo final de PIB", font=("Arial", 50)) # Ese titulo lo podemos cambiar
myLabel.pack()

# Inicialmente no hay imagen
panelA = None
panelB = None

# El boton que nos lleva a cargar las imagenes
btn = Button(root, text="Seleccionar una imagen para detectar", command=select_image, font=("Arial", 25))
btn.pack(fill="both", expand="yes", padx="5", pady="5")

# Exit
exit_button = Button(root, text="Exit", command=root.quit)
exit_button.pack(side="bottom",fill="both", expand="yes")

root.mainloop()