
"""
Perimetro de la isla
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]

#sns.heatmap(grid, annot=True)
#plt.show()

for fila in range(len(grid)):
    for col in range(len(grid[fila])):
        print(grid[fila][col])