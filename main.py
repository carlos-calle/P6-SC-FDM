# main.py
import sys
import os

# Asegurarnos de que Python encuentre los módulos internos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow

if __name__ == "__main__":
    # Instalar requerimientos si no están:
    # pip install customtkinter matplotlib opencv-python pillow numpy
    
    app = MainWindow()
    app.mainloop()