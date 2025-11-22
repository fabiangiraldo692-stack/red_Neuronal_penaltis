import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

# --- Librerías de Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =================================================================
# 1. CARGA, PREPARACIÓN Y ENTRENAMIENTO DEL MODELO
# =================================================================

# --- CARGA DE DATOS DESDE EL ARCHIVO CSV ---
try:
    # Asegúrate de que este archivo esté en la misma carpeta que el script de Python.
    df = pd.read_csv('penaltis.csv')
except FileNotFoundError:
    print("ERROR: Archivo 'penaltis.csv' no encontrado.")
    print("Asegúrate de que el archivo esté en la misma carpeta que el script.")
    exit()

# Renombrar la columna 'Gol (Target)' a 'Gol' para simplificar
df = df.rename(columns={'Gol (Target)': 'Gol'})

# Eliminar espacios en blanco de los nombres de las columnas
df.columns = df.columns.str.strip()

# Eliminar columnas no necesarias para el entrenamiento
df = df.drop(['ID', 'Jugador'], axis=1)

print("Datos cargados exitosamente desde 'penaltis.csv'.")
print(f"Número de registros: {len(df)}")


# --- Preprocesamiento de Entrenamiento ---

# 1. Codificación One-Hot para 'Presion_Partido'
df = pd.get_dummies(df, columns=['Presion_Partido'], prefix='Presion', drop_first=True) 

# 2. Codificación Binaria para 'Pie_Dominante'
df['Pie_Dominante'] = df['Pie_Dominante'].map({'Derecho': 1, 'Izquierdo': 0})

# --- Separación de Características (X) y Objetivo (y) ---
X = df.drop('Gol', axis=1)
y = df['Gol']

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado (Guarda el scaler para usarlo en la GUI)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definición de columnas esperadas (CRUCIAL para la GUI)
columnas_esperadas = X.columns.tolist() 

# --- Construcción y Entrenamiento del Modelo ---
input_dim = X_train_scaled.shape[1] 
model = Sequential()
model.add(Dense(units=16, activation='relu', input_shape=(input_dim,)))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Entrenando Red Neuronal...")
model.fit(X_train_scaled, y_train, epochs=100, batch_size=4, verbose=0)
print("Entrenamiento completado. Abriendo Interfaz Gráfica.")


# ------------------------------------------------------------------
# FUNCIÓN DE PREDICCIÓN LLAMADA POR LA GUI
# ------------------------------------------------------------------

def hacer_prediccion():
    """Captura los datos de la GUI, los preprocesa y predice."""
    try:
        # 1. CAPTURAR ENTRADAS DEL USUARIO
        vel = float(entry_vel.get())
        ang = float(entry_ang.get())
        dist = float(entry_dist.get())
        pie = pie_var.get() 
        presion = presion_var.get()

        # 2. PREPROCESAMIENTO DE LAS ENTRADAS DEL USUARIO
        
        # Crear DataFrame con las entradas
        datos_entrada = pd.DataFrame([[vel, ang, dist, presion, pie]], 
                                     columns=['Velocidad_kmh', 'Angulo_grados', 'Distancia_Portero_m', 'Presion_Partido', 'Pie_Dominante'])

        # Aplicar el MISMO One-Hot Encoding
        datos_entrada = pd.get_dummies(datos_entrada, columns=['Presion_Partido'], prefix='Presion', drop_first=True)
        datos_entrada['Pie_Dominante'] = datos_entrada['Pie_Dominante'].map({'Derecho': 1, 'Izquierdo': 0})
        
        # 3. CONSOLIDAR Y ORDENAR COLUMNAS PARA EL MODELO
        final_data = {}
        for col in columnas_esperadas:
            if col in datos_entrada.columns:
                final_data[col] = datos_entrada[col].iloc[0]
            else:
                final_data[col] = 0
        
        # Crear DataFrame final con el orden correcto
        X_nuevo = pd.DataFrame([final_data])[columnas_esperadas]

        # Aplicar el MISMO escalado
        X_nuevo_escalado = scaler.transform(X_nuevo)

        # 4. PREDICCIÓN
        probabilidad = model.predict(X_nuevo_escalado, verbose=0)[0][0] * 100
        
        # 5. MOSTRAR RESULTADO EN LA GUI
        resultado_label.config(text=f"Probabilidad de Gol: {probabilidad:.2f}%")
        
        # Mostrar el resultado con un color indicativo
        if probabilidad >= 50:
            resultado_label.config(fg="green")
        else:
            resultado_label.config(fg="red")

    except ValueError:
        messagebox.showerror("Error de Entrada", "Por favor, ingrese valores numéricos válidos para la Velocidad, Ángulo y Distancia.")


# ------------------------------------------------------------------
# CONFIGURACIÓN DE LA GUI CON TKINTER
# ------------------------------------------------------------------

# Definir el color gris y el estilo de fuente
COLOR_FONDO = "#e0e0e0" # Gris claro
FUENTE_GENERAL = ('Helvetica', 12)
FUENTE_RESULTADO = ('Helvetica', 18, 'bold')


root = tk.Tk()
root.title("⚽ Predictor de Penaltis con Red Neuronal")
root.resizable(False, False)

# Dimensiones de la ventana
ANCHO_VENTANA = 400
ALTO_VENTANA = 550

# 🚀 LÓGICA PARA CENTRAR LA VENTANA 🚀
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calcular coordenadas X e Y para el centrado
pos_x = (screen_width // 2) - (ANCHO_VENTANA // 2)
pos_y = (screen_height // 2) - (ALTO_VENTANA // 2)

# Establecer tamaño y posición (Centrado)
root.geometry(f'{ANCHO_VENTANA}x{ALTO_VENTANA}+{pos_x}+{pos_y}')
root.configure(bg=COLOR_FONDO)

# Variables de control
pie_var = tk.StringVar(root)
pie_var.set("Derecho") 
presion_var = tk.StringVar(root)
presion_var.set("Baja")

# --- Frame principal ---
main_frame = tk.Frame(root, bg=COLOR_FONDO)
# El padding se aplica en el grid
main_frame.grid(row=0, column=0, padx=30, pady=30, sticky=(tk.W, tk.E, tk.N, tk.S))


# Permitir que la ventana principal se expanda con el frame
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)


# --- Creación de Widgets de Entrada ---

# 1. Entrada de Velocidad
tk.Label(main_frame, text="Velocidad (km/h):", font=FUENTE_GENERAL, bg=COLOR_FONDO).grid(row=0, column=0, padx=5, pady=10, sticky='w')
entry_vel = ttk.Entry(main_frame, font=FUENTE_GENERAL)
entry_vel.grid(row=0, column=1, padx=5, pady=10, sticky='ew')
entry_vel.insert(0, "95") 

# 2. Entrada de Ángulo
tk.Label(main_frame, text="Ángulo (grados):", font=FUENTE_GENERAL, bg=COLOR_FONDO).grid(row=1, column=0, padx=5, pady=10, sticky='w')
entry_ang = ttk.Entry(main_frame, font=FUENTE_GENERAL)
entry_ang.grid(row=1, column=1, padx=5, pady=10, sticky='ew')
entry_ang.insert(0, "30") 

# 3. Entrada de Distancia al Portero
tk.Label(main_frame, text="Distancia Portero (m):", font=FUENTE_GENERAL, bg=COLOR_FONDO).grid(row=2, column=0, padx=5, pady=10, sticky='w')
entry_dist = ttk.Entry(main_frame, font=FUENTE_GENERAL)
entry_dist.grid(row=2, column=1, padx=5, pady=10, sticky='ew')
entry_dist.insert(0, "0.9") 

# 4. Selección de Pie Dominante
tk.Label(main_frame, text="Pie Dominante:", font=FUENTE_GENERAL, bg=COLOR_FONDO).grid(row=3, column=0, padx=5, pady=10, sticky='w')
ttk.OptionMenu(main_frame, pie_var, 'Derecho', 'Derecho', 'Izquierdo').grid(row=3, column=1, padx=5, pady=10, sticky='ew')

# 5. Selección de Presión
tk.Label(main_frame, text="Presión del Partido:", font=FUENTE_GENERAL, bg=COLOR_FONDO).grid(row=4, column=0, padx=5, pady=10, sticky='w')
ttk.OptionMenu(main_frame, presion_var, 'Baja', 'Baja', 'Media', 'Alta').grid(row=4, column=1, padx=5, pady=10, sticky='ew')

# --- Botón de Predicción ---
predict_button = ttk.Button(main_frame, text="⚽ PREDECIR GOL", command=hacer_prediccion)
predict_button.grid(row=5, column=0, columnspan=2, pady=30, sticky='ew')

# --- Etiqueta de Resultado ---
resultado_label = tk.Label(main_frame, 
                            text="Esperando datos...", 
                            font=FUENTE_RESULTADO,
                            bg=COLOR_FONDO, # Fondo gris para la etiqueta
                            pady=10)
resultado_label.grid(row=6, column=0, columnspan=2, pady=10)

# Iniciar el bucle principal de la GUI
root.mainloop()