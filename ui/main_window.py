import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

# Importamos el Controlador
from controller.simulation_mgr import OFDMSimulationManager

# Configuración inicial de apariencia
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Configuración de la Ventana Principal
        self.title("Simulador LTE OFDM")
        self.geometry("1100x750") # Un poco más alto para el nuevo botón

        # Inicializar el "Gerente"
        self.manager = OFDMSimulationManager()
        
        # Variable para almacenar la ruta de la imagen seleccionada
        self.selected_image_path = None
        
        # Mapeos
        self.bw_map = {"1.4 MHz": 1, "3 MHz": 2, "5 MHz": 3, "10 MHz": 4, "15 MHz": 5, "20 MHz": 6}
        self.cp_map = {"Normal (4.7µs)": 1, "Extendido (16.6µs)": 2}
        self.mod_map = {"QPSK": 1, "16-QAM": 2, "64-QAM": 3}

        # 2. Construcción de la Interfaz
        self.setup_ui()

    def setup_ui(self):
        """Disposición de elementos visuales (Grid Layout)"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- PANEL LATERAL (CONTROLES) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(15, weight=1) # Espaciador al final

        # Título
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="LTE SIMULATOR", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Grupo 1: Física
        self.lbl_phys = ctk.CTkLabel(self.sidebar_frame, text="Parámetros Físicos:", anchor="w")
        self.lbl_phys.grid(row=1, column=0, padx=20, pady=(10, 0))
        
        self.option_bw = ctk.CTkOptionMenu(self.sidebar_frame, values=list(self.bw_map.keys()))
        self.option_bw.grid(row=2, column=0, padx=20, pady=5)
        self.option_bw.set("10 MHz")

        self.option_cp = ctk.CTkOptionMenu(self.sidebar_frame, values=list(self.cp_map.keys()))
        self.option_cp.grid(row=3, column=0, padx=20, pady=5)
        self.option_cp.set("Normal (4.7µs)")

        # Grupo 2: Enlace y Canal
        self.lbl_link = ctk.CTkLabel(self.sidebar_frame, text="Enlace y Canal:", anchor="w")
        self.lbl_link.grid(row=4, column=0, padx=20, pady=(20, 0))

        self.option_mod = ctk.CTkOptionMenu(self.sidebar_frame, values=list(self.mod_map.keys()))
        self.option_mod.grid(row=5, column=0, padx=20, pady=5)
        self.option_mod.set("16-QAM")

        self.lbl_snr = ctk.CTkLabel(self.sidebar_frame, text="SNR: 15 dB")
        self.lbl_snr.grid(row=6, column=0, padx=20, pady=(10,0))
        self.slider_snr = ctk.CTkSlider(self.sidebar_frame, from_=0, to=40, number_of_steps=40, command=self.update_snr_label)
        self.slider_snr.grid(row=7, column=0, padx=20, pady=5)
        self.slider_snr.set(15)

        self.lbl_paths = ctk.CTkLabel(self.sidebar_frame, text="Caminos (Multipath): 1")
        self.lbl_paths.grid(row=8, column=0, padx=20, pady=(10,0))
        self.slider_paths = ctk.CTkSlider(self.sidebar_frame, from_=1, to=6, number_of_steps=5, command=self.update_paths_label)
        self.slider_paths.grid(row=9, column=0, padx=20, pady=5)
        self.slider_paths.set(1)

        self.switch_scfdm = ctk.CTkSwitch(self.sidebar_frame, text="Activar SC-FDM (LTE Uplink)")
        self.switch_scfdm.grid(row=10, column=0, padx=20, pady=(10, 0))

        # Grupo 3: Selección de Archivo (NUEVO)
        self.lbl_source = ctk.CTkLabel(self.sidebar_frame, text="Fuente de Datos:", anchor="w")
        self.lbl_source.grid(row=11, column=0, padx=20, pady=(20, 0))

        self.btn_select_file = ctk.CTkButton(self.sidebar_frame, text="Seleccionar Imagen...", 
                                             fg_color="#4B4B4B", hover_color="#5B5B5B", 
                                             command=self.select_file)
        self.btn_select_file.grid(row=12, column=0, padx=20, pady=5)

        self.lbl_filename = ctk.CTkLabel(self.sidebar_frame, text="[Ningún archivo]", font=("Arial", 11), text_color="gray")
        self.lbl_filename.grid(row=13, column=0, padx=20, pady=0)

        # Grupo 4: Botones de Acción
        self.btn_run_img = ctk.CTkButton(self.sidebar_frame, text="TRANSMITIR IMAGEN", 
                                         fg_color="#1f538d", hover_color="#14375e", # Azul profesional
                                         command=self.action_run_image)
        self.btn_run_img.grid(row=14, column=0, padx=20, pady=(20, 10))

        self.btn_run_ber = ctk.CTkButton(self.sidebar_frame, text="GENERAR CURVA BER", 
                                         fg_color="transparent", border_width=2, 
                                         command=self.action_plot_ber)
        self.btn_run_ber.grid(row=15, column=0, padx=20, pady=10)

        self.btn_run_papr = ctk.CTkButton(self.sidebar_frame, text="ANALIZAR PAPR", 
                                          fg_color="transparent", border_width=2, 
                                          command=self.action_plot_papr)
        self.btn_run_papr.grid(row=16, column=0, padx=20, pady=10)


        # --- PANEL CENTRAL (VISUALIZACIÓN) ---
        self.tabview = ctk.CTkTabview(self, width=800)
        self.tabview.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        
        self.tab_img = self.tabview.add("Visualización Imagen")
        self.tab_ber = self.tabview.add("Análisis BER")
        self.tab_papr = self.tabview.add("Análisis PAPR")

        # Configuración Tab Imagen
        self.tab_img.grid_columnconfigure(0, weight=1)
        self.tab_img.grid_columnconfigure(1, weight=1)
        
        self.lbl_tx_title = ctk.CTkLabel(self.tab_img, text="Imagen Transmitida", font=("Arial", 16, "bold"))
        self.lbl_tx_title.grid(row=0, column=0, pady=10)
        self.lbl_tx_img = ctk.CTkLabel(self.tab_img, text="\n\n[Seleccione una imagen\npara comenzar]", font=("Arial", 14), text_color="gray")
        self.lbl_tx_img.grid(row=1, column=0)

        self.lbl_rx_title = ctk.CTkLabel(self.tab_img, text="Imagen Recibida", font=("Arial", 16, "bold"))
        self.lbl_rx_title.grid(row=0, column=1, pady=10)
        self.lbl_rx_img = ctk.CTkLabel(self.tab_img, text="\n\n[Esperando simulación...]", font=("Arial", 14), text_color="gray")
        self.lbl_rx_img.grid(row=1, column=1)

        self.lbl_status = ctk.CTkLabel(self.tab_img, text="Estado: Esperando configuración", font=("Courier", 14), text_color="yellow")
        self.lbl_status.grid(row=2, column=0, columnspan=2, pady=20)


    # --- EVENTOS Y LÓGICA DE INTERFAZ ---

    def update_snr_label(self, value):
        self.lbl_snr.configure(text=f"SNR: {int(value)} dB")

    def update_paths_label(self, value):
        self.lbl_paths.configure(text=f"Caminos (Multipath): {int(value)}")

    def select_file(self):
        """Abre cuadro de diálogo para seleccionar imagen"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            # Mostrar solo el nombre del archivo para no saturar la UI
            filename = os.path.basename(file_path)
            if len(filename) > 25:
                filename = filename[:22] + "..."
            self.lbl_filename.configure(text=filename, text_color="#30D760") # Verde si hay archivo
            self.lbl_status.configure(text="Imagen cargada. Lista para transmitir.", text_color="white")

    def action_run_image(self):
        """Ejecuta la simulación de imagen"""
        # 0. Validación de entrada
        if not self.selected_image_path:
            messagebox.showwarning("Falta Imagen", "Por favor selecciona una imagen primero usando el botón 'Seleccionar Imagen'.")
            return

        # 1. Recolectar Inputs
        bw_idx = self.bw_map[self.option_bw.get()]
        prof_idx = self.cp_map[self.option_cp.get()]
        mod_idx = self.mod_map[self.option_mod.get()]
        snr = int(self.slider_snr.get())
        paths = int(self.slider_paths.get())
        
        self.lbl_status.configure(text="Procesando OFDM... Espere.")
        self.update() # Forzar refresco de UI

        # 2. Llamar al Controlador
        result = self.manager.run_image_transmission(self.selected_image_path, bw_idx, prof_idx, mod_idx, snr, paths)


        # Leer estado del switch
        is_scfdm = bool(self.switch_scfdm.get()) 
        
        self.lbl_status.configure(text=f"Procesando {'SC-FDM' if is_scfdm else 'OFDM'}... Espere.")
        self.update()

        # Pasar el parámetro is_scfdm al manager
        result = self.manager.run_image_transmission(
            self.selected_image_path, bw_idx, prof_idx, mod_idx, snr, paths, is_scfdm=is_scfdm
        )

        # 3. Mostrar Resultados
        if result["success"]:
            # Convertir Matrices Numpy a Imágenes CTk
            img_tx_pil = Image.fromarray(result["tx_image"]).resize((300, 300), Image.Resampling.NEAREST)
            img_rx_pil = Image.fromarray(result["rx_image"]).resize((300, 300), Image.Resampling.NEAREST)

            tk_img_tx = ctk.CTkImage(light_image=img_tx_pil, dark_image=img_tx_pil, size=(300, 300))
            tk_img_rx = ctk.CTkImage(light_image=img_rx_pil, dark_image=img_rx_pil, size=(300, 300))

            self.lbl_tx_img.configure(image=tk_img_tx, text="")
            self.lbl_rx_img.configure(image=tk_img_rx, text="")
            self.lbl_status.configure(text=result["info"], text_color="#30D760") # Verde SPTF
            self.tabview.set("Visualización Imagen") # Cambiar foco a la pestaña relevante
        else:
            self.lbl_status.configure(text=f"Error: {result.get('error')}", text_color="red")
            messagebox.showerror("Error de Simulación", f"Ocurrió un error al procesar la imagen:\n{result.get('error')}")

    def action_plot_ber(self):
        """Genera y muestra gráfica BER con la IMAGEN"""
        # Validación de seguridad
        if not self.selected_image_path:
            messagebox.showwarning("Falta Imagen", "Selecciona una imagen para analizar su BER.")
            return

        bw_idx = self.bw_map[self.option_bw.get()]
        prof_idx = self.cp_map[self.option_cp.get()]
        mod_idx = self.mod_map[self.option_mod.get()]
        paths = int(self.slider_paths.get())

        self.lbl_status.configure(text="Calculando BER de la imagen... (Espere)")
        self.update()

        try:
            # --- CAMBIO: Pasamos self.selected_image_path ---
            snr_axis, ber_vals = self.manager.calculate_ber_curve(
                self.selected_image_path, bw_idx, prof_idx, mod_idx, paths
            )

            self.embed_plot(self.tab_ber, snr_axis, ber_vals, 
                           "Curva BER vs SNR (Datos de Imagen)", "SNR (dB)", "Bit Error Rate (BER)", log_y=True)
            self.lbl_status.configure(text="Gráfica BER generada.")
            self.tabview.set("Análisis BER")
            
        except Exception as e:
             self.lbl_status.configure(text="Error en cálculo BER")
             messagebox.showerror("Error", str(e))

    def action_plot_papr(self):
        if not self.selected_image_path:
            messagebox.showwarning("Falta Imagen", "Selecciona una imagen para comparar PAPR.")
            return

        bw_idx = self.bw_map[self.option_bw.get()]
        prof_idx = self.cp_map[self.option_cp.get()]
        mod_idx = self.mod_map[self.option_mod.get()]

        self.lbl_status.configure(text="Comparando PAPR (OFDM vs SC-FDM) con imagen...")
        self.update()

        # Llamar a la nueva función comparativa que usa la imagen
        thresholds, ccdf_ofdm, ccdf_scfdm = self.manager.calculate_papr_comparison(
            self.selected_image_path, bw_idx, prof_idx, mod_idx
        )

        # Preparar diccionario para graficar múltiple
        data = {
            "OFDM (Estándar)": ccdf_ofdm,
            "SC-FDM (Precodificado)": ccdf_scfdm
        }

        self.embed_plot(self.tab_papr, thresholds, data, 
                       "Impacto de SC-FDM en PAPR (Datos Reales Scrambleados)", 
                       "Umbral (dB)", "Probabilidad (CCDF)", log_y=True)
        
        self.lbl_status.configure(text="Comparación PAPR generada.")
        self.tabview.set("Análisis PAPR")

    def embed_plot(self, parent_frame, x_data, y_data, title, xlabel, ylabel, log_y=False):
        """Funcion auxiliar para incrustar graficos Matplotlib (Soporta múltiples series)"""
        # Limpiar gráfico anterior
        for widget in parent_frame.winfo_children():
            widget.destroy()

        # Crear Figura
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#2b2b2b') 
        
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        # --- LÓGICA MULTI-LÍNEA ---
        if isinstance(y_data, dict):
            # Si es diccionario, graficamos varias curvas
            colors = ['#1f77b4', '#ff7f0e'] # Azul y Naranja estándar
            for i, (label, y_vals) in enumerate(y_data.items()):
                color = colors[i % len(colors)]
                ax.plot(x_data, y_vals, label=label, linewidth=2, color=color)
            ax.legend() # Mostrar leyenda
        else:
            # Comportamiento antiguo (una sola línea verde)
            ax.plot(x_data, y_data, marker='o', color='#30D760', linewidth=2)
        # --------------------------

        ax.set_title(title, color='white', fontsize=12)
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, color='#444444', linestyle='--')

        if log_y:
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-4, top=1.0) # Límites fijos para ver mejor la "cascada"

        # Canvas de Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)