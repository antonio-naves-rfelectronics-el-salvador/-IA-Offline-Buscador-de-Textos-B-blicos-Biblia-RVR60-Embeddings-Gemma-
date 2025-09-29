import sys, os, json, sqlite3, numpy as np, torch, shutil, tempfile
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTextEdit, QLineEdit, QListWidget, QComboBox, QSpinBox,
    QSplashScreen, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QTextCursor, QPixmap
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# ---------------- RUTAS ----------------
# Obtener el directorio donde está el ejecutable (.exe) o el script (.py)
if getattr(sys, 'frozen', False):
    # Si es ejecutable, usar la carpeta donde está el .exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Si es script, usar la carpeta del script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# La carpeta "data" debe estar junto al .exe o .py
DATA_DIR = os.path.join(BASE_DIR, "data")

# Rutas a los archivos/carpetas dentro de "data"
DB_PATH = os.path.join(DATA_DIR, "rvr60_embeddings.db")
MODELO_LOCAL = os.path.join(DATA_DIR, "embeddinggemma-300m")
MODELO_LOCAL_CrossEncoder = os.path.join(DATA_DIR, "cross-encoderms-marco-MiniLM-L6-v2")
SPLASH_IMAGE_PATH = os.path.join(DATA_DIR, "portada.png")
ICON_PATH = os.path.join(DATA_DIR, "icon.ico")

# ---------------- FUNCIONES DE CARGA ----------------
def load_vectors_from_db(path=DB_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encuentra la base de datos en la ruta esperada: {path}\n"
            "Asegúrate de que el archivo .db esté dentro de la carpeta 'data'."
        )
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    
    q = """SELECT id, type, verse_ids, book, chapter, verse_start, verse_end, text_concat, emb, dim 
           FROM vectors WHERE type == 'chunk'"""
    cur.execute(q)
    rows = cur.fetchall()
    conn.close()
    
    ids=[]; types=[]; verse_ids_list=[]; book_list=[]; chapter_list=[]; vs_start=[]; vs_end=[]; text_list=[]; embs=[]
    for r in rows:
        ids.append(r[0])
        types.append(r[1])
        verse_ids_list.append(json.loads(r[2]))
        book_list.append(r[3])
        chapter_list.append(r[4])
        vs_start.append(r[5])
        vs_end.append(r[6])
        text_list.append(r[7] or "")
        blob = r[8]; dim = r[9]
        vec = np.frombuffer(blob, dtype=np.float32).reshape((dim,))
        embs.append(vec)
    embs = np.vstack(embs).astype(np.float32)
    return ids, types, verse_ids_list, book_list, chapter_list, vs_start, vs_end, text_list, embs

# ---------------- CARGA DE RECURSOS ----------------
def cargar_recursos():
    """Función para encapsular toda la carga pesada."""
    print("Cargando modelo local de embeddings...")
    global model
    model = SentenceTransformer(MODELO_LOCAL)
    print("✅ Modelo cargado")
    
    print("Cargando CrossEncoder local...")
    global reranker
    reranker = CrossEncoder(MODELO_LOCAL_CrossEncoder)
    print("✅ CrossEncoder cargado")
    
    print("Leyendo DB...")
    global ids, types, verse_ids_list, book_list, chapter_list, verse_start_list, verse_end_list, text_list, embs_db
    ids, types, verse_ids_list, book_list, chapter_list, verse_start_list, verse_end_list, text_list, embs_db = load_vectors_from_db(DB_PATH)
    print("✅ DB cargada:", embs_db.shape)

# ---------------- ORDEN CANÓNICO ----------------
orden_biblia = [
    "Gen","Exod","Lev","Num","Deut","Josh","Judg","Ruth",
    "1Sam","2Sam","1Kgs","2Kgs","1Chr","2Chr","Ezra","Neh","Esth",
    "Job","Ps","Prov","Eccl","Song","Isa","Jer","Lam","Ezek","Dan",
    "Hos","Joel","Amos","Obad","Jon","Mic","Nah","Hab","Zeph","Hag","Zech","Mal",
    "Matt","Mark","Luke","John","Acts","Rom","1Cor","2Cor","Gal","Eph","Phil",
    "Col","1Thess","2Thess","1Tim","2Tim","Titus","Phlm","Heb","Jas","1Pet","2Pet",
    "1John","2John","3John","Jude","Rev"
]

# ---------------- FUNCIONES ----------------
def buscar_en_biblia_por_libro(pregunta, top_k=10):
    pregunta_emb = model.encode_query(pregunta)
    embs_tensor = torch.tensor(embs_db)
    cos_scores = util.cos_sim(torch.tensor(pregunta_emb), embs_tensor)[0]

    resultados_globales = {}
    seen_verse_ids = set()

    for libro in set(book_list):
        idx_libro = [i for i, b in enumerate(book_list) if b == libro]
        if not idx_libro: continue
        scores_libro = cos_scores[idx_libro]
        indices_positivos = (scores_libro > 0.0).nonzero(as_tuple=True)[0]
        if len(indices_positivos) == 0: continue
        k_real = min(top_k, len(indices_positivos))
        top_results = torch.topk(scores_libro[indices_positivos], k=k_real)
        candidatos_idx = [idx_libro[i.item()] for i in indices_positivos[top_results[1]]]

        candidatos_idx = [i for i in candidatos_idx if tuple(verse_ids_list[i]) not in seen_verse_ids]
        for i in candidatos_idx:
            seen_verse_ids.add(tuple(verse_ids_list[i]))

        if not candidatos_idx: continue

        candidatos_textos = [text_list[i] for i in candidatos_idx]
        candidatos_pairs = [(pregunta, txt) for txt in candidatos_textos]
        rerank_scores = reranker.predict(candidatos_pairs)
        ordenados = sorted(zip(rerank_scores, candidatos_idx), key=lambda x: x[0], reverse=True)
        ordenados = [(score, idx) for score, idx in ordenados if score > 0.0]
        if not ordenados: continue
        ordenados = sorted(ordenados, key=lambda x: min(verse_ids_list[x[1]]) if verse_ids_list[x[1]] else 1e9)
        resultados_globales[libro] = ordenados[:k_real]

    return resultados_globales

def get_capitulo(libro, cap):
    resultado = []
    for i, b in enumerate(book_list):
        if b == libro and chapter_list[i] == cap:
            vs = verse_start_list[i]
            ve = verse_end_list[i]
            resultado.append((vs, ve, text_list[i]))
    resultado.sort(key=lambda x: x[0])
    return resultado

# ---------------- HILO ASÍNCRONO ----------------
class BusquedaThread(QThread):
    resultados_ready = pyqtSignal(dict)
    def __init__(self, pregunta, top_k=10):
        super().__init__()
        self.pregunta = pregunta
        self.top_k = top_k
    def run(self):
        resultados = buscar_en_biblia_por_libro(self.pregunta, self.top_k)
        self.resultados_ready.emit(resultados)

# ---------------- INTERFAZ ----------------
class BibliaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biblia Offline - Highlight")
        self.setGeometry(200,100,950,700)
        self.historial = []

        layout = QVBoxLayout()

        # Input
        input_layout = QHBoxLayout()
        self.input_pregunta = QLineEdit()
        self.input_pregunta.setPlaceholderText("Escribe tu pregunta...")
        self.btn_buscar = QPushButton("Buscar")
        self.btn_buscar.clicked.connect(self.iniciar_busqueda)
        input_layout.addWidget(self.input_pregunta)
        input_layout.addWidget(self.btn_buscar)
        layout.addLayout(input_layout)

        # Indicador
        self.lbl_carga = QLabel("")
        layout.addWidget(self.lbl_carga)

        # Resultados
        self.resultados_lista = QListWidget()
        self.resultados_lista.itemClicked.connect(self.goto_resultado)
        layout.addWidget(QLabel("Resultados:"))
        layout.addWidget(self.resultados_lista)

        # Detalle capitulo
        layout.addWidget(QLabel("Capítulo:"))
        self.resultado_text = QTextEdit()
        self.resultado_text.setReadOnly(True)
        layout.addWidget(self.resultado_text)

        # Goto capitulo
        goto_layout = QHBoxLayout()
        self.libro_input = QComboBox()
        self.libro_input.addItems(orden_biblia)
        self.cap_input = QSpinBox()
        self.cap_input.setMinimum(1)
        self.cap_input.setMaximum(150)
        self.btn_ir = QPushButton("Ir a Capítulo")
        self.btn_ir.clicked.connect(self.mostrar_capitulo)
        goto_layout.addWidget(QLabel("Libro:"))
        goto_layout.addWidget(self.libro_input)
        goto_layout.addWidget(QLabel("Capítulo:"))
        goto_layout.addWidget(self.cap_input)
        goto_layout.addWidget(self.btn_ir)
        
        # <<< INICIO MODIFICACIÓN >>>
        # Se añade el botón de Información
        self.btn_info = QPushButton("Info")
        self.btn_info.clicked.connect(self.mostrar_info)
        goto_layout.addWidget(self.btn_info)
        # <<< FIN MODIFICACIÓN >>>

        layout.addLayout(goto_layout)

        # Historial
        layout.addWidget(QLabel("Historial:"))
        self.historial_lista = QListWidget()
        self.historial_lista.itemClicked.connect(self.cargar_historial)
        layout.addWidget(self.historial_lista)

        self.setLayout(layout)

    # <<< INICIO MODIFICACIÓN >>>
    # Nueva función para mostrar la ventana de información
    def mostrar_info(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Información del Software")
        msg_box.setIcon(QMessageBox.Icon.Information)
        
        info_text = """
        <p><b>Autor del Software:</b><br>
        RF Electronics de El Salvador 2025, Antonio Naves.</p>

        <p><b>Condiciones de Uso:</b><br>
        Este software es de uso libre. Se permite su distribución, copia y modificación sin fines de lucro, siempre y cuando se mantenga el reconocimiento a los autores originales. El software se proporciona "tal cual", sin garantías de ningún tipo.</p>
        
        <p><b>Riesgos y Limitaciones de Uso:</b><br>
        Esta es una herramienta de consulta y estudio bíblico basada en inteligencia artificial. Los textos mostrados corresponden con precisión a la versión Reina-Valera 1960.  
        Sin embargo, la relación entre los versos sugeridos no está garantizada, por lo que en ocasiones podría omitirse algún verso al buscar únicamente por palabras aisladas.  
        Cuando se proporciona un contexto más amplio y preciso, los resultados suelen ser más completos y exactos, facilitando la identificación de los pasajes correspondientes.</p>
        """
        
        msg_box.setText(info_text)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    # <<< FIN MODIFICACIÓN >>>

    def iniciar_busqueda(self):
        pregunta = self.input_pregunta.text().strip()
        if not pregunta: return
        self.historial.append(pregunta)
        self.historial_lista.addItem(pregunta)
        self.resultados_lista.clear()
        self.lbl_carga.setText("🔄 Buscando, por favor espere...")
        self.thread = BusquedaThread(pregunta, top_k=10)
        self.thread.resultados_ready.connect(self.mostrar_resultados)
        self.thread.start()

    def mostrar_resultados(self, resultados):
        self.lbl_carga.setText("")
        self.resultados_actuales = resultados
        for libro in orden_biblia:
            if libro in resultados:
                for score, idx in resultados[libro]:
                    texto = text_list[idx][:100].replace("\n"," ")
                    self.resultados_lista.addItem(f"{libro} {chapter_list[idx]}:{verse_start_list[idx]}-{verse_end_list[idx]} ({score:.2f}) - {texto}")

    def goto_resultado(self, item):
        partes = item.text().split(" ")
        libro = partes[0]
        cap = int(partes[1].split(":")[0])
        vs = int(partes[1].split(":")[1].split("-")[0])
        ve = int(partes[1].split(":")[1].split("-")[1])
        
        self.libro_input.setCurrentText(libro)
        self.cap_input.setValue(cap)
        versos = get_capitulo(libro, cap)
        texto_html = f"<b>{libro} {cap}</b><br><br>"
        
        primer_verso_a_buscar = None

        for v_start, v_end, txt in versos:
            if v_start >= vs and v_end <= ve:
                if primer_verso_a_buscar is None:
                    primer_verso_a_buscar = f"{v_start}-{v_end}:"
                
                texto_html += f"<span style='background-color:yellow'>{v_start}-{v_end}: {txt}</span><br>"
            else:
                texto_html += f"{v_start}-{v_end}: {txt}<br>"
                
        self.resultado_text.setHtml(texto_html)

        if primer_verso_a_buscar:
            if self.resultado_text.find(primer_verso_a_buscar):
                # <<< INICIO MODIFICACIÓN >>>
                # Lógica mejorada para centrar el scroll
                self.resultado_text.ensureCursorVisible() # Primero, nos aseguramos de que el cursor sea visible
                cursor_rect = self.resultado_text.cursorRect()
                scrollbar = self.resultado_text.verticalScrollBar()
                
                # Posición Y absoluta del cursor en todo el documento
                absolute_cursor_y = scrollbar.value() + cursor_rect.top()
                
                # Altura de la parte visible del QTextEdit
                viewport_height = self.resultado_text.viewport().height()
                
                # Calculamos el nuevo valor para la barra de scroll para que el cursor quede en el centro
                new_scroll_value = absolute_cursor_y - (viewport_height / 2)
                
                # Establecemos la nueva posición del scroll
                scrollbar.setValue(int(new_scroll_value))
                # <<< FIN MODIFICACIÓN >>>

            else:
                # Fallback por si no se encuentra el texto
                cursor = self.resultado_text.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                self.resultado_text.setTextCursor(cursor)
        else:
            cursor = self.resultado_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.resultado_text.setTextCursor(cursor)

    def mostrar_capitulo(self):
        libro = self.libro_input.currentText()
        cap = self.cap_input.value()
        versos = get_capitulo(libro, cap)
        texto = f"<b>{libro} {cap}</b><br><br>"
        for vs, ve, txt in versos:
            texto += f"{vs}-{ve}: {txt}<br>"
        self.resultado_text.setHtml(texto)

    def cargar_historial(self, item):
        self.input_pregunta.setText(item.text())
        self.iniciar_busqueda()

# ---------------- EJECUCIÓN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Lógica de la pantalla de carga (Splash Screen)
    if os.path.exists(SPLASH_IMAGE_PATH):
        pixmap = QPixmap(SPLASH_IMAGE_PATH)
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents() 
    else:
        print(f"Advertencia: No se encontró la imagen de portada en {SPLASH_IMAGE_PATH}")
        splash = None
    
    # Se realiza la carga pesada de modelos y base de datos
    cargar_recursos()

    # Se crea y muestra la ventana principal
    window = BibliaApp()
    window.show()

    # Se cierra el splash screen cuando la ventana principal está lista
    if splash:
        splash.finish(window)
    
    sys.exit(app.exec())