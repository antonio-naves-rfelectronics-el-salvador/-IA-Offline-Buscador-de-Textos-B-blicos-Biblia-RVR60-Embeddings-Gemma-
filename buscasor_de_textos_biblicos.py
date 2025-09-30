# búsquedor_de_textos_biblicos_async.py
# Versión final corregida:
# - La lista de resultados se limpia al iniciar una nueva búsqueda.
# - Se corrigió la generación de HTML para las estrellas según lo solicitado.
# - Se implementó un método de centrado de versos robusto y funcional usando anclas HTML.
# - Se mantiene la solución al lag durante la búsqueda léxica.

import sys, os, json, sqlite3, numpy as np, re, gc, math, html, unicodedata
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QLineEdit, QListWidget, QComboBox, QSpinBox,
    QSplashScreen, QMessageBox, QListWidgetItem, QProgressBar, QFrame, QCheckBox,
    QDialog, QGridLayout, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QTextCursor, QPixmap, QFont, QIcon, QTextDocument

# --- CONFIGURACIÓN DE RUTAS Y RECURSOS ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "rvr60_embeddings.db")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
MODELO_LOCAL = os.path.join(DATA_DIR, "embeddinggemma-300m")
MODELO_LOCAL_CrossEncoder = os.path.join(DATA_DIR, "cross-encoderms-marco-MiniLM-L6-v2")
SPLASH_IMAGE_PATH = os.path.join(DATA_DIR, "portada.png")
ICO_IMAGE_PATH = os.path.join(DATA_DIR, "icon.ico")
PNG_IMAGE_PATH = os.path.join(DATA_DIR, "icon.png")

# --- DICCIONARIOS Y CONSTANTES ---
book_code_to_spanish = {
    "Gen":"Génesis","Exod":"Éxodo","Lev":"Levítico","Num":"Números","Deut":"Deuteronomio","Josh":"Josué","Judg":"Jueces","Ruth":"Rut",
    "1Sam":"1 Samuel","2Sam":"2 Samuel","1Kgs":"1 Reyes","2Kgs":"2 Reyes","1Chr":"1 Crónicas","2Chr":"2 Crónicas","Ezra":"Esdras","Neh":"Nehemías","Esth":"Ester",
    "Job":"Job","Ps":"Salmos","Prov":"Proverbios","Eccl":"Eclesiastés","Song":"Cantares","Isa":"Isaías","Jer":"Jeremías","Lam":"Lamentaciones","Ezek":"Ezequiel","Dan":"Daniel",
    "Hos":"Oseas","Joel":"Joel","Amos":"Amós","Obad":"Abdías","Jon":"Jonás","Mic":"Miqueas","Nah":"Nahúm","Hab":"Habacuc","Zeph":"Sofonías","Hag":"Hageo","Zech":"Zacarías","Mal":"Malaquías",
    "Matt":"Mateo","Mark":"Marcos","Luke":"Lucas","John":"Juan","Acts":"Hechos","Rom":"Romanos","1Cor":"1 Corintios","2Cor":"2 Corintios","Gal":"Gálatas","Eph":"Efesios","Phil":"Filipenses",
    "Col":"Colosenses","1Thess":"1 Tesalonicenses","2Thess":"2 Tesalonicenses","1Tim":"1 Timoteo","2Tim":"2 Timoteo","Titus":"Tito","Phlm":"Filemón","Heb":"Hebreos","Jas":"Santiago","1Pet":"1 Pedro","2Pet":"2 Pedro",
    "1John":"1 Juan","2John":"2 Juan","3John":"3 Juan","Jude":"Judas","Rev":"Apocalipsis"
}
spanish_to_book_code = {v:k for k,v in book_code_to_spanish.items()}
orden_biblia = list(book_code_to_spanish.keys())
try:
    idx_nt = orden_biblia.index("Matt")
except ValueError:
    idx_nt = len(orden_biblia)//2
ANTIGUO_CODES = orden_biblia[:idx_nt]
NUEVO_CODES = orden_biblia[idx_nt:]

# --- FUNCIONES DE UTILIDAD ---
def sanitize_text_for_display(text: str) -> str:
    if not text: return ""
    s = re.sub(r'(?i)type\s*:\s*chunk|\[/?chunk\]|</?chunk>|^chunk\s*[:\-]\s*', '', text)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def split_verses_from_chunk(text: str, start: int, end: int) -> dict:
    verses = {}
    if not text: return verses
    matches = list(re.finditer(r'(\d{1,3})\s*[:\-\.)]\s*', text))
    if matches and len(matches)>=1:
        for i,m in enumerate(matches):
            try: num = int(m.group(1))
            except: continue
            if num < start or num > end: continue
            start_pos = m.end()
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(text)
            verses[num] = text[start_pos:end_pos].strip()
    if not verses:
        for v in range(start, end+1): verses[v] = text.strip()
    return verses

def load_vectors_from_db(path=DB_PATH):
    if not os.path.exists(path): raise FileNotFoundError(f"No se encuentra la base de datos en: {path}")
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, type, verse_ids, book, chapter, verse_start, verse_end, text_concat, emb, dim FROM vectors")
    chunk_rows = cur.fetchall()
    cur.execute("SELECT id, type, verse_ids, book, chapter, verse_start, verse_end, text_concat FROM vectors WHERE type = 'verse'")
    verse_rows = cur.fetchall()
    conn.close()
    ids=[]; types=[]; verse_ids_list=[]; book_list=[]; chapter_list=[]
    vs_start=[]; vs_end=[]; text_list=[]; embs=[]
    for r in chunk_rows:
        ids.append(r[0]); types.append(r[1])
        try: verse_ids_list.append(json.loads(r[2]))
        except: verse_ids_list.append([])
        book_list.append(r[3]); chapter_list.append(r[4])
        vs_start.append(r[5]); vs_end.append(r[6]); text_list.append(r[7] or "")
        vec = np.frombuffer(r[8], dtype=np.float32).reshape((r[9],))
        embs.append(vec)
    embs = np.vstack(embs).astype(np.float32)
    verse_map = {}
    for r in verse_rows:
        try: verse_map[(r[3], int(r[4]), int(r[5]))] = sanitize_text_for_display(r[7] or "")
        except: continue
    for i, txt in enumerate(text_list):
        try: b=book_list[i]; ch=int(chapter_list[i]); vs=int(vs_start[i]); ve=int(vs_end[i])
        except: continue
        partes = split_verses_from_chunk(txt, vs, ve)
        for vnum, vtext in partes.items():
            if (b,ch,vnum) not in verse_map: verse_map[(b,ch,vnum)] = sanitize_text_for_display(vtext)
    return ids, types, verse_ids_list, book_list, chapter_list, vs_start, vs_end, text_list, embs, verse_map

def cargar_recursos(progress_callback=None):
    global SentenceTransformer, util, CrossEncoder, torch, model, reranker, ids, types, verse_ids_list, book_list, chapter_list, verse_start_list, verse_end_list, text_list, embs_db, embs_tensor, verse_map
    from sentence_transformers import SentenceTransformer, util, CrossEncoder
    import torch
    print("Cargando modelos y BD (puede tardar)...")
    model = SentenceTransformer(MODELO_LOCAL)
    reranker = CrossEncoder(MODELO_LOCAL_CrossEncoder)
    ids, types, verse_ids_list, book_list, chapter_list, verse_start_list, verse_end_list, text_list, embs_db, verse_map = load_vectors_from_db(DB_PATH)
    embs_tensor = torch.from_numpy(embs_db).to(torch.float32)
    globals()['embs_tensor'] = embs_tensor
    if progress_callback: progress_callback(100)

def highlight_text(original_text: str, query: str, ignore_accents: bool=False, ignore_case: bool=True, exact_phrase: bool=False, **kwargs):
    if not query or not original_text: return html.escape(original_text or "")
    try:
        # Create a searchable version of the text based on options
        text_for_search = original_text
        if ignore_accents:
            text_for_search = ''.join(c for c in unicodedata.normalize('NFD', text_for_search) if unicodedata.category(c) != 'Mn')
        if ignore_case:
            text_for_search = text_for_search.casefold()

        # Create a searchable version of the query based on options
        query_for_search = query
        if ignore_accents:
            query_for_search = ''.join(c for c in unicodedata.normalize('NFD', query_for_search) if unicodedata.category(c) != 'Mn')
        if ignore_case:
            query_for_search = query_for_search.casefold()

        # Find all occurrences
        spans = []
        words_to_find = [query_for_search] if exact_phrase else [w for w in re.split(r'\W+', query_for_search, flags=re.UNICODE) if w]
        for word in words_to_find:
            start_index = 0
            while start_index < len(text_for_search):
                pos = text_for_search.find(word, start_index)
                if pos == -1:
                    break
                spans.append((pos, pos + len(word)))
                start_index = pos + 1

        if not spans: return html.escape(original_text)

        # Merge overlapping spans
        spans.sort()
        merged = []
        if spans:
            cur_s, cur_e = spans[0]
            for s, e in spans[1:]:
                if s < cur_e: cur_e = max(cur_e, e)
                else: merged.append((cur_s, cur_e)); cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))

        # Build the final HTML
        out, last = [], 0
        for s, e in merged:
            out.append(html.escape(original_text[last:s]))
            out.append("<mark style='background:yellow;color:black'>")
            out.append(html.escape(original_text[s:e]))
            out.append("</mark>")
            last = e
        out.append(html.escape(original_text[last:]))
        return ''.join(out)
    except Exception:
        return html.escape(original_text)

def get_capitulo(libro_code, cap):
    verse_map_local = globals().get('verse_map', {})
    versos=[]
    keys = [k for k in verse_map_local.keys() if k[0]==libro_code and k[1]==cap]
    verse_nums = sorted({k[2] for k in keys})
    for v in verse_nums: versos.append((v, v, verse_map_local.get((libro_code, cap, v), "")))
    return versos

def buscar_semantica(pregunta, top_k=10, progress_callback=None, **kwargs):
    with torch.inference_mode():
        pregunta_emb = model.encode_query(pregunta)
        query_tensor = torch.tensor(pregunta_emb, dtype=torch.float32)
        embs_t = globals().get('embs_tensor')
        if embs_t is None: raise RuntimeError("embs_tensor no disponible")
        cos_scores = util.cos_sim(query_tensor, embs_t)[0].cpu().numpy()
    gc.collect()
    resultados_globales, seen_verse_ids, book_array = {}, set(), np.array(book_list)
    unique_books = np.unique(book_array)
    for i, libro in enumerate(unique_books):
        if progress_callback: progress_callback(int((i / len(unique_books)) * 100))
        idx_libro = np.where(book_array == libro)[0]
        if idx_libro.size==0: continue
        scores_libro = cos_scores[idx_libro]
        indices_positivos_local = np.where(scores_libro > 0.0)[0]
        if indices_positivos_local.size==0: continue
        k_real = min(top_k, indices_positivos_local.size)
        top_pos_idx = indices_positivos_local[np.argsort(-scores_libro[indices_positivos_local])[:k_real]]
        candidatos_idx = [int(idx_libro[i]) for i in top_pos_idx if tuple(verse_ids_list[int(idx_libro[i])]) not in seen_verse_ids]
        for idx in candidatos_idx: seen_verse_ids.add(tuple(verse_ids_list[idx]))
        if not candidatos_idx: continue
        candidatos_textos = [text_list[i] for i in candidatos_idx]
        with torch.inference_mode():
            rerank_scores = reranker.predict([(pregunta, txt) for txt in candidatos_textos])
        ordenados = sorted([(float(s), int(i)) for s, i in zip(rerank_scores, candidatos_idx) if float(s)>0.0], key=lambda x:x[0], reverse=True)
        if not ordenados: continue
        resultados_globales[libro] = sorted(ordenados, key=lambda x: min(verse_ids_list[x[1]]) if verse_ids_list[x[1]] else 1e9)[:k_real]
    if progress_callback: progress_callback(100)
    return resultados_globales

def buscar_lexica(pregunta, top_k=10, progress_callback=None, options=None):
    options = options or {}
    ignore_accents = bool(options.get('ignore_accents', False))
    ignore_case = bool(options.get('ignore_case', True))
    exact_phrase = bool(options.get('exact_phrase', False))
    match_all = bool(options.get('match_all', False))
    qp = pregunta.strip()
    if not qp: return {}

    verse_map_local = globals().get('verse_map', {})
    matches = {}
    total = len(verse_map_local)
    processed = 0

    q_cmp = qp
    if ignore_accents: q_cmp = ''.join(c for c in unicodedata.normalize('NFD', q_cmp) if unicodedata.category(c) != 'Mn')
    if ignore_case: q_cmp = q_cmp.casefold()

    qwords_cmp = [re.escape(w) for w in re.split(r'\W+', q_cmp, flags=re.UNICODE) if w] if not exact_phrase else [re.escape(q_cmp)]
    if not qwords_cmp: return {}
    
    for (b,ch,v), verse_text in verse_map_local.items():
        processed += 1
        if progress_callback and processed % 200 == 0:
            progress_callback(int((processed/total)*100))

        text_cmp = verse_text
        if ignore_accents: text_cmp = ''.join(c for c in unicodedata.normalize('NFD', text_cmp) if unicodedata.category(c) != 'Mn')
        if ignore_case: text_cmp = text_cmp.casefold()
        
        score, matched_count = 0.0, 0
        for word in qwords_cmp:
            if re.search(word, text_cmp):
                score += len(word)
                matched_count += 1

        if (match_all and matched_count < len(qwords_cmp)) or score == 0:
            continue
            
        found_idx = next((i for i, book in enumerate(book_list) if book==b and int(chapter_list[i])==ch and int(verse_start_list[i])<=v<=int(verse_end_list[i])), None)
        if found_idx is not None:
            matches.setdefault(b, []).append((score, found_idx))

    resultados = {}
    for b, lst in matches.items():
        agg = {}
        for sc, idx in lst: agg[idx] = agg.get(idx, 0.0) + sc
        resultados[b] = sorted([(s, i) for i,s in agg.items()], key=lambda x:x[0], reverse=True)[:top_k]
    
    if progress_callback: progress_callback(100)
    return resultados

# --- HILOS (THREADS) ---
class ResultsBuildThread(QThread):
    finished_build = pyqtSignal(list)
    def __init__(self, latest_resultados, orden_biblia_local, book_code_to_spanish_local, book_list_local,
                 chapter_list_local, verse_start_list_local, verse_end_list_local, text_list_local,
                 current_filters_books=None, current_star_filter=None, current_query="", lex_options=None):
        super().__init__()
        self.latest_resultados = latest_resultados or {}
        self.orden_biblia_local = orden_biblia_local
        self.book_code_to_spanish_local, self.book_list_local, self.chapter_list_local = book_code_to_spanish_local, book_list_local, chapter_list_local
        self.verse_start_list_local, self.verse_end_list_local, self.text_list_local = verse_start_list_local, verse_end_list_local, text_list_local
        self.current_filters_books = set(current_filters_books) if current_filters_books else None
        self.current_star_filter = current_star_filter
        self.current_query, self.lex_options = current_query or "", lex_options or {}

    def run(self):
        candidates = []
        for libro_code in self.orden_biblia_local:
            if self.isInterruptionRequested(): return
            if (libro_code not in self.latest_resultados) or \
               (self.current_filters_books and libro_code not in self.current_filters_books):
                continue
            for score, idx in self.latest_resultados[libro_code]:
                vs, ve = self.verse_start_list_local[idx], self.verse_end_list_local[idx]
                texto_muestra = sanitize_text_for_display(self.text_list_local[idx])[:300].replace("\n"," ")
                candidates.append({'book': libro_code, 'score': float(score), 'idx': idx, 'vs':vs, 've':ve, 'texto_muestra': texto_muestra})
        if not candidates: self.finished_build.emit([]); return

        scores = [c['score'] for c in candidates]
        min_s, max_s = min(scores), max(scores)
        for c in candidates:
            s = c['score']
            norm = (s - min_s) / (max_s - min_s) if max_s > min_s else 1.0
            c['stars'] = max(1, min(3, 1 + int(round(norm * 2))))
        
        if self.current_star_filter is not None:
            candidates = [c for c in candidates if c['stars'] == int(self.current_star_filter)]
        
        candidates.sort(key=lambda x: (-x['score'], x['idx']))
        
        out_list = []
        for c in candidates:
            if self.isInterruptionRequested(): return
            spanish_book = self.book_code_to_spanish_local.get(c['book'], c['book'])
            cap, vs, ve = int(self.chapter_list_local[c['idx']]), c['vs'], c['ve']
            texto_muestra_escaped = html.escape(c['texto_muestra'])
            filled = c['stars']
            if vs == ve:
                # Si el verso de inicio es igual al verso de fin (ej: 1:1-1), mostrar solo 1:1
                ref_verso = f"{cap}:{vs}"
            else:
                # Si es un rango (ej: 1:1-3), mostrar el rango completo
                ref_verso = f"{cap}:{vs}-{ve}"
            
            filled_stars_html = ''.join(["<span style='color:gold;font-weight:700'>★</span>" for _ in range(filled)])
            empty_stars_html = ''.join(["<span style='color:lightgray'>☆</span>" for _ in range(3 - filled)])
            stars_html = f"{filled_stars_html}{empty_stars_html}"

            # Usar la nueva variable ref_verso
            main_html = f"<b style='margin-left:6px'>{spanish_book} {ref_verso}</b> <small>({c['score']:.2f})</small><br><span style='color:#333'>{texto_muestra_escaped}</span>"
            
            widget_html = f"<div style='display:flex; gap:8px; align-items:center'>{stars_html}{main_html}</div>"
            out_list.append({'idx': c['idx'], 'score': c['score'], 'book': c['book'], 'vs': vs, 've': ve, 'html': widget_html})
    
            ## <--- MODIFICACIÓN: Uso del código exacto solicitado por el usuario para las estrellas
            #filled_stars_html = ''.join(["<span style='color:gold;font-weight:700'>★</span>" for _ in range(filled)])
            #empty_stars_html = ''.join(["<span style='color:lightgray'>☆</span>" for _ in range(3 - filled)])
            #stars_html = f"{filled_stars_html}{empty_stars_html}"
#
            #main_html = f"<b style='margin-left:6px'>{spanish_book} {cap}:{vs}-{ve}</b> <small>({c['score']:.2f})</small><br><span style='color:#333'>{texto_muestra_escaped}</span>"
            #widget_html = f"<div style='display:flex; gap:8px; align-items:center'>{stars_html}{main_html}</div>"
            #out_list.append({'idx': c['idx'], 'score': c['score'], 'book': c['book'], 'vs': vs, 've': ve, 'html': widget_html})
        self.finished_build.emit(out_list)

class ChapterBuildThread(QThread):
    finished_build = pyqtSignal(str, str, int)
    def __init__(self, libro_code, cap_num, current_query, lex_options, selected_vs=None, selected_ve=None):
        super().__init__()
        self.libro_code, self.cap_num = libro_code, cap_num
        self.current_query, self.lex_options = current_query, lex_options
        self.selected_vs, self.selected_ve = selected_vs, selected_ve

    def run(self):
        versos = get_capitulo(self.libro_code, self.cap_num)
        titulo = book_code_to_spanish.get(self.libro_code, self.libro_code)
        texto_html = f"<div style='font-family: Segoe UI, sans-serif; font-size: 11pt;'><b>{html.escape(titulo)} {self.cap_num}</b><br><br>"
        primer_verso_num = None
        for vs, ve, txt in versos:
            if self.isInterruptionRequested(): return
            if primer_verso_num is None: primer_verso_num = vs
            #highlighted_words = highlight_text(txt, self.current_query, **(self.lex_options or {}))
            # Determinar si el verso actual (vs) está dentro del rango seleccionado por el usuario.
            is_part_of_selection = self.selected_vs is not None and self.selected_ve is not None and \
                                   self.selected_vs <= vs <= self.selected_ve

            if is_part_of_selection:
                # Si el verso está en la selección, aplicar resaltado amarillo de la búsqueda.
                highlighted_words = highlight_text(txt, self.current_query, **(self.lex_options or {}))
            else:
                # Si no, solo mostrar el texto plano (escapado para HTML para evitar errores).
                highlighted_words = html.escape(txt)
            
            line_html = f'<a name="verse_{vs}"></a><b>{vs}</b>. {highlighted_words}'
            # <--- MODIFICACIÓN: Añadir un ancla HTML para el centrado fiable
            line_html = f'<a name="verse_{vs}"></a><b>{vs}</b>. {highlighted_words}'
            is_selected = self.selected_vs is not None and self.selected_ve is not None and self.selected_vs <= vs <= self.selected_ve
            style = "background-color: #E6F2FF; border-radius: 4px; padding: 2px 5px; margin: 1px 0;" if is_selected else "padding: 2px 5px; margin: 1px 0;"
            texto_html += f"<div style='{style}'>{line_html}</div>"
        texto_html += "</div>"
        reference = f"{titulo} {self.cap_num}:{primer_verso_num}" if primer_verso_num is not None else f"{titulo} {self.cap_num}"
        verse_to_center = self.selected_vs if self.selected_vs is not None else -1
        self.finished_build.emit(texto_html, reference, verse_to_center)

class BusquedaThread(QThread):
    resultados_ready = pyqtSignal(dict)
    progreso = pyqtSignal(int)
    def __init__(self, pregunta, top_k=10, mode="semantica", options=None):
        super().__init__(); self.pregunta, self.top_k, self.mode, self.options = pregunta, top_k, mode, options
    def run(self):
        func = buscar_semantica if self.mode == "semantica" else buscar_lexica
        res = func(self.pregunta, top_k=self.top_k, progress_callback=self.progreso.emit, options=self.options)
        self.resultados_ready.emit(res)

# --- HISTORIAL Y PROYECTOR ---
def load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except: pass
    return []
def save_history(hist):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f: json.dump(hist[-500:], f, ensure_ascii=False, indent=2)
    except: pass
class ProjectorWindow(QDialog):
    def __init__(self, text_html, reference_text):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowFullScreen)
        self.setStyleSheet("background:black; color:white;"); layout = QVBoxLayout()
        self.label = QLabel(f"<div style='color:white; font-size:calc(2vw + 18px); line-height:1.2; text-align:center;'>{text_html}<br><br><small>{html.escape(reference_text)} — RVR60</small></div>")
        self.label.setTextFormat(Qt.TextFormat.RichText); self.label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.label.setWordWrap(True)
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        btn_close = QPushButton("Cerrar (Esc)"); btn_close.setFixedWidth(140); btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter); self.setLayout(layout)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape: self.close()
        else: super().keyPressEvent(event)

# --- CLASE PRINCIPAL DE LA APLICACIÓN ---
class BibliaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Buscador de Textos bíblicos Offline RVR60")
        if os.path.exists(ICO_IMAGE_PATH): self.setWindowIcon(QIcon(ICO_IMAGE_PATH))
        self.setGeometry(120,60,1180,820)
        self.thread = self.results_builder = self.chapter_builder = None
        self.latest_resultados = {}; self.current_filters_books = set(); self.current_star_filter = None
        self.last_selected_ref = self.current_query = ""
        self.items_to_populate = []
        self.lex_options = {'ignore_accents': True, 'ignore_case': True, 'exact_phrase': False, 'match_all': False}
        self.setFont(QFont("Segoe UI", 10))
        self.init_ui()
        self.apply_books_status_label()

    def init_ui(self):
        main = QVBoxLayout()
        top_h = QHBoxLayout()

        left_v = QVBoxLayout()
        input_h = QHBoxLayout()
        self.input_pregunta = QLineEdit()
        self.input_pregunta.setPlaceholderText("Escribe tu búsqueda...")
        self.input_pregunta.returnPressed.connect(self.iniciar_busqueda)
        self.btn_buscar = QPushButton("Buscar")
        self.btn_buscar.clicked.connect(self.iniciar_busqueda)
        input_h.addWidget(self.input_pregunta)
        input_h.addWidget(self.btn_buscar)
        left_v.addLayout(input_h)

        mode_h = QHBoxLayout()
        mode_h.addWidget(QLabel("Modo:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Búsqueda Semántica", "Búsqueda Léxica"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_h.addWidget(self.mode_combo)
        left_v.addLayout(mode_h)

        self.lexic_frame = QFrame()
        self.lexic_frame.setFrameShape(QFrame.Shape.StyledPanel)
        lex_layout = QGridLayout()
        self.lexic_frame.setLayout(lex_layout)
        self.chk_ignore_accents = QCheckBox("Ignorar acentos")
        self.chk_ignore_accents.setChecked(self.lex_options['ignore_accents'])
        self.chk_ignore_case = QCheckBox("Ignorar mayúsculas/minúsculas")
        self.chk_ignore_case.setChecked(self.lex_options['ignore_case'])
        self.chk_exact_phrase = QCheckBox("Frase exacta")
        self.chk_exact_phrase.setChecked(self.lex_options['exact_phrase'])
        self.chk_match_all = QCheckBox("Todas las palabras")
        self.chk_match_all.setChecked(self.lex_options['match_all'])
        lex_layout.addWidget(self.chk_ignore_accents, 0, 0)
        lex_layout.addWidget(self.chk_ignore_case, 0, 1)
        lex_layout.addWidget(self.chk_exact_phrase, 1, 0)
        lex_layout.addWidget(self.chk_match_all, 1, 1)
        left_v.addWidget(self.lexic_frame)
        self.lexic_frame.setVisible(False)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_v.addWidget(self.progress)

        filtros_frame = QFrame()
        filtros_frame.setFrameShape(QFrame.Shape.StyledPanel)
        filtros_frame.setFixedWidth(340)
        filtros_layout = QVBoxLayout()
        filtros_frame.setLayout(filtros_layout)
        filtros_layout.addWidget(QLabel("Filtrar por estrellas:"))
        stars_h = QHBoxLayout()
        self.star_buttons={}
        for lbl in ["Todos","1★","2★","3★"]:
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.clicked.connect(self.on_star_button_clicked)
            stars_h.addWidget(b)
            self.star_buttons[lbl]=b
        self.star_buttons["Todos"].setChecked(True)
        filtros_layout.addLayout(stars_h)

        filtros_layout.addSpacing(8)
        filtros_layout.addWidget(QLabel("Filtrar por libros:"))
        self.books_list = QListWidget()
        self.books_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.books_list.addItems(["— Acciones rápidas —", "Todos", "Antiguo Testamento", "Nuevo Testamento", "— Libros —"])
        for code in orden_biblia:
            item = QListWidgetItem(book_code_to_spanish.get(code, code))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, code)
            self.books_list.addItem(item)
        self.books_list.itemClicked.connect(self.on_books_list_item_clicked)
        self.books_list.itemDoubleClicked.connect(self.on_books_list_item_doubleclicked)
        filtros_layout.addWidget(self.books_list)
        self.lbl_books_status = QLabel("Todos")
        filtros_layout.addWidget(self.lbl_books_status)
        btn_clear = QPushButton("Borrar selección")
        btn_clear.clicked.connect(self.clear_books_selection)
        filtros_layout.addWidget(btn_clear)
        left_v.addWidget(filtros_frame)

        right_v = QVBoxLayout()
        self.resultados_lista = QListWidget()
        self.resultados_lista.currentRowChanged.connect(self.on_result_selection_changed)
        right_v.addWidget(QLabel("Resultados:"))
        right_v.addWidget(self.resultados_lista, stretch=1)

        res_loader_h = QHBoxLayout()
        res_loader_h.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.results_loader_label = QLabel("Cargando resultados...")
        self.results_loader_label.setVisible(False)
        self.results_loader_label.setStyleSheet("background:#222;color:white;padding:6px;border-radius:6px;font-size:11px;")
        res_loader_h.addWidget(self.results_loader_label)
        right_v.addLayout(res_loader_h)

        right_v.addWidget(QLabel("Capítulo:"))
        self.resultado_text = QTextEdit()
        self.resultado_text.setReadOnly(True)
        right_v.addWidget(self.resultado_text, stretch=2)

        chap_loader_h = QHBoxLayout()
        chap_loader_h.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.chapter_loader_label = QLabel("Cargando capítulo...")
        self.chapter_loader_label.setVisible(False)
        self.chapter_loader_label.setStyleSheet("background:#222;color:white;padding:6px;border-radius:6px;font-size:11px;")
        chap_loader_h.addWidget(self.chapter_loader_label)
        right_v.addLayout(chap_loader_h)

        goto_h = QHBoxLayout()
        self.libro_input = QComboBox()
        [self.libro_input.addItem(book_code_to_spanish.get(c,c)) for c in orden_biblia]
        self.cap_input = QSpinBox()
        self.cap_input.setMinimum(1)
        self.cap_input.setMaximum(150)
        self.btn_ir = QPushButton("Ir a Capítulo")
        self.btn_ir.clicked.connect(self.mostrar_capitulo)
        goto_h.addWidget(QLabel("Libro:"))
        goto_h.addWidget(self.libro_input)
        goto_h.addWidget(QLabel("Capítulo:"))
        goto_h.addWidget(self.cap_input)
        goto_h.addWidget(self.btn_ir)

        self.btn_copy_ref = QPushButton("Copiar referencia")
        self.btn_copy_ref.clicked.connect(self.copy_reference_to_clipboard)
        self.btn_proyect = QPushButton("Proyectar texto")
        self.btn_proyect.clicked.connect(self.project_current_text)
        goto_h.addWidget(self.btn_copy_ref)
        right_v.addLayout(goto_h)

        hist_h = QHBoxLayout()
        hist_h.addWidget(QLabel("Historial:"))
        btn_clear_hist = QPushButton("Borrar historial")
        btn_clear_hist.clicked.connect(self.clear_history)
        hist_h.addWidget(btn_clear_hist)
        right_v.addLayout(hist_h)
        self.historial_lista = QListWidget()
        self.load_history_into_widget()
        self.historial_lista.itemClicked.connect(self.cargar_historial)
        right_v.addWidget(self.historial_lista)

        top_h.addLayout(left_v, stretch=0)
        top_h.addLayout(right_v, stretch=1)
        main.addLayout(top_h)
        self.setLayout(main)
        self.setStyleSheet("QListWidget{font-size:12px;}QLabel{color:#222;}QPushButton{background:#f6f9ff;border:1px solid #d8e6ff;border-radius:6px;padding:6px;}QPushButton:pressed{background:#e8f2ff;}mark{background:yellow;color:black;}")

    def stop_thread(self, thread):
        if thread and thread.isRunning():
            thread.requestInterruption()
            thread.wait(2000)

    def on_search_results(self, resultados):
        self.latest_resultados = resultados
        self.apply_filters_and_populate_async()

    def apply_filters_and_populate_async(self):
        self.stop_thread(self.results_builder)
        self.items_to_populate.clear()
        if self.resultados_lista.count() > 0: self.resultados_lista.clear() # Limpia visualmente si hay algo
        if not self.latest_resultados: return
        self.results_loader_label.setVisible(True)
        self.results_builder = ResultsBuildThread(self.latest_resultados, orden_biblia, book_code_to_spanish, book_list, chapter_list, verse_start_list, verse_end_list, text_list, self.current_filters_books, self.current_star_filter, self.current_query, self.lex_options)
        self.results_builder.finished_build.connect(self._on_results_built)
        self.results_builder.start()

    def _on_results_built(self, items):
        self.results_loader_label.setVisible(False)
        if items:
            self.items_to_populate = items
            QTimer.singleShot(0, self._populate_results_iteratively)

    def _populate_results_iteratively(self):
        if not self.items_to_populate:
            if self.resultados_lista.count() > 0: self.resultados_lista.setCurrentRow(0)
            return
        it = self.items_to_populate.pop(0)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, {"idx": it['idx'], "score": it['score'], "book": it['book'], "vs": it['vs'], "ve": it['ve']})
        lbl = QLabel(it['html']); lbl.setTextFormat(Qt.TextFormat.RichText); lbl.setWordWrap(True)
        self.resultados_lista.addItem(item); self.resultados_lista.setItemWidget(item, lbl)
        QTimer.singleShot(0, self._populate_results_iteratively)

    def _on_chapter_built(self, html_content, reference, verse_to_center):
        self.chapter_loader_label.setVisible(False)
        if html_content:
            self.resultado_text.setHtml(html_content)
            self.last_selected_ref = reference
            if verse_to_center > 0:
                # <--- MODIFICACIÓN: Usar el método fiable scrollToAnchor
                QTimer.singleShot(50, lambda: self.resultado_text.scrollToAnchor(f"verse_{verse_to_center}"))
    
    def mostrar_resultado_por_idx_async(self, idx, libro_code, vs, ve):
        cap_num = int(chapter_list[idx])
        self.libro_input.setCurrentText(book_code_to_spanish.get(libro_code, libro_code))
        self.cap_input.setValue(cap_num)
        self.chapter_loader_label.setVisible(True)
        self.stop_thread(self.chapter_builder)
        self.chapter_builder = ChapterBuildThread(libro_code, cap_num, self.current_query, self.lex_options, selected_vs=vs, selected_ve=ve)
        self.chapter_builder.finished_build.connect(self._on_chapter_built)
        self.chapter_builder.start()

    def on_result_selection_changed(self, row):
        if row < 0: return
        # Prevenir doble ejecución si el click ya lo hizo
        if self.resultados_lista.signalsBlocked(): return
        item = self.resultados_lista.item(row)
        if not item: return
        data = item.data(Qt.ItemDataRole.UserRole)
        if data: self.mostrar_resultado_por_idx_async(data["idx"], data["book"], data["vs"], data["ve"])

    def iniciar_busqueda(self):
        pregunta = self.input_pregunta.text().strip()
        if not pregunta or (self.thread and self.thread.isRunning()): return
        
        # <--- MODIFICACIÓN: Limpiar la lista de resultados inmediatamente
        self.stop_thread(self.results_builder)
        self.items_to_populate.clear()
        self.resultados_lista.clear()
        self.resultado_text.clear() # Opcional: limpiar también el visor
        
        opts = [self.chk_ignore_accents,self.chk_ignore_case,self.chk_exact_phrase,self.chk_match_all]
        keys = ['ignore_accents','ignore_case','exact_phrase','match_all']
        self.lex_options = {k: c.isChecked() for k,c in zip(keys, opts)}
        
        self.prepend_history(pregunta)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_buscar.setEnabled(False)
        self.current_query = pregunta
        mode = "lexica" if "Léxica" in self.mode_combo.currentText() else "semantica"
        self.thread = BusquedaThread(pregunta, top_k=50, mode=mode, options=self.lex_options)
        self.thread.resultados_ready.connect(self.on_search_results)
        self.thread.progreso.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()
        
    def on_thread_finished(self):
        self.btn_buscar.setEnabled(True)
        self.progress.setVisible(False)
        try:
            import winsound
            winsound.MessageBeep()
        except ImportError:
            app.beep()
        except Exception: pass

    def load_history_into_widget(self):
        self.historial_lista.clear()
        for h in reversed(load_history()):
            self.historial_lista.addItem(h)

    def prepend_history(self, item_text):
        hist = load_history()
        if item_text in hist: hist.remove(item_text)
        hist.append(item_text)
        save_history(hist)
        self.load_history_into_widget()
        
    def on_mode_changed(self, idx): self.lexic_frame.setVisible("Léxica" in self.mode_combo.currentText())
    
    def on_books_list_item_clicked(self, item):
        txt = item.text()
        if "—" in txt: return
        if txt == "Todos": self.select_all_books()
        elif "Testamento" in txt: self.select_testament(ANTIGUO_CODES if "Antiguo" in txt else NUEVO_CODES)
        else: self.update_current_books_from_ui(); self.apply_filters_and_populate_async()

    def on_books_list_item_doubleclicked(self, item):
        if not item.data(Qt.ItemDataRole.UserRole): return
        is_checked = item.checkState() == Qt.CheckState.Checked
        for i in range(self.books_list.count()):
            it = self.books_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole): it.setCheckState(Qt.CheckState.Unchecked)
        item.setCheckState(Qt.CheckState.Checked if not is_checked else Qt.CheckState.Unchecked)
        self.update_current_books_from_ui(); self.apply_filters_and_populate_async()

    def select_all_books(self): self.set_book_checkstate(lambda code: True); self.update_current_books_from_ui(); self.apply_filters_and_populate_async()
    def select_testament(self, codes): self.set_book_checkstate(lambda code: code in codes); self.update_current_books_from_ui(); self.apply_filters_and_populate_async()
    def clear_books_selection(self): self.set_book_checkstate(lambda code: False); self.update_current_books_from_ui(); self.apply_filters_and_populate_async()
    def set_book_checkstate(self, condition):
        for i in range(self.books_list.count()):
            it = self.books_list.item(i); code = it.data(Qt.ItemDataRole.UserRole)
            if code: it.setCheckState(Qt.CheckState.Checked if condition(code) else Qt.CheckState.Unchecked)
            
    def update_current_books_from_ui(self):
        self.current_filters_books = {self.books_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.books_list.count()) if self.books_list.item(i).data(Qt.ItemDataRole.UserRole) and self.books_list.item(i).checkState() == Qt.CheckState.Checked}
        self.apply_books_status_label()

    def apply_books_status_label(self):
        sel_count = len(self.current_filters_books)
        if sel_count == 0 or sel_count == len(orden_biblia): self.lbl_books_status.setText("Todos")
        elif sel_count == 1: self.lbl_books_status.setText(book_code_to_spanish.get(next(iter(self.current_filters_books))))
        else: self.lbl_books_status.setText(f"{sel_count} libros seleccionados")

    def on_star_button_clicked(self):
        sender_btn = self.sender()
        is_all_btn = sender_btn.text() == "Todos"
        
        if is_all_btn:
             self.current_star_filter = None
        else:
            if sender_btn.isChecked():
                self.current_star_filter = int(sender_btn.text()[0])
            else:
                self.current_star_filter = None

        self.star_buttons["Todos"].setChecked(self.current_star_filter is None)
        for lbl, btn in self.star_buttons.items():
            if lbl != "Todos":
                btn.setChecked(self.current_star_filter is not None and int(lbl[0]) == self.current_star_filter)

        self.apply_filters_and_populate_async()

    def mostrar_capitulo(self):
        spanish = self.libro_input.currentText(); libro_code = spanish_to_book_code.get(spanish, spanish); cap = self.cap_input.value()
        self.chapter_loader_label.setVisible(True); self.stop_thread(self.chapter_builder)
        self.chapter_builder = ChapterBuildThread(libro_code, cap, self.current_query, self.lex_options)
        self.chapter_builder.finished_build.connect(self._on_chapter_built); self.chapter_builder.start()

    def copy_reference_to_clipboard(self):
        ref = self.last_selected_ref or f"{self.libro_input.currentText()} {self.cap_input.value()}"
        QApplication.clipboard().setText(ref); QMessageBox.information(self, "Copiado", f"Referencia copiada:\n{ref}")

    def project_current_text(self): ProjectorWindow(self.resultado_text.toPlainText().replace("\n", "<br>"), self.last_selected_ref).exec()
    def cargar_historial(self, item): self.input_pregunta.setText(item.text()); self.iniciar_busqueda()
    def clear_history(self): save_history([]); self.historial_lista.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = None
    if os.path.exists(SPLASH_IMAGE_PATH):
        splash = QSplashScreen(QPixmap(SPLASH_IMAGE_PATH))
        splash.show()
        app.processEvents()
    
    cargar_recursos()

    window = BibliaApp()
    window.show()
    if splash:
        splash.finish(window)

    sys.exit(app.exec())