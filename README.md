# -IA-Offline-Buscador-de-Textos-B-blicos-Biblia-RVR60-Embeddings-Gemma-
Este proyecto es una aplicación offline diseñada para realizar búsquedas rápidas y precisas en la Biblia Reina-Valera 1960 (RVR60) utilizando embeddings generados con Gemma.  La herramienta permite localizar pasajes bíblicos relacionados semánticamente con la consulta del usuario, incluso si no se escribe la cita o palabra exacta.

CÓMO USAR
----------

1. Abre el archivo "buscasor_de_textos_biblicos.exe" para ejecutar el programa.

2. Si deseas crear un acceso directo:
   - Haz clic derecho sobre el archivo .exe
   - Selecciona "Enviar a → Escritorio (crear acceso directo)"

3. Para compartir el programa:
   - Comprime la carpeta completa "IA Offline Buscador Biblia RVR60"
   - Comparte el archivo comprimido.

IMPORTANTE:
- No muevas el archivo "buscasor_de_textos_biblicos.exe".
- El .exe debe estar siempre junto a la carpeta "data", ya que ahí se encuentran
  los modelos y bases de datos necesarias.


ESTRUCTURA DE CARPETAS Y ARCHIVOS
---------------------------------

IA Offline Buscador Biblia RVR60/
│
├── buscasor_de_textos_biblicos.exe
├── data/
│   ├── cross-encoderms-marco-MiniLM-L6-v2/   (Modelo CrossEncoder para reranqueo)
│   ├── embeddinggemma-300m/                  (Modelo de embeddings)
│   ├── icon.ico                              (Icono del programa)
│   ├── icon.png                              (Imagen de icono en PNG)
│   ├── portada.png                           (Imagen de portada splash)
│   ├── rvr60.sqlite3                         (Texto bíblico RVR60)
│   └── rvr60_embeddings.db                   (Base de datos de embeddings vectoriales)


PROCESO DE CREACIÓN
-------------------

- Se utilizó el modelo de incrustación "embeddinggemma-300m".
- El archivo rvr60_embeddings.db tardó 12 horas en procesar los 66 libros
  de la Biblia, generando un total de 89,850 incrustaciones vectoriales
  de 768 dimensiones.
- Se implementó un algoritmo de reranqueo basado en el modelo
  "cross-encoderms-marco-MiniLM-L6-v2" para mejorar la precisión de los resultados.


CONDICIONES DE USO
------------------

- Puedes usar, modificar y compartir este programa siempre que mantengas
  visibles los créditos a RF Electronics de El Salvador.
- El software es gratuito y de uso libre, pero se proporciona "tal cual",
  sin garantías de ningún tipo.
- Los textos mostrados corresponden con precisión a la versión
  Reina-Valera 1960.
- La relación entre los versos sugeridos no está garantizada. En ocasiones
  puede omitirse un verso si se busca por palabras aisladas. Con consultas
  más amplias y contextuales, los resultados tienden a ser más completos
  y exactos.


LICENCIAS DE TERCEROS
----------------------

Este software depende de modelos y librerías con licencias propias.
Debes verificar por separado las condiciones de uso de:

- embeddinggemma-300m
- cross-encoderms-marco-MiniLM-L6-v2


COLABORACIÓN
------------

Si realizas mejoras o modificaciones, por favor comparte una copia al correo:
molina.naves@gmail.com


CRÉDITOS
--------

Autor y Contribuidores: Antonio Naves, RF Electronics de El Salvador (2025).


