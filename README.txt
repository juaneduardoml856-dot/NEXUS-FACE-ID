NEXUS FACE ID — integración dlib offline

ESTRUCTURA DEL ZIP
- app/src/main/cpp/CMakeLists.txt
- app/src/main/cpp/native-lib.cpp
- app/src/main/java/com/example/reconocimientofacial/NativeFaceBridge.kt
- app/src/main/java/com/example/reconocimientofacial/DlibModelFiles.kt
- app/src/main/java/com/example/reconocimientofacial/DlibOfflineRecognizer.kt
- app/src/main/java/com/example/reconocimientofacial/MainActivity.kt
- build.gradle.kts_DLIB.txt
- descargar_modelos_dlib.ps1

QUÉ HACE
- Compila dlib v19.24.6 como C++ nativo.
- Abre la cámara y espera a que el usuario pulse "Tomar foto".
- dlib detecta un rostro, lo alinea con el predictor de 5 puntos y genera un encoding de 128 valores.
- Compara el encoding con encodings generados por face_recognition con model="small".
- No necesita internet durante el reconocimiento una vez instalada la app.

ANTES DE COPIAR
Haz una copia de seguridad de tu MainActivity.kt y build.gradle.kts actuales.

PASOS
1. Extrae este ZIP en la raíz del proyecto Android Studio, donde está la carpeta app.
2. Copia/reemplaza los archivos dentro de app/src/main/cpp y app/src/main/java/com/example/reconocimientofacial.
3. Abre build.gradle.kts_DLIB.txt y copia su contenido a build.gradle.kts (Module: app). Incluye Compose, CameraX, ExifInterface, ML Kit, TensorFlow Lite y CMake. Limita la compilación nativa a arm64-v8a para el Motorola; no funcionará en emuladores x86.
4. Abre PowerShell en la raíz del proyecto y ejecuta:
   powershell -ExecutionPolicy Bypass -File .\descargar_modelos_dlib.ps1
   Descarga los modelos oficiales y los descomprime en app/src/main/assets/models/.
5. Ejecuta generar_encodings_face_recognition.py en la carpeta que contiene la carpeta personas. El archivo JSON resultante quedará en esa carpeta de datos.
6. Copia encodings_face_recognition.json a app/src/main/assets/encodings_face_recognition.json.
7. En Android Studio: File > Sync Project with Gradle Files.
8. Luego Build > Assemble Project. La primera compilación descarga el código fuente de dlib desde GitHub y puede tardar varios minutos.

MODELOS REQUERIDOS
- app/src/main/assets/models/shape_predictor_5_face_landmarks.dat
- app/src/main/assets/models/dlib_face_recognition_resnet_model_v1.dat

COMPATIBILIDAD
El script Python de este proyecto usa face_recognition.face_encodings(..., model="small"), que corresponde al predictor de 5 puntos. Android usa HOG con upsample=1, chip 150x150 y padding 0.25 para mantener el preprocesamiento compatible.
No mezcles estos encodings con los vectores CNN/MobileNet anteriores.

IMPORTANTE
- La pantalla usa dlib en lugar del híbrido anterior y analiza la foto solo cuando se pulsa "Tomar foto".
- El umbral 0.50 y el margen 0.04 son valores iniciales de prueba, no demuestran exactitud >90%.
- No hay detección de vida; una foto impresa o en otra pantalla podría engañar a un sistema biométrico sin controles adicionales.
- Los modelos ocupan decenas de MB. La primera compilación requiere internet para descargar el código fuente dlib; el reconocimiento en ejecución funciona sin internet.
