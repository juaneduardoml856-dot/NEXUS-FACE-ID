package com.example.reconocimientofacial

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

data class DlibModelPaths(
    val fivePointPredictor: String,
    val faceRecognitionModel: String
)

object DlibModelFiles {
    /**
     * Copia los modelos desde app/src/main/assets/models/ al almacenamiento privado.
     * Se ejecuta en el hilo de trabajo, no en el hilo principal.
     */
    fun ensure(context: Context): DlibModelPaths {
        val directory = File(context.applicationContext.filesDir, "dlib_models")
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("No se pudo crear la carpeta privada para los modelos dlib.")
        }

        val predictor = File(directory, "shape_predictor_5_face_landmarks.dat")
        val recognition = File(directory, "dlib_face_recognition_resnet_model_v1.dat")

        copyAssetIfNeeded(
            context,
            "models/shape_predictor_5_face_landmarks.dat",
            predictor,
            1_000_000L
        )
        copyAssetIfNeeded(
            context,
            "models/dlib_face_recognition_resnet_model_v1.dat",
            recognition,
            10_000_000L
        )

        return DlibModelPaths(
            fivePointPredictor = predictor.absolutePath,
            faceRecognitionModel = recognition.absolutePath
        )
    }

    private fun copyAssetIfNeeded(
        context: Context,
        assetName: String,
        destination: File,
        minimumBytes: Long
    ) {
        if (destination.isFile && destination.length() > minimumBytes) return

        if (destination.exists() && !destination.delete()) {
            throw IOException("No se pudo reemplazar el archivo incompleto: ${destination.name}")
        }

        val temporary = File(destination.parentFile, destination.name + ".tmp")
        try {
            context.applicationContext.assets.open(assetName).use { input ->
                FileOutputStream(temporary).use { output ->
                    input.copyTo(output)
                }
            }

            if (!temporary.isFile || temporary.length() <= minimumBytes) {
                throw IOException("El modelo $assetName está incompleto o vacío.")
            }

            if (!temporary.renameTo(destination)) {
                temporary.copyTo(destination, overwrite = true)
                temporary.delete()
            }
        } catch (error: Exception) {
            temporary.delete()
            throw IOException(
                "No se pudo cargar '$assetName'. Comprueba que esté en app/src/main/assets/$assetName.",
                error
            )
        }
    }
}
