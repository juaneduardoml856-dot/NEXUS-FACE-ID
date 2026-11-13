package com.example.reconocimientofacial

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.IOException
import kotlin.math.sqrt

data class DlibRecognitionResult(
    val accepted: Boolean,
    val personName: String?,
    val distance: Double?,
    val message: String
)

/**
 * Comparador offline de encodings generados con face_recognition/dlib.
 *
 * La tolerancia 0.50 es solo un valor inicial conservador para pruebas.
 * Debe calibrarse con fotografías independientes antes de usar el resultado
 * para autorizar accesos reales.
 */
class DlibOfflineRecognizer(context: Context) {
    private val appContext = context.applicationContext

    private data class KnownEncoding(
        val name: String,
        val values: FloatArray
    )

    private val knownEncodings: List<KnownEncoding> by lazy {
        loadEncodings()
    }

    fun nativeStatus(): String = NativeFaceBridge.nativeStatus()

    fun recognize(
        bitmap: Bitmap,
        tolerance: Double = 0.50,
        minimumMargin: Double = 0.04
    ): DlibRecognitionResult {
        if (knownEncodings.isEmpty()) {
            return DlibRecognitionResult(
                false, null, null,
                "La base de encodings está vacía."
            )
        }

        val paths = try {
            DlibModelFiles.ensure(appContext)
        } catch (error: Exception) {
            return DlibRecognitionResult(
                false, null, null,
                error.message ?: "No se pudieron cargar los modelos dlib."
            )
        }

        val query = try {
            NativeFaceBridge.encode(
                bitmap,
                paths.fivePointPredictor,
                paths.faceRecognitionModel
            )
        } catch (error: UnsatisfiedLinkError) {
            return DlibRecognitionResult(
                false, null, null,
                "No se pudo cargar la biblioteca nativa nexusface: ${error.message}"
            )
        } catch (error: Exception) {
            return DlibRecognitionResult(
                false, null, null,
                error.message ?: "Falló la generación del encoding."
            )
        }

        if (query.size != 128) {
            return DlibRecognitionResult(
                false, null, null,
                NativeFaceBridge.nativeLastError()
            )
        }

        // Obtener la distancia mínima por persona, no por archivo.
        val bestByPerson = mutableMapOf<String, Double>()
        for (known in knownEncodings) {
            val distance = euclideanDistance(query, known.values)
            val previous = bestByPerson[known.name]
            if (previous == null || distance < previous) {
                bestByPerson[known.name] = distance
            }
        }

        val ranked = bestByPerson.entries.sortedBy { it.value }
        if (ranked.isEmpty()) {
            return DlibRecognitionResult(false, null, null, "No hay identidades válidas.")
        }

        val best = ranked[0]
        val secondDistance = ranked.getOrNull(1)?.value
        val margin = secondDistance?.minus(best.value)

        if (best.value > tolerance) {
            return DlibRecognitionResult(
                false, null, best.value,
                "Desconocido. Candidato más cercano: ${best.key}, distancia ${"%.3f".format(java.util.Locale.US, best.value)}."
            )
        }

        if (margin != null && margin < minimumMargin) {
            return DlibRecognitionResult(
                false, null, best.value,
                "Coincidencia ambigua entre personas. No se autoriza el acceso."
            )
        }

        return DlibRecognitionResult(
            true, best.key, best.value,
            "Coincidencia provisional: ${best.key}, distancia ${"%.3f".format(java.util.Locale.US, best.value)}."
        )
    }

    private fun loadEncodings(): List<KnownEncoding> {
        val jsonText = try {
            appContext.assets.open("encodings_face_recognition.json")
                .bufferedReader(Charsets.UTF_8)
                .use { it.readText() }
        } catch (error: Exception) {
            throw IOException(
                "Falta app/src/main/assets/encodings_face_recognition.json. " +
                    "Genera ese archivo con el script Python y cópialo a assets.",
                error
            )
        }

        val root = JSONObject(jsonText)
        val names = root.getJSONArray("known_face_names")
        val vectors = root.getJSONArray("known_face_encodings")
        if (names.length() != vectors.length()) {
            throw IOException("La base JSON tiene diferente número de nombres y encodings.")
        }

        val result = ArrayList<KnownEncoding>(names.length())
        for (index in 0 until names.length()) {
            val name = names.getString(index).trim()
            val vectorJson = vectors.getJSONArray(index)
            if (name.isEmpty() || vectorJson.length() != 128) continue

            val vector = FloatArray(128)
            var valid = true
            for (dimension in 0 until 128) {
                val value = vectorJson.getDouble(dimension)
                if (!value.isFinite()) {
                    valid = false
                    break
                }
                vector[dimension] = value.toFloat()
            }
            if (valid) result += KnownEncoding(name, vector)
        }
        return result
    }

    private fun euclideanDistance(first: FloatArray, second: FloatArray): Double {
        var sum = 0.0
        for (index in 0 until 128) {
            val difference = first[index].toDouble() - second[index].toDouble()
            sum += difference * difference
        }
        return sqrt(sum)
    }
}
