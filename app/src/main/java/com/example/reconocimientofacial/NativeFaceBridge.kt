
package com.example.reconocimientofacial

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import kotlin.math.roundToInt

object NativeFaceBridge {
    init {
        System.loadLibrary("nexusface")
    }

    external fun nativeStatus(): String
    external fun nativeLastError(): String

    private external fun nativeEncode(
        argbPixels: IntArray,
        width: Int,
        height: Int,
        predictorPath: String,
        recognitionModelPath: String
    ): FloatArray

    fun encode(
        bitmap: Bitmap,
        predictorPath: String,
        recognitionModelPath: String
    ): FloatArray {
        val started = SystemClock.elapsedRealtimeNanos()

        // Limitar el tamaño para acelerar la detección facial.
        val maxDimension = 640
        val scale = minOf(
            1f,
            maxDimension.toFloat() /
                    maxOf(bitmap.width, bitmap.height)
        )

        val scaledBitmap = if (scale < 1f) {
            Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * scale).roundToInt().coerceAtLeast(1),
                (bitmap.height * scale).roundToInt().coerceAtLeast(1),
                true
            )
        } else {
            bitmap
        }

        val workingBitmap =
            if (scaledBitmap.config == Bitmap.Config.ARGB_8888) {
                scaledBitmap
            } else {
                scaledBitmap.copy(Bitmap.Config.ARGB_8888, false)
                    ?: throw IllegalArgumentException(
                        "No se pudo convertir la fotografía a ARGB_8888."
                    )
            }

        try {
            val pixels = IntArray(
                workingBitmap.width * workingBitmap.height
            )

            workingBitmap.getPixels(
                pixels,
                0,
                workingBitmap.width,
                0,
                0,
                workingBitmap.width,
                workingBitmap.height
            )

            val beforeNative = SystemClock.elapsedRealtimeNanos()

            val result = nativeEncode(
                pixels,
                workingBitmap.width,
                workingBitmap.height,
                predictorPath,
                recognitionModelPath
            )

            val finished = SystemClock.elapsedRealtimeNanos()

            Log.i(
                "NEXUS_PERF",
                "Original=${bitmap.width}x${bitmap.height}, " +
                        "procesada=${workingBitmap.width}x${workingBitmap.height}, " +
                        "preparación=${(beforeNative - started) / 1_000_000} ms, " +
                        "motor nativo=${(finished - beforeNative) / 1_000_000} ms, " +
                        "total=${(finished - started) / 1_000_000} ms"
            )

            return result
        } finally {
            if (workingBitmap !== bitmap && !workingBitmap.isRecycled) {
                workingBitmap.recycle()
            }

            if (
                scaledBitmap !== bitmap &&
                scaledBitmap !== workingBitmap &&
                !scaledBitmap.isRecycled
            ) {
                scaledBitmap.recycle()
            }
        }
    }
}
