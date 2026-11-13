#include <jni.h>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms.h>
#include <dlib/serialize.h>

using namespace dlib;

// Arquitectura oficial usada por face_recognition/dlib para producir descriptores 128D.
template <template <int, template<typename> class, int, typename> class block,
          int N, template<typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename> class, int, typename> class block,
          int N, template<typename> class BN, typename SUBNET>
using residual_down =
    add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;

template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;

template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;

template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;

template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;

template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

static std::mutex g_mutex;
static std::string g_last_error = "Listo; todavÃƒÂ­a no se ha ejecutado reconocimiento.";
static std::unique_ptr<shape_predictor> g_shape_predictor;
static std::unique_ptr<anet_type> g_face_net;

static jstring toJavaString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

static std::string fromJavaString(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        throw std::runtime_error("Se recibiÃƒÂ³ una ruta vacÃƒÂ­a.");
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        throw std::runtime_error("No se pudo leer una ruta desde Kotlin.");
    }
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static void loadModelsIfNeeded(const std::string& predictorPath,
                               const std::string& recognitionModelPath) {
    if (!g_shape_predictor) {
        auto predictor = std::make_unique<shape_predictor>();
        deserialize(predictorPath) >> *predictor;
        g_shape_predictor = std::move(predictor);
    }
    if (!g_face_net) {
        auto net = std::make_unique<anet_type>();
        deserialize(recognitionModelPath) >> *net;
        g_face_net = std::move(net);
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_reconocimientofacial_NativeFaceBridge_nativeStatus(
        JNIEnv* env, jobject /* this */) {
    return toJavaString(
        env,
        "NEXUS JNI activo. dlib se inicializarÃƒÂ¡ al recibir los archivos de modelo."
    );
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_reconocimientofacial_NativeFaceBridge_nativeLastError(
        JNIEnv* env, jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return toJavaString(env, g_last_error);
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_reconocimientofacial_NativeFaceBridge_nativeEncode(
        JNIEnv* env,
        jobject /* this */,
        jintArray argbPixels,
        jint width,
        jint height,
        jstring predictorPath,
        jstring recognitionModelPath) {

    std::lock_guard<std::mutex> lock(g_mutex);

    auto emptyResult = [&]() -> jfloatArray {
        return env->NewFloatArray(0);
    };

    try {
        g_last_error.clear();

        if (argbPixels == nullptr || width <= 0 || height <= 0 ||
            width > 4096 || height > 4096) {
            g_last_error = "La imagen recibida no tiene dimensiones vÃƒÂ¡lidas.";
            return emptyResult();
        }

        const jsize pixelCount = env->GetArrayLength(argbPixels);
        const jlong expectedCount = static_cast<jlong>(width) * static_cast<jlong>(height);
        if (expectedCount != pixelCount) {
            g_last_error = "El tamaÃƒÂ±o de los pÃƒÂ­xeles no coincide con ancho Ãƒâ€” alto.";
            return emptyResult();
        }

        const std::string predictorFile = fromJavaString(env, predictorPath);
        const std::string modelFile = fromJavaString(env, recognitionModelPath);
        loadModelsIfNeeded(predictorFile, modelFile);

        std::vector<jint> pixels(static_cast<size_t>(pixelCount));
        env->GetIntArrayRegion(argbPixels, 0, pixelCount, pixels.data());
        if (env->ExceptionCheck()) {
            g_last_error = "No se pudieron leer los pÃƒÂ­xeles de la fotografÃƒÂ­a.";
            return emptyResult();
        }

        array2d<rgb_pixel> image(height, width);
        for (jint y = 0; y < height; ++y) {
            for (jint x = 0; x < width; ++x) {
                const uint32_t argb =
                    static_cast<uint32_t>(pixels[static_cast<size_t>(y) * width + x]);
                image[y][x] = rgb_pixel(
                    static_cast<unsigned char>((argb >> 16) & 0xFF),
                    static_cast<unsigned char>((argb >> 8) & 0xFF),
                    static_cast<unsigned char>(argb & 0xFF)
                );
            }
        }

        // Igual que face_recognition.face_locations(..., model="hog", upsample=1).
        static frontal_face_detector detector = get_frontal_face_detector();
        const std::vector<rectangle> faces = detector(image, 0);

        if (faces.empty()) {
            g_last_error = "No se detectÃƒÂ³ ningÃƒÂºn rostro. Centra la cara y mejora la iluminaciÃƒÂ³n.";
            return emptyResult();
        }
        if (faces.size() != 1) {
            g_last_error = "Se detectaron varios rostros. Deja una sola cara dentro del encuadre.";
            return emptyResult();
        }

        // El script Python usa model='small': predictor facial de 5 puntos.
        const full_object_detection landmarks = (*g_shape_predictor)(image, faces.front());
        if (landmarks.num_parts() != 5) {
            g_last_error = "El predictor debe ser el modelo dlib de 5 puntos.";
            return emptyResult();
        }

        // Mismo alineamiento que face_recognition: chip 150x150 y padding 0.25.
        matrix<rgb_pixel> faceChip;
        extract_image_chip(image, get_face_chip_details(landmarks, 150, 0.25), faceChip);

        std::vector<matrix<rgb_pixel>> chips;
        chips.push_back(faceChip);
        const auto descriptors = (*g_face_net)(chips, 1);

        if (descriptors.size() != 1 || descriptors[0].size() != 128) {
            g_last_error = "dlib no generÃƒÂ³ un vector facial vÃƒÂ¡lido de 128 dimensiones.";
            return emptyResult();
        }

        jfloatArray output = env->NewFloatArray(128);
        if (output == nullptr) {
            g_last_error = "No se pudo reservar memoria para el encoding facial.";
            return emptyResult();
        }

        jfloat values[128];
        for (int i = 0; i < 128; ++i) {
            values[i] = descriptors[0](i);
        }
        env->SetFloatArrayRegion(output, 0, 128, values);
        if (env->ExceptionCheck()) {
            g_last_error = "No se pudo devolver el encoding a Kotlin.";
            return emptyResult();
        }

        g_last_error = "OK: encoding dlib de 128 dimensiones generado.";
        return output;

    } catch (const std::exception& ex) {
        g_last_error = std::string("Error dlib: ") + ex.what();
        return emptyResult();
    } catch (...) {
        g_last_error = "Error nativo desconocido al generar el encoding.";
        return emptyResult();
    }
}
