cmake_minimum_required(VERSION 3.22.1)
project("nexusface")

include(FetchContent)

# dlib se compila como biblioteca estÃƒÂ¡tica dentro de la app.
# La app no necesita GUI, CUDA, BLAS ni lectores de imÃƒÂ¡genes de escritorio.
set(DLIB_ISO_CPP_ONLY OFF CACHE BOOL "" FORCE)
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "" FORCE)
set(DLIB_USE_CUDA OFF CACHE BOOL "" FORCE)
set(DLIB_USE_BLAS OFF CACHE BOOL "" FORCE)
set(DLIB_USE_LAPACK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    dlib
    URL https://github.com/davisking/dlib/archive/refs/tags/v19.24.9.zip
)
FetchContent_MakeAvailable(dlib)

add_library(
    nexusface
    SHARED
    native-lib.cpp
)

find_library(log-lib log)

target_compile_features(nexusface PRIVATE cxx_std_17)
target_link_libraries(nexusface PRIVATE dlib ${log-lib})

# NEXUS_NATIVE_OPTIMIZATION
# Optimizar el motor facial para ejecucion en CPU Android.
if(TARGET nexusface)
    target_compile_options(nexusface PRIVATE -O3)
endif()

if(TARGET dlib)
    target_compile_options(dlib PRIVATE -O3)
endif()