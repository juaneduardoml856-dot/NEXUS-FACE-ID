$ErrorActionPreference = "Stop"

$modelsDir = Join-Path $PSScriptRoot "app\src\main\assets\models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$models = @(
    @{
        File = "shape_predictor_5_face_landmarks.dat"
        Url = "https://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
        MinimumBytes = 1000000
    },
    @{
        File = "dlib_face_recognition_resnet_model_v1.dat"
        Url = "https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
        MinimumBytes = 10000000
    }
)

foreach ($model in $models) {
    $target = Join-Path $modelsDir $model.File

    if ((Test-Path $target) -and ((Get-Item $target).Length -gt $model.MinimumBytes)) {
        Write-Host "Ya existe: $target"
        continue
    }

    $compressed = "$target.bz2"
    Write-Host ""
    Write-Host "Descargando modelo oficial: $($model.File)"
    Invoke-WebRequest -Uri $model.Url -OutFile $compressed

    $pythonCode = "import bz2, pathlib, sys; src=pathlib.Path(sys.argv[1]); dst=src.with_suffix(''); data=bz2.decompress(src.read_bytes()); dst.write_bytes(data); print(str(dst) + ': ' + str(len(data)) + ' bytes')"
    & python -c $pythonCode $compressed

    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo descomprimir $compressed. Comprueba que Python esté instalado."
    }

    if (!(Test-Path $target) -or ((Get-Item $target).Length -le $model.MinimumBytes)) {
        throw "El modelo $($model.File) quedó incompleto."
    }

    Remove-Item $compressed -Force
    Write-Host "Modelo preparado: $target"
}

Write-Host ""
Write-Host "Modelos dlib descargados y descomprimidos."
Write-Host "No se descargan durante el uso de la app: quedarán empaquetados para funcionar offline."
