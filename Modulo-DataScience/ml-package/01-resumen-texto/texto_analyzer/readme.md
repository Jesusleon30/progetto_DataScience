# Setup

Para crear el entorno virtual e instalar los requerimientos:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

# Descarga del modelo

Para utilizar le modelo, es necesario exportar sus archivos y pegarlos en la carpeta `./model`.
El [ejemplo](https://colab.research.google.com/drive/1F2oKyA-Kx5qljHufP5aIiy93-uUQIETj) que se utiliza en el ejercicio.

# Para empaquetar

En la carpeta `texto_analyzer`:

1. Generar los archivos del modelo en colab y colocarlos en una carpeta `./texto_analyzer/texto_analyzer/model`.
2. Crear un entorno virtual e instalar los requerimientos.
3. Generar el paquete
   ```shell
   python .\setup.py sdist bdist_wheel
   ```
4. Para instalarlo, copiar el contenido de `./dist` a `./libs` en otro proyecto:
   ```shell
   pip install .\libs\texto_analyzer-1.0.0-py3-none-any.whl
   ```