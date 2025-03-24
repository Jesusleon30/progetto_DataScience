from setuptools import setup, find_packages

setup(
  name="texto_analyzer",
  version="1.0.0",
  description="Librería para resumen de texto.",
  author="JesusLeon",
  author_email="jesusleon301192@gmail.com",
  packages=find_packages(),
  include_package_data=True,
  package_data={
    "texto_analyzer":["model/*"]
  },
  install_requires=[
    "transformers==4.49.0",
    "numpy==2.2.3",
    "torch==2.6.0",
    "protobuf==3.20.0",
    "sentencepiece==0.2.0"
  ]
)
