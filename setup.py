# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="beatricevec",
    version="1.0.1",
    description="BeatriceVec is a powerful Python package/tool designed for generating word embeddings in the dimension of 600",
    url="git@github.com:foscraft/beatrice-project.git#egg=main",
    author="Nyarbari Reuben",
    author_email="anyaribari@gmail.com",
    license="Apache2.0",
    packages=["beatricevec"],
    install_requires=[
        "cython>=0.29.0",  # Add Cython as a dependency
    ],
    ext_modules=cythonize(
        "beatricevec/beats.pyx",  # Path to your .pyx file
        compiler_directives={'language_level': "3"}
    ),
    zip_safe=False,
)