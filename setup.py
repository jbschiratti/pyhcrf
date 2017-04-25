from setuptools import setup
from codecs import open
from os import path
from Cython.Build import cythonize
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pyhcrf',
    version='0.0.1',
    packages=['pyhcrf'],
    install_requires=['numpy>=1.9', 'Cython'],
    ext_modules=cythonize('pyhcrf/algorithms.pyx'),
    include_dirs=[numpy.get_include()],
    url='https://github.com/dirko/pyhcrf',
    license='GPL',
    author='Dirko Coetsee',
    author_email='dpcoetsee@gmail.com',
    description='Hidden conditional random field, a sequence classifier',
    long_description=long_description,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        ],
    )
