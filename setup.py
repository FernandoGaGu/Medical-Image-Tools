from setuptools import find_packages, setup
import numpy


# python setup.py install
setup(name='mitools',
      version='0.1.0',
      license='BSD3',
      description='Framework for visualizing and processing medical brain images',
      author='Fernando García Gutiérrez',
      author_email='fegarc05@ucm.es',
      url='https://github.com/FernandoGaGu/Medical-Image-Tools',
      install_requires=[
                'numpy',
                'pandas',
                'plotly',
                'dash',
                'tqdm',
                'joblib',
                'matplotlib',
                'nibabel',
                'nilearn'
            ],
      keywords=['Medical imaging', 'Neuroimaging'],
      packages=find_packages(),
      include_dirs=[numpy.get_include()],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'],
      )
