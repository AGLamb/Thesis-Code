from setuptools import setup

setup(
    name='SpiPy',
    version='1.0.0',
    author='Andres Guzman Cordero',
    author_email='andres@andresguzco.com',
    description='Python package for my BSc thesis. The package is designed for'
                'spatio-temporal analysis of polluting particles',
    packages=['SciPy'],
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'sklearn',
        ''
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)