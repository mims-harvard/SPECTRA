from setuptools import setup

setup(
    name='spectra',
    version='0.1.0',    
    description='SPECTRA: The spectral framework for model evaluation',
    url='https://github.com/mims-harvard/SPECTRA/',
    author='Yasha Ektefaie',
    author_email='yasha_ektefaie@g.harvard.edu',
    license='MIT License',
    packages=['spectra'],
    install_requires=['networkx',
                      'numpy', 
                      'torch', 
                      'sklearn',
                      'pandas',
                      'tqdm',
                      'cell-gears',
                      'torch_geometric',
                      'PyTDC'                   
                      ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
    ],
)