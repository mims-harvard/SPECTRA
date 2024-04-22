from setuptools import setup

setup(
    name='spectrae',
    version='1.0.1',    
    description='SPECTRA: The spectral framework for model evaluation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mims-harvard/SPECTRA/',
    author='Yasha Ektefaie',
    author_email='yasha_ektefaie@g.harvard.edu',
    license='MIT License',
    packages=['spectrae'],
    install_requires=['networkx',
                      'numpy', 
                      'torch', 
                      'scikit-learn',
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
