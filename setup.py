from setuptools import setup, find_packages

__version__ = "0.0.1"

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="HERGAST",
    version=__version__,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    install_requires=[
        'torch',
        'scanpy',
        'scikit-learn',
        'torch_geometric',
        'scipy',
        'numba',
        ],
    author="Yuqiao Gong",
    author_email="gyq123@sjtu.edu.cn",
    keywords=["spatial transcriptomic", "HERGAST", "spatial clustering", "signal amplification"],
    description="Relational Graph Attention Network for Spatial Transcriptome Analysis",
    license="MIT",
    url='https://github.com/GYQ-form/HERGAST',
    long_description_content_type='text/markdown',
    long_description=long_description
)
