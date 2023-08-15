from setuptools import setup,find_packages

DESCRIPTION = "Python package to execute the Leading Eigenvector Dynamics Analysis (LEiDA) on functional MRI data"

with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

VERSION = "1.1"

base_packages = [
    "numpy>=1.16.0",
    "pandas",
    "scipy>=1.8",
    "matplotlib>=3.4.2",
    "seaborn",
    "nilearn",
    "scikit-learn",
    "imageio"
    ]

setup(
    name="pyleida",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url="https://github.com/PSYMARKER/leida-python",
    author="Alvaro Deleglise",
    author_email="alvarodeleglise@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=base_packages,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        ],
    zip_safe=False,
    include_package_data=True,
    #package_data={'pyleida': ['clustering/parc_MNI2mm.npz']},
    package_data={'pyleida': ['*.npz']}
    )