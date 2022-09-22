from setuptools import setup,find_packages

DESCRIPTION = "Python package to execute the Leading Eigenvector Dynamics Analysis (LEiDA) on functional MRI data"

VERSION = "1.0"

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
    url="https://github.com/PSYCHOMARK/leida-python",
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
    zip_safe=False
    )