import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyraug",
    version="0.0.1",
    author="Clement Chadebec (HekA team INRIA)",
    author_email="clement.chadebec@inria.fr",
    description="Data Augmentation with VAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={"Bug Tracker": "https://github.com/pypa/sampleproject/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.8.1",
        "dill>=0.3.3",
        "nibabel>=3.2.1",
        "pydantic>=1.8.2",
        "dataclasses>=0.6",
        "Pillow>=8.3.0",
    ],
    python_requires=">=3.6",
)
