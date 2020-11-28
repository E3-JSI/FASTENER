from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fastener",
    version="1.0.4",
    author="Filip Koprivec, Klemen Kenda, Gal Petkovsek",
    author_email="",
    description="Feature selection enabled by entropy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JozefStefanInstitute/FASTENER.git",
    packages=["fastener"],
    install_requires=[
        "mypy",
        "scikit-learn"
        ],
    extras_require={
        "examples":  ["pandas==1.1.3"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    license="MIT",
    python_requires=">=3.6",
)
