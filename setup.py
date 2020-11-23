from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fastener',
    version='1.0.3',
    author='Filip Koprivec, Klemen Kenda, Gal Petkovsek',
    author_email='',
    description='Feature selection enabled by entropy',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JozefStefanInstitute/FASTENER.git",
    packages=["fastener"],
    #py_modules=["fastener", "item", "random_utils"],
    install_requires=["joblib==0.14.1",
                    "mypy==0.761",
                    "mypy-extensions==0.4.3",
                    "numpy==1.18.1",
                    "scikit-learn>=0.22.2",
                    "scipy>=1.4.1",
                    "typed-ast>=1.4.1",
                    "typing-extensions>=3.7.4.1"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ],
    license='MIT',
    python_requires='>=3.6',
    package_dir = {'': 'src'},
)