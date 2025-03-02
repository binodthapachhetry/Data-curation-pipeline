from setuptools import setup, find_packages

setup(
    name="datacuration",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy",
        ],
    },
    description="Pipeline to curate data for training, testing, evaluating large models",
    author="DataCuration Team",
    python_requires=">=3.8",
)
