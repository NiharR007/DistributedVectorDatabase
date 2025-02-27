from setuptools import setup, find_packages

setup(
    name="distributed_vector_db",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.5,<1.25.0",
        "scipy>=1.4.1",
        "faiss-cpu>=1.7.0",
        "flask>=2.0.0",
        "pyyaml>=5.1",
        "requests>=2.25.0",
        "scikit-learn>=0.24.2",
    ],
)
