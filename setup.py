from setuptools import setup, find_packages

setup(
    name="physics-informed-neural-network",
    version="0.1.0",
    description="A modular framework for solving partial differential equations using Physics-Informed Neural Networks",
    author="Kanhaiya Gupta",
    author_email="kanhaiya.lgupta21@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "optuna>=2.10.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
) 