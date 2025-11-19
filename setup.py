"""
Setup configuration for basket trading with Bayesian Optimization.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="basket-trading-bayesian-optimization",
    version="1.0.0",
    author="Senior Quant Researcher",
    description="Production-grade basket trading system with Bayesian Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scipy>=1.11.2",
        "yfinance>=0.2.28",
        "pandas-datareader>=0.10.0",
        "statsmodels>=0.14.0",
        "scikit-optimize>=0.9.0",
        "scikit-learn>=1.3.0",
        "tables>=3.8.0",
        "pyarrow>=13.0.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.17.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.2",
        "pytest-cov>=4.1.0",
        "hypothesis>=6.88.1",
        "tqdm>=4.66.1",
        "joblib>=1.3.2",
    ],
    extras_require={
        "dev": [
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "isort>=5.12.0",
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.3",
            "jupyter>=1.0.0",
            "ipywidgets>=8.1.1",
        ],
    },
)
