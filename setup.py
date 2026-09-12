"""Package metadata. Runtime versions are pinned in requirements.txt."""

from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="basket-trading-bo",
    version="2.0.0",
    description="Walk-forward research framework for cointegration basket trading with Bayesian-optimised signals",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Shaunak Batra",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24,<2",
        "pandas>=2.0,<2.1",
        "scipy>=1.11",
        "statsmodels>=0.14",
        "scikit-learn>=1.3",
        "scikit-optimize>=0.10.2",
        "yfinance>=0.2.66",
        "matplotlib>=3.7",
        "PyYAML>=6.0",
    ],
    extras_require={"dev": ["pytest>=7.4", "pytest-cov>=4.1", "hypothesis>=6.88"]},
)
