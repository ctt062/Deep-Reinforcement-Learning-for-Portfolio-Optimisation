"""Setup script for Deep Reinforcement Learning Portfolio Optimization."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="drl-portfolio-optimization",
    version="1.0.0",
    author="IEDA4000F Project",
    description="Deep Reinforcement Learning for Portfolio Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ctt062/Deep-Reinforcement-Learning-for-Portfolio-Optimisation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.28.0",
        "yfinance>=0.2.18",
        "cvxpy>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.64.0",
    ],
)
