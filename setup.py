from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deep-nonlinear-causality",
    version="1.1.0",
    author="Dr. Abdullah Mohammad Ghazi Al khatib",
    author_email="abduall.kh.foas@aspu.edu.sy",
    maintainer_email="genius.275@hotmail.com",
    description="Multivariate Nonlinear Granger Causality Detection with Deep Learning and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ABDULLAH-AL-KHATIB/deep-nonlinear-causality",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "all": ["tensorflow>=2.6.0", "xgboost>=1.5.0", "lightgbm>=3.3.0", "pytest>=7.0.0"]
    }
)
