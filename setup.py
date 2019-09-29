import io
import os
import re

from setuptools import find_packages
from setuptools import setup

setup(
    name="webcompat_ml",
    version="0.0.1",
    url="https://github.com/mozilla/webcompat-ml",
    author="John Giannelos",
    author_email="jgiannelos@mozilla.com",
    description="WebCompat machine learning pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "": [
            "models/invalid/model.bin",
            "models/invalid/encoders/*",
            "models/invalid/metadata/*",
            "utils/encoders/*.joblib",
        ]
    },
    entry_points={
        "console_scripts": ["webcompat-ml-invalid=webcompat_ml.models.invalid.cli:main"]
    },
    install_requires=["scikit-learn", "pandas", "xgboost"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
    license="MPL 2.0",
)
