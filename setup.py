"""
NetraVision AI — Setup Script
An Intelligent Assistive Vision System for the Visually Impaired
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="netravision-ai",
    version="1.0.0",
    author="Harshit Tyagi",
    description="An Intelligent Assistive Vision System for the Visually Impaired",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tyagism/NetraVision-AI",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    entry_points={
        "console_scripts": [
            "netravision=main:main",
        ],
    },
)
