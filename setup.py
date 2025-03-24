from setuptools import setup, find_packages

setup(
    name="objdetect-rf-detr",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.3",
        "pillow>=8.3.1",
        "tqdm>=4.62.3",
    ],
    description="Object Detection Library with RF-DETR support",
    author="Hasan Goni",
    author_email="hasanme1412@gmail.com",
    url="https://github.com/HasanGoni/objdetect-rf-detr",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)