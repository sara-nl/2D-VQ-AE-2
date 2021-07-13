import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="2D-VQ-AE-2",
    version="0.0.1",
    author="Robert Jan Schlimbach",
    description="2D Vector-Quantized Auto-Encoder for compression of Whole-Slide Images in Histopathology",
    long_description=long_description,
    url="https://github.com/sara-nl/2D-VQ-AE-2/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)