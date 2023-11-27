import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='analyzr-sdk-python',
    version='1.3.41',
    description='Python SDK for Analyzr API',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/analyzr-ai/analyzr-sdk-python",
    author='Analyzr Team',
    author_email='support@analyzr.ai',
    license='Apache 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=['analyzrclient'],
    install_requires=[
    ]
)
