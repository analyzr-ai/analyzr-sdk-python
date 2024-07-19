import pathlib
from setuptools import setup


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='analyzr-sdk-python',
    version='1.4.0',
    description='Python SDK for the G2M Platform API',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/analyzr-ai/analyzr-sdk-python",
    author='G2M Team',
    author_email='support@g2m.ai',
    license='Apache 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=['analyzrclient'],
    install_requires=[
    ]
)
