# GPT Suite
<p>
<a href="https://pypi.org/project/gpt-suite/">
    <img src="https://img.shields.io/pypi/v/gpt-suite" alt="PyPI">
    <img src="https://img.shields.io/pypi/pyversions/gpt-suite" alt="Python 3">
</a>
</p>

This package is designed to make it easier to work with OpenAI's API by providing a set of tools that can be used to interact with the API in a more user-friendly way. It also supports multi-threading, which can be used to speed up the process of generating responses.

## Quick Start
To install the package from PyPI, run the following command:
```bash
pip install gpt-suite
```

## Building
To build the project from source, run the following command in the root directory of the project:
```bash
python3 -m build
```
Once the project is built, you can install the package by running the following command in the root directory of the project, replacing `x.x.x` with the version number:
```bash
pip install dist/gpt_suite-x.x.x-py3-none-any.whl
```