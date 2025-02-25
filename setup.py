from setuptools import setup, find_packages

setup(
    name="kai_bert",
    version="1.0.0",
    use_scm_version=True,
    setup_requires=["setuptools>=42", "setuptools_scm"],
    description="A BERT-based Masked Language Model implementation using PyTorch Lightning",
    author="Khairi Abidi",
    author_email="khairi.abidi@majesteye.com",
    url="https://github.com/abidikhairi/kai-bert",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.10",
        "transformers>=4.12",
        "torchmetrics>=0.6",
        "pytorch-lightning>=1.6",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
