import setuptools

requirements = [
    "torch~=1.10.0",
    "torch-tb-profiler~=0.3.1",
    "torchvision~=0.11.1",
    "theano~=1.0.5",
]

# NOTE: Suggest Just Using the Habana / Gaudi Docker Container...
habana_requirements = []
setup_requirements = []

extra_requirements = {
    "setup": setup_requirements,
    "habana": [
        *habana_requirements,
    ],
}

setuptools.setup(
    name="msls",
    version="0.0.1",
    author="Dustin Wilson",
    author_email="<dmw2151 [at] columbia [dot] edu>",
    description="...",
    project_urls={
        "Bug Tracker": "https://github.com/DMW2151/msls-pytorch-dcgan/issues",
        "Documentation": "https://github.com/DMW2151/msls-pytorch-dcgan/readme.md",
        "Source": "https://github.com/DMW2151/msls-pytorch-dcgan/model/msls",
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6, <4",
    extras_require=extra_requirements,
)
