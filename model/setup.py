import setuptools


with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []
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
    package_dir={"": "msls"},
    packages=setuptools.find_packages(where="msls"),
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.7, <4",
    extras_require=extra_requirements,
)
