from setuptools import setup, find_packages

setup(
    name="State of the Artefact",
    version="0.1.0",
    url="https://github.com/maxvstheuniverse/state-of-the-artefact.git",
    author="Max Peeperkorn",
    author_email="max.peeperkorn@protonmail.com",
    description="TODO",
    packages=find_packages("src"),
    package_dir={'': 'src'},
    install_requires=[],
    entry_points={
        "console_scripts": [
            "sota=state_of_the_artefact.__main__:main",
            "init=state_of_the_artefact.__main__:init"
        ]
    }
)
