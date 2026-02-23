from setuptools import setup, find_packages

setup(
    name="ufc_phase_transition",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
    ],
)
