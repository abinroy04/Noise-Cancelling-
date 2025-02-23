from setuptools import setup, find_packages

setup(
    name="noise_removal_abin",
    version="0.1",
    description="A package for noise removal and speech enhancement",
    author="Abin Roy",
    packages=find_packages(),
    install_requires=[
        "pydub",
        "librosa",
        "soundfile",
        "noisereduce",
        "speechbrain",
        "scipy",
    ],
)
