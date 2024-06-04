from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='oplas',
    version='0.0.1',
    url = 'https://github.com/drscotthawley/oplas',
    license='MIT',
    author='Scott H. Hawley',
    author_email='scott.hawley@belmont.edu', 
    description='Operational Latent Spaces (OpLaS)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops',
        'stempeg',
        'tqdm',
        'wandb',
        'laion-clap',
        'aeiou',
        'matplotlib',
    ],
)
