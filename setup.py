try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='auto_cpca',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="Automatic Contrastive Principle Components Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lucas Plagwitz',
    author_email='lucas.plagwitz@uni-muenster.de',
    download_url="https://github.com/lucasplagwitz/auto_cpca/archive/' + __version__ + '.tar.gz",
    url= "https://github.com/lucasplagwitz/auto_cpca.git",
    keywords=['machine learning', 'dimensional reduction', 'supervised learning'],
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        ]
)