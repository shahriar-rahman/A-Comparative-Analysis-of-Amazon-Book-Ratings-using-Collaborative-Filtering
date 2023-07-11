from setuptools import find_packages, setup

setup(
    name='amazon_ratings',
    packages=find_packages(),
    version='0.8.0',
    description='Juxtaposing different Recommender Algorithms by utilizing the concept of Collaborative Filtering to analyze the Amazon Book Ratings. '
                'the Audible website.',
    author='Shahriar Rahman',
    license='MIT License',
    author_email='shahriarrahman1101@gmail.com',
    install_requires=[
        'pandas',
        'missingNo',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scikit-surprise',
        'plotly',
        'numpy',
    ],
