from setuptools import setup, find_packages

setup(
    name='senseis',
    version='0.0.1',
    description='Recon Chess RL code repo',
    authors='senseis',
    author_emails='armand.li@hotmail.com',
    url='https://github.com/armandli/ReconChessRL',
    packages=find_packages(exclude=['senseis']),
    package_data={},
    data_files={},
    install_requires=[
        'reconchess',
        'torch',
        'torch-optimizer',
    ],
    entry_points={
        'console_scripts':[]
    },
    scripts=[]
)
