from setuptools import setup, find_packages


setup(name='MMT',
      version='0.1.0',
      description='PyTorch implementation for the ICLR-2020 submission Mutual Mean-Teaching',
      author='Anonymous',
      author_email='Anonymous',
      url='https://github.com/Pre-release/MMT.git',
      install_requires=[
          'numpy', 'torch==1.1.0', 'torchvision==0.2.1', 
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification',
          'Deep Learning',
      ])
