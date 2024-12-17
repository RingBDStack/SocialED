from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='SocialED',
    version='1.0.3',
    packages=find_packages(),
    author='beici',
    author_email='zhangkun23@buaa.edu.cn',
    description='A Python Library for Social Event Detection',
    install_requires=requirements,
    include_package_data=True,
    long_description=open('PyPI.rst', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RingBDStack/SocialED',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
