from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='nn',
    version='0.0.1',
    description='A neural network library with a high-level API built on top of TensorFlow.',
    long_description=long_description,
    author='Ravindra Marella',
    author_email='mv.ravindra007@gmail.com',
    url='https://marella.github.io/nn/',
    license='MIT',
    packages=['nn'],
    install_requires=['tensorflow'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='nn tensorflow neural networks deep learning machine learning artificial intelligence ml ai',
)
