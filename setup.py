import setuptools

setuptools.setup(
    name="sharetools",
    version="1.0.0",
    author="Anders Rasmussen",
    author_email="arasmussen@flatironinstitute.org",
    description="Sharetools stuff",
    url="",
    packages=['sharetools'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    scripts=['bin/sharetools'],

    install_requires=[
        'matplotlib',
        'numpy',
        'keras',
        'bioseq',
        'alive_progress',
        'pandas',
        'halo',
        'tensorflow',
    ],

)
