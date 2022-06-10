from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(name='vamtoolbox',
      version=get_version("vamtoolbox/__init__.py"),
      description='',
      url='https://github.com/computed-axial-lithography/VAMToolbox',
      author='Joseph Toombs',
      author_email='jtoombs@berkeley.edu',
      license='gpl-3.0',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
