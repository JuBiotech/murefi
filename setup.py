# Copyright 2020 Forschungszentrum Jülich GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import setuptools

__packagename__ = 'murefi'

def get_version():
    import pathlib, re
    VERSIONFILE = pathlib.Path(pathlib.Path(__file__).parent, __packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(), # this must be the same as the name above
        version=__version__,
        description='Package for multiple replicate fitting and Bayesian modeling.',
        url='https://github.com/JuBiotech/murefi',
        author='Laura Marie Helleckes, Michael Osthege',
        author_email='l.helleckes@fz-juelich.de, m.osthege@fz-juelich.de',
        license='GNU Affero General Public License v3',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
        ],
        install_requires=[
            'pandas',
            'numpy',
            'scipy',
            'h5py',
            'calibr8>=6.0.0'
        ]
)
