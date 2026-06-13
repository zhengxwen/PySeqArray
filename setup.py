from distutils.core import setup, Extension
import numpy
import os
import pygds


src_fnlst = [ os.path.join('src', fn) for fn in [
	'GetData.cpp', 'Index.cpp', 'Methods.cpp', 'ReadByVariant.cpp',
	'PySeqArray.cpp', 'LinkGDS.c', 'vectorization.cpp' ] ]


setup(name='PySeqArray',
	version = '0.1.0',
	description = 'Python Interface to SeqArray Files for Data Management of Whole-Genome Sequence Variant Calls',
	url = 'http://github.com/CoreArray/PySeqArray',
	author = 'Xiuwen Zheng',
	author_email = 'zhengxwen@gmail.com',
	license = 'GPLv3',
	packages = [ 'PySeqArray' ],
	install_requires = [ 'numpy', 'pygds' ], # 'multiprocessing' ],
	ext_modules = [ Extension('PySeqArray.ccall',
		src_fnlst,
		include_dirs = [ pygds.get_include(), numpy.get_include() ],
		define_macros = [ ('USING_PYTHON', None) ],
	) ],
	package_data = {
		'PySeqArray': [ 'data/*.gds' ]
	}
)
