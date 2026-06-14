PySeqArray: data manipulation of whole-genome sequencing variants with SeqArray files in Python
===

[![CI](https://github.com/CoreArray/PySeqArray/actions/workflows/ci.yml/badge.svg)](https://github.com/CoreArray/PySeqArray/actions/workflows/ci.yml)

![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)
[GNU General Public License, GPLv3](http://www.gnu.org/copyleft/gpl.html) (2017)

[![Build Status](https://travis-ci.org/CoreArray/PySeqArray.png)](https://travis-ci.org/CoreArray/PySeqArray)

pre-release version: v0.1


## Features

Data management of whole-genome sequence variant calls with thousands of individuals: genotypic data (e.g., SNVs, indels and structural variation calls) and annotations in SeqArray files are stored in an array-oriented and compressed manner, with efficient data access using the Python programming language.

The SeqArray format is built on top of Genomic Data Structure (GDS) data format, and defines required data structure. GDS is a flexible and portable data container with hierarchical structure to store multiple scalable array-oriented data sets. It is suited for large-scale datasets, especially for data which are much larger than the available random-access memory. It also offers the efficient operations specifically designed for integers of less than 8 bits, since a diploid genotype usually occupies fewer bits than a byte. Data compression and decompression are available with relatively efficient random access.


## Prerequisites

Python 2 (2.6-2.7), and Python 3 (3.3-3.6)

NumPy 1.6.0 or later

[pygds](https://github.com/CoreArray/pygds)


## Installation

```sh
## require the pygds package
pip install git+git://github.com/CoreArray/pygds.git
## install PySeqArray
pip install git+git://github.com/CoreArray/PySeqArray.git
```


## Citation

#### Original paper (implemented in an [R/Bioconductor](http://bioconductor.org/packages/SeqArray) package):

[SeqArray](http://bioconductor.org/packages/SeqArray)

Zheng X, Gogarten S, Lawrence M, Stilp A, Conomos M, Weir BS, Laurie C, Levine D (2017). SeqArray -- A storage-efficient high-performance data format for WGS variant calls. *Bioinformatics*. [DOI: 10.1093/bioinformatics/btx145](http://dx.doi.org/10.1093/bioinformatics/btx145).



## SeqArray File Download

* [1000 Genomes Project](http://bochet.gcc.biostat.washington.edu/seqarray/1000genomes)


## Examples

```python
import PySeqArray as ps

fn = ps.seqExample('1KG_phase1_release_v3_chr22.gds')
f = ps.SeqArrayFile()
f.open(fn)
f.show()
f.close()
```

```
File: PySeqArray/data/1KG_phase1_release_v3_chr22.gds (1.1M)
+    [  ] *
|--+ description   [  ] *
|--+ sample.id   { Str8 1092 LZMA_ra(10.5%), 914B } *
|--+ variant.id   { Int32 19773 LZMA_ra(8.39%), 6.5K } *
|--+ position   { Int32 19773 LZMA_ra(52.0%), 40.1K } *
|--+ chromosome   { Str8 19773 LZMA_ra(0.28%), 166B } *
|--+ allele   { Str8 19773 LZMA_ra(22.7%), 109.2K } *
|--+ genotype   [  ] *
|  |--+ data   { Bit2 19773x1092x2 LZMA_ra(8.17%), 861.8K } *
|  |--+ extra.index   { Int32 0x3 LZMA_ra, 19B } *
|  \--+ extra   { Int16 0 LZMA_ra, 19B }
|--+ phase   [  ]
|  |--+ data   { Bit1 19773x1092 LZMA_ra(0.02%), 550B } *
|  |--+ extra.index   { Int32 0x3 LZMA_ra, 19B } *
|  \--+ extra   { Bit1 0 LZMA_ra, 19B }
|--+ annotation   [  ]
|  |--+ id   { Str8 19773 LZMA_ra(35.2%), 75.2K } *
|  |--+ qual   { Float32 19773 LZMA_ra(3.62%), 2.8K } *
|  |--+ filter   { Int32,factor 19773 LZMA_ra(0.21%), 170B } *
|  |--+ info   [  ]
|  \--+ format   [  ]
\--+ sample.annotation   [  ]
   |--+ Family.ID   { Str8 1092 LZMA_ra(15.3%), 1.1K }
   |--+ Population   { Str8 1092 LZMA_ra(5.08%), 222B }
   |--+ Gender   { Str8 1092 LZMA_ra(5.85%), 386B }
   \--+ Ancestry   { Str8 1092 LZMA_ra(2.43%), 233B }
```


### More examples

Python tutorial with SeqArray files: [docs/demo/tutorial.ipynb](docs/demo/tutorial.ipynb)

Python tutorial with multiprocessing: [docs/demo/tutorial_parallel.ipynb](docs/demo/tutorial_parallel.ipynb)

