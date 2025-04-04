# Configuration file for the Sphinx documentation builder.

import os
import sys
from os.path import dirname, abspath

# 将 SocialED 项目的根目录添加到 sys.path
sys.path.insert(0, abspath('../'))
root_dir = dirname(dirname(abspath(__file__)))

class MockSpacy:
    def load(self, *args, **kwargs):
        return self
    
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self

sys.modules['spacy'] = MockSpacy()
sys.modules['en_core_web_lg'] = MockSpacy()


# -- Project information -----------------------------------------------------

project = 'SocialED'
copyright = '2024 beici'
author = 'beici'

# 如果您有版本信息，可以使用以下代码获取版本号
#version_path = os.path.join(root_dir, 'SocialED', 'version.py')
#exec(open(version_path).read())
version = '1.1.5'
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex'
]

bibtex_bibfiles = ['zreferences.bib']

autodoc_member_order = 'bysource'

autodoc_mock_imports = [
    # 'en_core_web_lg',
    # 'fr_core_news_lg',
    # 'dgl',
    # 'dgl.function',
    # 'dgl.dataloading',
    # 'spacy',
    # 'torch',
    # 'torch.nn',
    # 'torch.cuda',
    # 'transformers',
    # 'transformers.modeling_bert',
    # 'transformers.tokenization_bert',
    # 'numpy',
    # 'pandas',
    # 'scikit-learn'
    'numpy',
    'torch',
    'pandas',
    'sklearn',
    'scipy',
    'networkx',
    'spacy',
    'thinc',
    'gensim',
    'transformers',
    'nltk',
    'tensorflow',
    'keras',
    'matplotlib',
    'seaborn',
    'tqdm',
    'cupy',
    'dgl',
    'torch_geometric',
    'torch_scatter',
    'torch_sparse',
    'torch_cluster',
    'torch_spline_conv',
    'tokenizers',
    'sentence_transformers',
    'allennlp',
    'overrides',
    'faiss',
    'numba',
    'cudf',
    'cugraph',
    'cucim',
    'en_core_web_lg',
    'ignite',
    'ignite.distributed',
    'ignite.distributed.auto',
    'ignite.distributed.utils',
    'ignite.distributed.comp_models',
    'ignite.engine',
    'ignite.metrics',
    'packaging',
    'packaging.version',
    'hdbscan',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_favicon = 'socialed.ico'
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = 'socialEDdoc'

# -- Options for LaTeX output ------------------------------------------------
latex_documents = [
    (master_doc, 'socialED.tex', 'SocialED Documentation',
     'beici', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'socialED', 'SocialED Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'socialED', 'SocialED Documentation',
     author, 'SocialED', 'A Python library for social event detection.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------
# from sphinx_gallery.sorting import FileNameSortKey

html_static_path = []



# sphinx_gallery_conf = {
#     'examples_dirs': 'examples/',   # Path to your example scripts
#     'gallery_dirs': 'tutorials/',
#     'within_subsection_order': FileNameSortKey,
#     'filename_pattern': '.py',
#     'download_all_examples': False,
# }

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    'torch': ("https://pytorch.org/docs/master", None),
    'torch_geometric': ("https://pytorch-geometric.readthedocs.io/en/latest", None),
}


# 添加类型提示支持
set_type_checking_flag = True
always_document_param_types = True

