.. SocialED documentation master file, created by
   sphinx-quickstart on [日期].
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. figure:: SocialED.png
    :scale: 30%
    :alt: logo

----


.. raw:: html

   <div style="text-align: center;">

|badge_pypi| |badge_docs| |badge_stars| |badge_forks| |badge_downloads| |badge_testing| |badge_coverage| |badge_license| |badge_codeql| |badge_arxiv| 

.. |badge_pypi| image:: https://img.shields.io/pypi/v/socialed.svg?color=brightgreen
   :target: https://pypi.org/project/SocialED/
   :alt: PyPI version

.. |badge_docs| image:: https://readthedocs.org/projects/socialed/badge/?version=latest
   :target: https://socialed.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. |badge_stars| image:: https://img.shields.io/github/stars/RingBDStack/SocialED?style=flat
   :target: https://github.com/RingBDStack/SocialED/stargazers
   :alt: GitHub stars

.. |badge_forks| image:: https://img.shields.io/github/forks/RingBDStack/SocialED?style=flat
   :target: https://github.com/RingBDStack/SocialED/network
   :alt: GitHub forks

.. |badge_downloads| image:: https://static.pepy.tech/personalized-badge/SocialED?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :target: https://pepy.tech/project/SocialED
   :alt: PyPI downloads
   
.. |badge_testing| image:: https://github.com/ChenBeici/SocialED/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/ChenBeici/SocialED/actions/workflows/pytest.yml
   :alt: Testing

.. |badge_coverage| image:: https://coveralls.io/repos/github/ChenBeici/SocialED/badge.svg?branch=main
   :target: https://coveralls.io/github/ChenBeici/SocialED?branch=main
   :alt: Coverage Status

.. |badge_license| image:: https://img.shields.io/github/license/RingBDStack/SocialED.svg
   :target: https://github.com/RingBDStack/SocialED/blob/master/LICENSE
   :alt: License

.. |badge_codeql| image:: https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml/badge.svg
   :target: https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml
   :alt: CodeQL

.. |badge_arxiv| image:: https://img.shields.io/badge/cs.LG-2412.13472-b31b1b?logo=arxiv&logoColor=red
   :target: https://arxiv.org/abs/2412.13472
   :alt: arXiv

.. raw:: html

   </div>


-----


SocialED
========

A Python Library for Social Event Detection

The field of Social Event Detection represents a pivotal area of research within the broader domains of artificial 
intelligence and natural language processing. Its objective is the automated identification and analysis of events from 
social media platforms such as Twitter and Facebook. Such events encompass a wide range of occurrences, including natural 
disasters and viral phenomena.

SocialED is a comprehensive, open-source Python library designed to support social event detection (SED) tasks, integrating 19 detection algorithms and 14 diverse datasets. It provides a unified API with detailed documentation, offering researchers and practitioners a complete solution for event detection in social media. The library is built with modularity in mind, enabling users to adapt and extend components for various usages easily. SocialED supports a wide range of preprocessing techniques, such as graph construction and tokenization, and includes standardized interfaces for training models and making predictions. With its integration of popular deep learning frameworks, SocialED ensures high efficiency and scalability across CPU and GPU environments. Built adhering to high code quality standards, including unit testing, continuous integration, and code coverage, SocialED ensures robust, maintainable software.

Key Features
-----------------

* **Comprehensive Algorithm Collection**: Integrates 19 detection algorithms and supports 15 widely-used datasets, with continuous updates to include emerging methods
* **Unified API Design**: Implements algorithms with a consistent interface, allowing seamless data preparation and integration across all models
* **Modular Components**: Provides customizable components for each algorithm, enabling users to adjust models to specific needs
* **Rich Utility Functions**: Offers tools designed to simplify the construction of social event detection workflows
* **Robust Implementation**: Includes comprehensive documentation, examples, unit tests, and maintainability features




SocialED includes **19** social event detection algorithms.
For consistency and accessibility, SocialED is developed on top of `DGL <https://www.dgl.ai/>`_ 
and `PyTorch <https://pytorch.org/>`_, and follows the API design of `PyOD <https://github.com/yzhao062/pyod>`_ 
and `PyGOD <https://github.com/pygod-team/pygod>`_.
See examples below for detecting outliers with SocialED in 7 lines!



SocialED plays a crucial role in various downstream applications, including:

* Crisis management
* Public opinion monitoring
* Fake news detection
* And more...

**Social Event Detection Using SocialED with 5 Lines of Code**\ :



.. code-block:: python


    from SocialED.dataset import Event2012                 # Load the dataset
    dataset = Event2012()                                  # Load "Event2012" dataset
    
    from SocialED.detector import KPGNN                    # Import KPGNN model
    kpgnn = KPGNN(dataset, batch_size=200)                # Initialize KPGNN model
    
    kpgnn.preprocess()                                     # Preprocess data
    kpgnn.fit()                                           # Train the model
    pres, trus = kpgnn.detection()                        # Detect events
    
    kpgnn.evaluate(pres, trus)                            # Evaluate detection results

----


Implemented Algorithms
----------------------


===================  ==================  ===============  =============  ============  =====================================
Algorithm            Year                Backbone         Scenario       Supervision   Ref
===================  ==================  ===============  =============  ============  =====================================
LDA                  2003                Topic            Offline        Unsupervised  :class:`SocialED.detector.LDA`
BiLSTM               2005                Deep learning    Offline        Supervised    :class:`SocialED.detector.BiLSTM`
Word2Vec             2013                Word embeddings  Offline        Unsupervised  :class:`SocialED.detector.Word2Vec`
GloVe                2014                Word embeddings  Offline        Unsupervised  :class:`SocialED.detector.GloVe`
WMD                  2015                Similarity       Offline        Unsupervised  :class:`SocialED.detector.WMD`
BERT                 2018                PLMs             Offline        Unsupervised  :class:`SocialED.detector.BERT`
SBERT                2019                PLMs             Offline        Unsupervised  :class:`SocialED.detector.SBERT`
EventX               2020                Community        Offline        Unsupervised  :class:`SocialED.detector.EventX`
CLKD                 2021                GNNs             Online         Supervised    :class:`SocialED.detector.CLKD`
KPGNN                2021                GNNs             Online         Supervised    :class:`SocialED.detector.KPGNN`
FinEvent             2022                GNNs             Online         Supervised    :class:`SocialED.detector.FinEvent`
QSGNN                2022                GNNs             Online         Supervised    :class:`SocialED.detector.QSGNN`
ETGNN                2023                GNNs             Offline        Supervised    :class:`SocialED.detector.ETGNN`
HCRC                 2023                GNNs             Online         Unsupervised  :class:`SocialED.detector.HCRC`
UCLSED               2023                GNNs             Offline        Supervised    :class:`SocialED.detector.UCLSED`
RPLMSED              2024                PLMs             Online         Supervised    :class:`SocialED.detector.RPLMSED`
HISEvent             2024                Community        Online         Unsupervised  :class:`SocialED.detector.HISEvent`
ADPSEMEvent          2024                Community        Online         Unsupervised  :class:`SocialED.detector.ADPSEMEvent`
HyperSED             2025                Community        Online         Unsupervised  :class:`SocialED.detector.HyperSED`
===================  ==================  ===============  =============  ============  =====================================


Modular Design and Utility Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SocialED is built with a modular design to improve reusability and reduce redundancy. It organizes social event detection into distinct modules:

* ``preprocessing``
* ``modeling``
* ``evaluation``


The library provides several utility functions including:

* ``utils.tokenize_text`` and ``utils.construct_graph`` for data preprocessing
* ``metric`` for evaluation metrics

Library Robustness and Accessibility
------------------------------------

Quality and Reliability
^^^^^^^^^^^^^^^^^^^^^^^

* Built with robustness and high-quality standards
* Continuous integration through GitHub Actions
* Automated testing across Python versions and operating systems
* >99% code coverage
* PyPI-compatible and PEP 625 compliant
* Follows PEP 8 style guide

Accessibility and Community Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Detailed API documentation on Read the Docs
* Step-by-step guides and tutorials
* Intuitive API design inspired by scikit-learn
* Open-source project hosted on GitHub
* Easy issue-reporting mechanism
* Clear contribution guidelines


----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API References

   SocialED.dataset
   SocialED.detector
   SocialED.metrics
   SocialED.utils    
   SocialED.dataprocess




.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   cite
   team
   reference
