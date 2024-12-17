.. image:: https://github.com/RingBDStack/SocialED/blob/main/docs/SocialED.png?raw=true
   :target: https://github.com/RingBDStack/SocialED/blob/main/docs/SocialED.png?raw=true
   :width: 1050
   :alt: SocialED Logo
   :align: center

|badge_pypi| |badge_docs| |badge_stars| |badge_forks| |badge_downloads| |badge_testing| |badge_coverage| |badge_license|

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
   
.. |badge_testing| image:: https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/ChenBeici/SocialED/actions/workflows/testing.yml
   :alt: testing

.. |badge_coverage| image:: https://coveralls.io/repos/github/pygod-team/pygod/badge.svg?branch=main
   :target: https://coveralls.io/github/pygod-team/pygod?branch=main
   :alt: Coverage Status

.. |badge_license| image:: https://img.shields.io/github/license/RingBDStack/SocialED.svg
   :target: https://github.com/RingBDStack/SocialED/blob/master/LICENSE
   :alt: License

.. |badge_codeql| image:: https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml/badge.svg
   :target: https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml
   :alt: CodeQL

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
-----------

* **Comprehensive Algorithm Collection**: Integrates 19 detection algorithms and supports 14 widely-used datasets, with continuous updates to include emerging methods
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


.. image:: https://github.com/RingBDStack/SocialED/blob/main/docs/API.png?raw=true
   :target: https://github.com/RingBDStack/SocialED/blob/main/docs/API.png?raw=true
   :width: 1050
   :alt: SocialED API
   :align: center





Folder Structure
================

::

   .
   ├── LICENSE
   ├── MANIFEST.in
   ├── README.rst
   ├── docs
   ├── SocialED
   │   ├── __init__.py
   │   ├── datasets    
   │   ├── detector  
   │   └── metrics  
   ├── requirements.txt
   ├── setup.cfg
   └── setup.py


Installation
------------



It is recommended to use **pip** for installation.
Please make sure **the latest version** is installed, as PyGOD is updated frequently:

.. code-block:: bash

   pip install SocialED           # normal install
   pip install --upgrade SocialED  # or update if needed


Alternatively, you could clone and run setup.py file:

.. code-block:: bash

    # Set up the environment
    conda create -n SocialED python=3.8
    conda activate SocialED

    # Installation
    git clone https://github.com/RingBDStack/SocialED.git
    cd SocialED
    pip install -r requirements.txt
    pip install .

**Required Dependencies**\ :

* python>=3.8
* numpy>=1.24.3
* scikit-learn>=1.2.2
* scipy>=1.10.1
* networkx>=2.3
* torch>=2.3.0
* torch_geometric>=2.5.3
* dgl>=0.6.0


Collected Algorithms
--------------------

The library integrates methods ranging from classic approaches like LDA and BiLSTM to specialized techniques such as KPGNN, QSGNN, FinEvent, and HISEvent. Despite significant advancements in detection methods, deploying these approaches or conducting comprehensive evaluations has remained challenging due to the absence of a unified framework. SocialED addresses this gap by providing a standardized platform for researchers and practitioners in the SED field.

Implemented Algorithms
----------------------


- **LDA**: Latent Dirichlet Allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups. It is particularly useful for discovering the hidden thematic structure in large text corpora.
- **BiLSTM**: Bi-directional Long Short-Term Memory (BiLSTM) networks enhance the capabilities of traditional LSTMs by processing sequences in both forward and backward directions. This bidirectional approach is effective for tasks like sequence classification and time series prediction.
- **Word2Vec**: Word2Vec is a family of models that generate word embeddings by training shallow neural networks to predict the context of words. These embeddings capture semantic relationships between words, making them useful for various natural language processing tasks.
- **GLOVE**: Global Vectors for Word Representation (GLOVE) generates word embeddings by aggregating global word-word co-occurrence statistics from a corpus. This approach produces vectors that capture meaning effectively, based on the frequency of word pairs in the training text.
- **WMD**: Word Mover's Distance (WMD) measures the semantic distance between two documents by computing the minimum distance that words from one document need to travel to match words from another document. This method is grounded in the concept of word embeddings.
- **BERT**: Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based model that pre-trains deep bidirectional representations by conditioning on both left and right context in all layers. BERT has achieved state-of-the-art results in many NLP tasks.
- **SBERT**: Sentence-BERT (SBERT) modifies the BERT network to generate semantically meaningful sentence embeddings that can be compared using cosine similarity. It is particularly useful for sentence clustering and semantic search.
- **EventX**: EventX is designed for online event detection in social media streams, processing tweets in real-time to identify emerging events by clustering similar content. This framework is optimized for high-speed data environments.
- **CLKD**: cross-lingual knowledge distillation (CLKD) combines a convolutional neural network with dynamic time warping to align sequences and detect events in streaming data. This online algorithm is effective for real-time applications.
- **KPGNN**: Knowledge-Preserving Graph Neural Network (KPGNN) is designed for incremental social event detection. It utilizes rich semantics and structural information in social messages to continuously detect events and extend its knowledge base. KPGNN outperforms baseline models, with potential for future research in event analysis and causal discovery in social data.
- **Finevent**: Fine-Grained Event Detection (FinEvent) uses a reinforced, incremental, and cross-lingual architecture for social event detection. It employs multi-agent reinforcement learning and density-based clustering (DRL-DBSCAN) to improve performance in various detection tasks. Future work will extend RL-guided GNNs for event correlation and evolution.
- **QSGNN**: Quality-Aware Self-Improving Graph Neural Network (QSGNN) improves open set social event detection with a pairwise loss and orthogonal constraint for training. It uses similarity distributions for pseudo labels and a quality-aware strategy to reduce noise, achieving state-of-the-art results in both closed and open set scenarios.
- **ETGNN**: Evidential Temporal-aware Graph Neural Network (ETGNN) enhances social event detection by integrating uncertainty and temporal information using Evidential Deep Learning and Dempster-Shafer theory. It employs a novel temporal-aware GNN aggregator, outperforming other methods.
- **HCRC**: Hybrid Graph Contrastive Learning for Social Event Detection (HCRC) captures comprehensive semantic and structural information from social messages. Using hybrid graph contrastive learning and reinforced incremental clustering, HCRC outperforms baselines across various experimental settings.
- **UCLSED**: Uncertainty-Guided Class Imbalance Learning Framework (UCLSED) enhances model generalization in imbalanced social event detection tasks. It uses an uncertainty-guided contrastive learning loss to handle uncertain classes and combines multi-view architectures with Dempster-Shafer theory for robust uncertainty estimation, achieving superior results.
- **RPLMSED**: Relational Prompt-Based Pre-Trained Language Models for Social Event Detection (RPLMSED) uses pairwise message modeling to address missing and noisy edges in social message graphs. It leverages content and structural information with a clustering constraint to enhance message representation, achieving state-of-the-art performance in various detection tasks.
- **HISevent**: Structural Entropy-Based Social Event Detection (HISevent) is an unsupervised tool that explores message correlations without the need for labeling or predetermining the number of events. HISevent combines GNN-based methods' advantages with efficient and robust performance, achieving new state-of-the-art results in closed- and open-set settings.
- **ADPSEMEvent**: Adaptive Differential Privacy Social Event Message Event Detection (ADPSEMEvent) is an unsupervised framework that prioritizes privacy while detecting social events. It uses a two-stage approach: first constructing a private message graph using adaptive differential privacy to maximize privacy budget usage, then applying a novel 2-dimensional structural entropy minimization algorithm for event detection. This method effectively balances privacy protection with data utility in open-world settings.



SocialED implements the following algorithms:
==================  ===============  ================    ============  ==============  =========================
     Algorithm      |      Year      |    Category       |  Environment  |  Supervision   |            Ref
==================  ===============  ================    ============  ==============  =========================
        LDA         |      2003      |       Topic       |    Offline    | Unsupervised    |  [#Blei2003lda]_
      BiLSTM        |      2005      |  Deep learning    |    Offline    | Unsupervised    |  [#Graves2005bilstm]_
     Word2Vec       |      2013      | Word embeddings   |    Offline    | Unsupervised    |  [#Mikolov2013word2vec]_
       GloVe        |      2014      | Word embeddings   |    Offline    | Unsupervised    |  [#Pennington2014glove]_
        WMD         |      2015      |    Similarity     |    Offline    | Unsupervised    |  [#Kusner2015wmd]_
       BERT         |      2018      |       PLMs        |    Offline    | Unsupervised    |  [#Devlin2018bert]_
      SBERT         |      2019      |       PLMs        |    Offline    | Unsupervised    |  [#Reimers2019sbert]_
      EventX        |      2020      | Community detection |  Offline    | Unsupervised    |  [#Liu2020eventx]_
       CLKD         |      2021      |       GNNs        |    Online     |   Supervised    |  [#Ren2021clkd]_
      KPGNN         |      2021      |       GNNs        |    Online     |   Supervised    |  [#Cao2021kpgnn]_
     FinEvent       |      2022      |       GNNs        |    Online     |   Supervised    |  [#Peng2022finevent]_
      QSGNN         |      2022      |       GNNs        |    Online     |   Supervised    |  [#Ren2022qsgnn]_
      ETGNN         |      2023      |       GNNs        |    Offline    | Unsupervised    |  [#Ren2023etgnn]_
       HCRC         |      2023      |       GNNs        |    Online     | Unsupervised    |  [#Guo2023hcrc]_
      UCLSED        |      2023      |       GNNs        |    Offline    | Unsupervised    |  [#Ren2023uclsad]_
     RPLMSED        |      2024      |       PLMs        |    Online     |   Supervised    |  [#Li2024rplmsed]_
     HISEvent       |      2024      | Community detection |  Online     | Unsupervised    |  [#Cao2024hisevent]_
   ADPSEMEvent      |      2024      | Community detection |  Online     | Unsupervised    |  [#Yang2024adpsemevent]_
     HyperSED       |      2025      | Community detection |  Online     | Unsupervised    |  [#Yu2025hypersed]_
==================  ===============  ================  ============  ==============  =========================




Supported Datasets
^^^^^^^^^^^^^^^^^


-   **Event2012**: Events2012 dataset contains 68,841 annotated English tweets covering 503 different event categories, encompassing tweets over a consecutive 29-day period.
-   **Event2018**: Events2018 includes 64,516 annotated French tweets covering 257 different event categories, with data spanning over a consecutive 23-day period.
-   **Arabic_Twitter**: Arabic-Twitter dataset comprises 9,070 annotated Arabic tweets, covering seven catastrophic-class events from various periods.
-   **MAVEN**: MAVEN contains 10,242 annotated English texts covering 164 different event types. It is designed to facilitate the development of robust event detection models across a wide variety of domains.
-   **CrisisLexT26**: CrisisLexT26 consists of 27,933 tweets related to 26 different crisis events. The dataset is used to study information dissemination and event detection in social media during emergencies.
-   **CrisisLexT6**: CrisisLexT6 contains 60,082 tweets focused on 6 major crisis events. It provides valuable insights into public response and information spread during crises through annotated social media data.
-   **CrisisMMD**: CrisisMMD includes 18,082 manually annotated tweets collected during 7 major natural disasters in 2017, including earthquakes, hurricanes, wildfires, and floods from different parts of the world.
-   **CrisisNLP**: CrisisNLP comprises 25,976 crisis-related tweets covering 11 different events. The dataset includes human-labeled tweets, dictionaries, word embeddings and related tools for crisis information analysis.
-   **HumAID**: HumAID contains 76,484 manually annotated tweets collected during 19 major natural disaster events from 2016 to 2019, including earthquakes, hurricanes, wildfires, and floods across different regions.
-   **Mix_data**: A combined dataset containing multiple crisis-related tweet collections:
    - **ICWSM2018**: 21,571 human-labeled tweets from the 2015 Nepal earthquake and 2013 Queensland floods
    - **ISCRAM2013**: 4,676 labeled tweets from the 2011 Joplin tornado  
    - **ISCRAM2018**: 49,804 tweets from Hurricanes Harvey, Irma, and Maria in 2017
    - **BigCrisisData**: 2,438 tweets with crisis-related classifications
-   **KBP**: KBP contains 85,569 texts covering 100 different event types. It focuses on extracting structured event information and serves as a benchmark dataset for information extraction systems.
-   **Event2012_100**: Event2012_100 contains 100 events with a total of 15,019 tweets, where the maximal event comprises 2,377 tweets, and the minimally has 55 tweets, with an imbalance ratio of approximately 43.
-   **Event2018_100**: Event2018_100 contains 100 events with a total of 19,944 tweets, where the maximal event comprises 4,189 tweets and the minimally has 27 tweets, an imbalance ratio of approximately 155.
-   **Arabic_100**: Arabic_100 contains 100 events with a total of 3,022 tweets, where the maximal event comprises 312 tweets and the minimally has 7 tweets, with an imbalance ratio of approximately 44.


Dataset
-------

===================  ==================  ======================================  =============  ==============  =============
Dataset              Subset              Long tail                               Language       Events          Texts
===================  ==================  ======================================  =============  ==============  =============
Event2012                                No                                      English        503             68,841
Event2018                                No                                      French         257             64,516
Arabic_Twitter                           No                                      Arabic         7               9,070
MAVEN                                    No                                      English        164             10,242
CrisisLexT26                             No                                      English        26              27,933
CrisisLexT6                              No                                      English        6               60,082
CrisisMMD                                No                                      English        7               18,082
CrisisNLP                                No                                      English        11              25,976
HumAID                                   No                                      English        19              76,484
Mix_data             ICWSM2018           No                                      English        5               21,571
                     ISCRAM2013                                                  English                        4,676
                     ISCRAM2018                                                  English                        49,804
                     BigCrisisData                                               English                        2,438
KBP                                      No                                      English        100             85,569
Event2012_100                            Yes                                     English        100             15,019
Event2018_100                            Yes                                     French         100             19,944
Arabic_100                               Yes                                     Arabic         7               3,022
===================  ==================  ======================================  =============  ==============  =============


Library Design and Implementation
-------------------------------

Dependencies and Technology Stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SocialED is compatible with Python 3.8 and above, and leverages well-established deep learning frameworks like PyTorch and Hugging Face Transformers for efficient model training and inference, supporting both CPU and GPU environments. In addition to these core frameworks, SocialED also integrates NumPy, SciPy, and scikit-learn for data manipulation, numerical operations, and machine learning tasks, ensuring versatility and performance across a range of workflows.

Unified API Design
^^^^^^^^^^^^^^^

Inspired by the API designs of established frameworks, we developed a unified API for all detection algorithms in SocialED:

1. ``preprocess`` provides a flexible framework for handling various preprocessing tasks, such as graph construction and tokenization
2. ``fit`` trains the detection algorithms on the preprocessed data, adjusting model parameters and generating necessary statistics for predictions
3. ``detection`` uses the trained model to identify events from the input data, returning the detected events
4. ``evaluate`` assesses the performance of the detection results by comparing predictions with ground truth data, providing metrics like precision, recall and F1-score

Example Usage
^^^^^^^^^^^^

.. code-block:: python

    from SocialED.dataset import MAVEN                 # Load the dataset
    dataset = MAVEN().load_data()   # Load "arabic_twitter" dataset
    
    from SocialED.detector import KPGNN        # Import KPGNN model
    args = args_define().args                  # Get training arguments
    kpgnn = KPGNN(args, dataset)              # Initialize KPGNN model
    
    kpgnn.preprocess()                        # Preprocess data
    kpgnn.fit()                               # Train the model
    pres, trus = kpgnn.detection()            # Detect events
    kpgnn.evaluate(pres, trus)                # Evaluate detection results

Modular Design and Utility Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SocialED is built with a modular design to improve reusability and reduce redundancy. It organizes social event detection into distinct modules:

* ``preprocessing``
* ``modeling``
* ``evaluation``


The library provides several utility functions including:

* ``utils.tokenize_text`` and ``utils.construct_graph`` for data preprocessing
* ``metric`` for evaluation metrics
* ``utils.load_data`` for built-in datasets

Library Robustness and Accessibility
----------------------------------

Quality and Reliability
^^^^^^^^^^^^^^^^^^^^

* Built with robustness and high-quality standards
* Continuous integration through GitHub Actions
* Automated testing across Python versions and operating systems
* >99% code coverage
* PyPI-compatible and PEP 625 compliant
* Follows PEP 8 style guide

Accessibility and Community Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Detailed API documentation on Read the Docs
* Step-by-step guides and tutorials
* Intuitive API design inspired by scikit-learn
* Open-source project hosted on GitHub
* Easy issue-reporting mechanism
* Clear contribution guidelines

Future Development Plans
----------------------

1. **Expanding Algorithms and Datasets**
   * Integrating advanced algorithms
   * Expanding datasets across languages, fields, and cultures

2. **Enhancing Intelligent Functions**
   * Automated machine learning for model selection
   * Hyperparameter optimization

3. **Supporting Real-time Detection**
   * Enhanced real-time event detection
   * Trend analysis capabilities
   * Support for streaming data




References
----------
.. [#Blei2003lda] Blei, D.M., Ng, A.Y., and Jordan, M.I., 2003. Latent Dirichlet allocation. Journal of Machine Learning Research, 3(Jan), pp. 993-1022.

.. [#Graves2005bilstm] Graves, A., and Schmidhuber, J., 2005. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5-6), pp. 602-610. Elsevier.

.. [#Mikolov2013word2vec] Mikolov, T., Chen, K., Corrado, G., and Dean, J., 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

.. [#Pennington2014glove] Pennington, J., Socher, R., and Manning, C.D., 2014. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1532-1543. Association for Computational Linguistics.

.. [#Kusner2015wmd] Kusner, M., Sun, Y., Kolkin, N., and Weinberger, K., 2015. From word embeddings to document distances. In International Conference on Machine Learning, pp. 957-966. PMLR.

.. [#Devlin2018bert] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K., 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

.. [#Reimers2019sbert] Reimers, N., and Gurevych, I., 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3980-3990. Association for Computational Linguistics.

.. [#Liu2020eventx] Liu, B., Han, F.X., Niu, D., Kong, L., Lai, K., and Xu, Y., 2020. Story forest: Extracting events and telling stories from breaking news. ACM Transactions on Knowledge Discovery from Data (TKDD), 14(3), pp. 1-28. ACM New York, NY, USA.

.. [#Ren2021clkd] Ren, J., Peng, H., Jiang, L., Wu, J., Tong, Y., Wang, L., Bai, X., Wang, B., and Yang, Q., 2021. Transferring knowledge distillation for multilingual social event detection. arXiv preprint arXiv:2108.03084.

.. [#Cui2021mvgan] Cui, W., Zhang, Y., Liu, Z., and Yu, P.S., 2021. MVGAN: A Multi-view Graph Generative Adversarial Network for Anomaly Detection. In Proceedings of the 2021 IEEE International Conference on Big Data (Big Data), pp. 4513-4522. IEEE.

.. [#Peng2021ppgcn] Peng, H., Wu, J., Cao, Y., Dou, Y., Li, J., and Yu, P.S., 2021. PP-GCN: Privacy-Preserving Graph Convolutional Networks for Social Event Detection. In Proceedings of the Web Conference 2021, pp. 3383-3395.

.. [#Cao2021kpgnn] Cao, Y., Peng, H., Wu, J., Dou, Y., Li, J., and Yu, P.S., 2021. Knowledge-preserving incremental social event detection via heterogeneous GNNs. In Proceedings of the Web Conference 2021, pp. 3383-3395.

.. [#Peng2022finevent] Peng, H., Li, J., Gong, Q., Song, Y., Ning, Y., Lai, K., and Yu, P.S., 2019. Fine-grained event categorization with heterogeneous graph convolutional networks. arXiv preprint arXiv:1906.04580.

.. [#Ren2022qsgnn] Ren, J., Jiang, L., Peng, H., Cao, Y., Wu, J., Yu, P.S., and He, L., 2022. From known to unknown: Quality-aware self-improving graph neural network for open set social event detection. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pp. 1696-1705.

.. [#Ren2023etgnn] Ren, J., Jiang, L., Peng, H., Liu, Z., Wu, J., and Yu, P.S., 2022. Evidential temporal-aware graph-based social event detection via Dempster-Shafer theory. In 2022 IEEE International Conference on Web Services (ICWS), pp. 331-336. IEEE.

.. [#Guo2023hcrc] Guo, Y., Zang, Z., Gao, H., Xu, X., Wang, R., Liu, L., and Li, J., 2024. Unsupervised social event detection via hybrid graph contrastive learning and reinforced incremental clustering. Knowledge-Based Systems, 284, p. 111225. Elsevier.

.. [#Ren2023uclsad] Ren, J., Jiang, L., Peng, H., Liu, Z., Wu, J., and Yu, P.S., 2023. Uncertainty-guided boundary learning for imbalanced social event detection. IEEE Transactions on Knowledge and Data Engineering. IEEE.

.. [#Li2024rplmsed] Li, P., Yu, X., Peng, H., Xian, Y., Wang, L., Sun, L., Zhang, J., and Yu, P.S., 2024. Relational Prompt-based Pre-trained Language Models for Social Event Detection. arXiv preprint arXiv:2404.08263.

.. [#Cao2024hisevent] Cao, Y., Peng, H., Yu, Z., and Philip, S.Y., 2024. Hierarchical and incremental structural entropy minimization for unsupervised social event detection. In Proceedings of the AAAI Conference on Artificial Intelligence, 38(8), pp. 8255-8264.

.. [#liu2024pygod] Liu, K., Dou, Y., Ding, X., Hu, X., Zhang, R., Peng, H., Sun, L., and Yu, P.S., 2024. PyGOD: A Python library for graph outlier detection. Journal of Machine Learning Research, 25(141), pp. 1-9.

.. [#zhao2019pyod] Zhao, Y., Nasrullah, Z., and Li, Z., 2019. PyOD: A python toolbox for scalable outlier detection. Journal of Machine Learning Research, 20(96), pp. 1-7.

.. [#wang2020maven] Wang, X., Wang, Z., Han, X., Jiang, W., Han, R., Liu, Z., Li, J., Li, P., Lin, Y., and Zhou, J., 2020. MAVEN: A massive general domain event detection dataset. arXiv preprint arXiv:2004.13590.

.. [#mcminn2013event2012] McMinn, A.J., Moshfeghi, Y., and Jose, J.M., 2013. Building a large-scale corpus for evaluating event detection on Twitter. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management, pp. 409-418.

.. [#mazoyer2020event2018] Mazoyer, B., Cagé, J., Hervé, N., and Hudelot, C., 2020. A French corpus for event detection on Twitter. European Language Resources Association (ELRA).


