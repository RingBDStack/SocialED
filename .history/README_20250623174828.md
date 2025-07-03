![SocialED Logo](https://github.com/RingBDStack/SocialED/blob/main/docs/SocialED.png?raw=true)
<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/socialed.svg?color=brightgreen)](https://pypi.org/project/SocialED/)
[![Documentation status](https://readthedocs.org/projects/socialed/badge/?version=latest)](https://socialed.readthedocs.io/en/latest/?badge=latest)
[![GitHub stars](https://img.shields.io/github/stars/RingBDStack/SocialED?style=flat)](https://github.com/RingBDStack/SocialED/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/RingBDStack/SocialED?style=flat)](https://github.com/RingBDStack/SocialED/network)
[![PyPI downloads](https://static.pepy.tech/personalized-badge/SocialED?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/SocialED)
[![Testing](https://github.com/ChenBeici/SocialED/actions/workflows/pytest.yml/badge.svg)](https://github.com/ChenBeici/SocialED/actions/workflows/pytest.yml)
[![License](https://img.shields.io/github/license/RingBDStack/SocialED.svg)](https://github.com/RingBDStack/SocialED/blob/master/LICENSE)
[![CodeQL](https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml/badge.svg)](https://github.com/RingBDStack/SocialED/actions/workflows/codeql.yml)
[![arXiv](https://img.shields.io/badge/cs.LG-2412.13472-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.13472)

</div>


---

# SocialED

A Python Library for Social Event Detection

## What is Social Event Detection?

Social Event Detection (SED) is a cutting-edge research area in AI and NLP that focuses on:

- 🔍 Automatically identifying and analyzing events from social media platforms (Twitter, Facebook, etc.)
- 🌎 Covering diverse event types from natural disasters to viral phenomena
- 🤖 Leveraging AI to understand real-world events through social data


## 📚 About SocialED

SocialED is your all-in-one Python toolkit for Social Event Detection that offers:

### 📊 Rich Resources
- 19 detection algorithms
- 15 diverse datasets
- Unified API with detailed documentation

### 🛠️ Key Capabilities
- Comprehensive preprocessing (graph construction, tokenization)
- Standardized interfaces for training & prediction
- Easy-to-extend modular architecture

### ⚡ Technical Excellence
- Deep learning framework integration
- CPU & GPU support for high performance
- Production-grade code quality with testing & CI/CD

## ⭐ Key Features

- **🤖 Comprehensive Algorithm Collection**: Integrates 19 detection algorithms and supports 15 widely-used datasets, with continuous updates to include emerging methods
- **📝 Unified API Design**: Implements algorithms with a consistent interface, allowing seamless data preparation and integration across all models
- **🔧 Modular Components**: Provides customizable components for each algorithm, enabling users to adjust models to specific needs
- **🛠️ Rich Utility Functions**: Offers tools designed to simplify the construction of social event detection workflows
- **✨ Robust Implementation**: Includes comprehensive documentation, examples, unit tests, and maintainability features

SocialED includes **19** social event detection algorithms.
For consistency and accessibility, SocialED is developed on top of [DGL](https://www.dgl.ai/) 
and [PyTorch](https://pytorch.org/), and follows the API design of [PyOD](https://github.com/yzhao062/pyod)
and [PyGOD](https://github.com/pygod-team/pygod).
See examples below for detecting outliers with SocialED in 7 lines!


SocialED plays a crucial role in various downstream applications, including:

* Crisis management
* Public opinion monitoring
* Fake news detection
* And more...

![SocialED API](https://github.com/RingBDStack/SocialED/blob/main/docs/API.png?raw=true)



## 📁 Folder Structure
```
SocialED
├── LICENSE
├── MANIFEST.in 
├── README.rst
├── docs
├── SocialED
│   ├── __init__.py
│   ├── datasets
│   ├── detector
│   ├── utils
│   ├── tests
│   └── metrics
├── requirements.txt
├── setup.cfg
└── setup.py
```



## 🔧 Installation




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


## 📋 Collected Algorithms


The library integrates methods ranging from classic approaches like LDA and BiLSTM to specialized techniques such as KPGNN, QSGNN, FinEvent, and HISEvent. Despite significant advancements in detection methods, deploying these approaches or conducting comprehensive evaluations has remained challenging due to the absence of a unified framework. SocialED addresses this gap by providing a standardized platform for researchers and practitioners in the SED field.

## Implemented Algorithms


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

|    Methods    |       Year        |       Backbone        |       Scenario        |    Supervision    |    Paper    |
| :-----------: | :---------------: | :------------------: | :------------------: | :---------------: | :---------------: |
|  LDA  |       2003       |       Topic        |        Offline         | Unsupervised | [Blei2003lda](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) |
|   BiLSTM   |       2005       |    Deep learning    |        Offline         | Supervised | [Graves2005bilstm](https://www.sciencedirect.com/science/article/abs/pii/S0893608005001206) |
| Word2Vec  |       2013       |  Word embeddings   |        Offline         | Unsupervised | [Mikolov2013word2vec](https://arxiv.org/abs/1301.3781) |
| GloVe | 2014 | Word embeddings | Offline | Unsupervised | [Pennington2014glove](https://aclanthology.org/D14-1162.pdf) |
| WMD | 2015 | Similarity | Offline | Unsupervised | [Kusner2015wmd](https://proceedings.mlr.press/v37/kusnerb15) |
| BERT | 2018 | PLMs | Offline | Unsupervised | [Devlin2018bert](https://aclanthology.org/N19-1423/) |
| SBERT | 2019 | PLMs | Offline | Unsupervised | [Reimers2019sbert](https://aclanthology.org/D19-1410.pdf)    |
| EventX | 2020 | Community | Offline | Unsupervised | [Liu2020eventx](https://dl.acm.org/doi/abs/10.1145/3377939) |
| CLKD | 2021 | GNNs | Online | Supervised | [Ren2021clkd](https://arxiv.org/abs/2108.03084) |
| KPGNN | 2021 | GNNs | Online | Supervised | [Cao2021kpgnn](https://dl.acm.org/doi/abs/10.1145/3442381.3449834) |
| FinEvent | 2022 | GNNs | Online | Supervised | [Peng2022finevent](https://ieeexplore.ieee.org/document/9790195) |
| QSGNN | 2022 | GNNs | Online | Supervised | [Ren2022qsgnn](https://dl.acm.org/doi/pdf/10.1145/3511808.3557329) |
| ETGNN | 2023 | GNNs | Offline | Supervised | [Ren2023etgnn](https://ieeexplore.ieee.org/abstract/document/9885765) |
| HCRC | 2023 | GNNs | Online | Unsupervised | [Guo2023hcrc](https://www.sciencedirect.com/science/article/abs/pii/S0950705123009759) |
| UCLSED | 2023 | GNNs | Offline | Supervised | [Ren2023uclsad](https://ieeexplore.ieee.org/abstract/document/10285435) |
| RPLMSED | 2024 | PLMs | Online | Supervised | [Li2024rplmsed](https://dl.acm.org/doi/abs/10.1145/3695869) |
| HISEvent | 2024 | Community | Online | Unsupervised | [Cao2024hisevent](https://ojs.aaai.org/index.php/AAAI/article/view/28666) |
| ADPSEMEvent | 2024 | Community | Online | Unsupervised | [Yang2024adpsemevent](https://dl.acm.org/doi/abs/10.1145/3627673.3679537) |
| HyperSED | 2024 | Community | Online | Unsupervised | [Yu2024hypersed](https://arxiv.org/abs/2412.10712) |

### 📊 Supported Datasets

Below is a summary of all datasets supported by SocialED:

-   **Event2012**: A comprehensive dataset containing 68,841 annotated English tweets spanning 503 distinct event categories. The data was collected over a continuous 29-day period, providing rich temporal context for event analysis.

-   **Event2018**: A French language dataset comprising 64,516 annotated tweets across 257 event categories. The collection period covers 23 consecutive days, offering valuable insights into French social media event patterns.

-   **Arabic_Twitter**: A specialized dataset of 9,070 annotated Arabic tweets focusing on seven major catastrophic events. This collection enables research into crisis-related social media behavior in Arabic-speaking regions.

-   **MAVEN**: A diverse English dataset containing 10,242 annotated texts distributed across 164 event types. Carefully curated to support development of domain-agnostic event detection models.

-   **CrisisLexT26**: An emergency-focused collection of 27,933 tweets covering 26 distinct crisis events. This dataset enables research into social media dynamics during critical situations.

-   **CrisisLexT6**: A focused dataset of 60,082 tweets documenting 6 major crisis events. Provides deep insights into public communication patterns during large-scale emergencies.

-   **CrisisMMD**: A multimodal dataset featuring 18,082 manually annotated tweets from 7 major natural disasters in 2017. Covers diverse events including earthquakes, hurricanes, wildfires, and floods across multiple geographical regions.

-   **CrisisNLP**: A comprehensive crisis-related collection of 25,976 tweets spanning 11 distinct events. Includes human-annotated data, lexical resources, and specialized tools for crisis information analysis.

-   **HumAID**: An extensive dataset of 76,484 manually annotated tweets documenting 19 major natural disasters between 2016-2019. Provides broad coverage of various disaster types across different geographical and temporal contexts.

-   **Mix_data**: A rich composite dataset integrating multiple crisis-related collections:
    - **ICWSM2018**: 21,571 expert-labeled tweets from the 2015 Nepal earthquake and 2013 Queensland floods
    - **ISCRAM2013**: 4,676 annotated tweets from the 2011 Joplin tornado
    - **ISCRAM2018**: 49,804 tweets covering Hurricanes Harvey, Irma, and Maria (2017)
    - **BigCrisisData**: 2,438 tweets with detailed crisis-related classifications

-   **KBP**: A structured dataset containing 85,569 texts across 100 event types, designed for benchmarking information extraction systems and event knowledge base population.

-   **Event2012_100**: A carefully curated subset containing 15,019 tweets distributed across 100 events. Features natural class imbalance with event sizes ranging from 55 to 2,377 tweets (imbalance ratio ~43).

-   **Event2018_100**: A French language subset comprising 19,944 tweets across 100 events. Exhibits significant class imbalance with event sizes from 27 to 4,189 tweets (imbalance ratio ~155).

-   **Arabic_7**: A focused Arabic dataset containing 3,022 tweets distributed across 100 events. Shows natural variation in event sizes from 7 to 312 tweets (imbalance ratio ~44).

-   **CrisisLexT7**: A dataset of 1,959 tweets across 7 events. Features a natural imbalance with event sizes ranging from 15 to 989 tweets (imbalance ratio ~66).

## Dataset


|    Dataset    |       Language        |       Events        |       Texts        |    Long tail    |
| :-----------: | :---------------: | :------------------: | :------------------: | :---------------: |
| Event2012 | English | 503 | 68,841 | No |
| Event2018 | French | 257 | 64,516 | No |
| Arabic_Twitter | Arabic | 7 | 9,070 | No |
| MAVEN | English | 164 | 10,242 | No |
| CrisisLexT26 | English | 26 | 27,933 | No |
| CrisisLexT6 | English | 6 | 60,082 | No |
| CrisisMMD | English | 7 | 18,082 | No |
| CrisisNLP | English | 11 | 25,976 | No |
| HumAID | English | 19 | 76,484 | No |
| Mix_Data | English | 5 | 78,489 | No |
| KBP | English | 100 | 85,569 | No |
| Event2012_100 | English | 100 | 15,019 | Yes |
| Event2018_100 | French | 100 | 19,944 | Yes |
| Arabic_7 | Arabic | 7 | 3,022 | Yes |
| CrisisLexT7 | English | 7 | 1,959 | Yes |



## 🏗️ Library Design and Implementation

### 🔧 Dependencies and Technology Stack

SocialED is compatible with Python 3.8 and above, and leverages well-established deep learning frameworks like PyTorch and Hugging Face Transformers for efficient model training and inference, supporting both CPU and GPU environments. In addition to these core frameworks, SocialED also integrates NumPy, SciPy, and scikit-learn for data manipulation, numerical operations, and machine learning tasks, ensuring versatility and performance across a range of workflows.

### 🔄 Unified API Design

Inspired by the API designs of established frameworks, we developed a unified API for all detection algorithms in SocialED:

1. ``preprocess`` provides a flexible framework for handling various preprocessing tasks, such as graph construction and tokenization
2. ``fit`` trains the detection algorithms on the preprocessed data, adjusting model parameters and generating necessary statistics for predictions
3. ``detection`` uses the trained model to identify events from the input data, returning the detected events
4. ``evaluate`` assesses the performance of the detection results by comparing predictions with ground truth data, providing metrics like precision, recall and F1-score

### 💻 Example Usage


    from SocialED.dataset import Event2012                 # Load the dataset
    dataset = Event2012()                                  # Load "Event2012" dataset
    
    from SocialED.detector import KPGNN                    # Import KPGNN model
    kpgnn = KPGNN(dataset, batch_size=2048)                # Initialize KPGNN model
    
    kpgnn.preprocess()                                     # Preprocess data
    kpgnn.fit()                                           # Train the model
    pres, trus = kpgnn.detection()                        # Detect events
    
    kpgnn.evaluate(pres, trus)                            # Evaluate detection results



## Demo
This is a demo showing how to use the KPGNN model for event detection. For quick demonstration purposes, we set epochs=3 and batch_size=2048 to speed up training. The code below demonstrates loading the Event2012 dataset, initializing and training a KPGNN model, and evaluating its performance.

<img src="docs/demo.gif">


### 🧩 Modular Design and Utility Functions

SocialED is built with a modular design to improve reusability and reduce redundancy. It organizes social event detection into distinct modules:

* ``preprocessing``
* ``modeling``
* ``evaluation``


The library provides several utility functions including:

* ``utils.tokenize_text`` and ``utils.construct_graph`` for data preprocessing
* ``metric`` for evaluation metrics

## 🛡️ Library Robustness and Accessibility

### ✅ Quality and Reliability

* Built with robustness and high-quality standards
* Continuous integration through GitHub Actions
* Automated testing across Python versions and operating systems
* PyPI-compatible and PEP 625 compliant
* Follows PEP 8 style guide

### 🤝 Accessibility and Community Support

* Detailed API documentation on Read the Docs
* Step-by-step guides and tutorials
* Intuitive API design inspired by scikit-learn
* Open-source project hosted on GitHub
* Easy issue-reporting mechanism
* Clear contribution guidelines

## 🔮 Future Development Plans

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

## 👥 Contributors

### 🌟 Core Team

<div align="center">
<h3>🌟 Core Team</h3>

| Name | Affiliation | Contact |
|:---:|:---:|:---:|
| **Kun Zhang** | 🎓 Beihang University | 📧 zhangkun23@buaa.edu.cn |
| **Xiaoyan Yu** | 🎓 Beijing Institute of Technology | 📧 xiaoyan.yu@bit.edu.cn |
| **Pu Li** | 🎓 Kunming University of Science and Technology | 📧 lip@stu.kust.edu.cn |
| **Ye Tian** | 🎓 Laboratory for Advanced Computing<br>and Intelligence Engineering | 📧 sweetwild@mail.ustc.edu.cn |
| **ZhiLin Xu** | 🎓 Beihang University | 📧 21377240@buaa.edu.cn |
| **Kaiwei Yang** | 🎓 Beihang University | 📧 yangkw@buaa.edu.cn |
| **Hao Peng** 📍<br>*Corresponding Author* | 🎓 Beihang University | 📧 penghao@buaa.edu.cn |
| **Philip S. Yu** | 🎓 University of Illinois at Chicago | 📧 psyu@uic.edu |

</div>



## 📊 Citation

```bibtex
@inproceedings{zhang2025socialed,
  author    = {Kun Zhang and Xiaoyan Yu and Pu Li and Hao Peng and Philip S. Yu},
  title     = {SocialED: A Python Library for Social Event Detection},
  booktitle = {Proceedings of the 2025 Asia Pacific Web and Web-Age Information Management Joint International Conference},
  publisher = {Springer}, 
  year      = {2025},
  pages     = {xx--yy}, 
  doi       = {10.1007/xxxxx},
  url       = {https://doi.org/10.1007/xxxxx} 
}

```

## 📚 References

* D.M. Blei, A.Y. Ng, and M.I. Jordan. Latent Dirichlet allocation. Journal of Machine Learning Research, 3(Jan):993-1022, 2003.

* A. Graves and J. Schmidhuber. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5-6):602-610, 2005.

* T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. In International Conference on Learning Representations, pages 1-12, 2013.

* J. Pennington, R. Socher, and C.D. Manning. GloVe: Global Vectors for Word Representation. In Proceedings of EMNLP, pages 1532-1543, 2014.

* M. Kusner, Y. Sun, N. Kolkin, and K. Weinberger. From word embeddings to document distances. In International Conference on Machine Learning, pages 957-966, 2015.

* J. Devlin, M.W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171-4186, 2019.

* N. Reimers and I. Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of EMNLP-IJCNLP, pages 3980-3990, 2019.

* B. Liu, F.X. Han, D. Niu, L. Kong, K. Lai, and Y. Xu. Story forest: Extracting events and telling stories from breaking news. ACM Transactions on Knowledge Discovery from Data, 14(3):1-28, 2020.

* J. Ren, H. Peng, L. Jiang, Z. Liu, J. Wu, and P.S. Yu. Toward cross-lingual social event detection with hybrid knowledge distillation. ACM Transactions on Knowledge Discovery from Data, 18(9):1-36, 2024.

* Y. Cao, H. Peng, J. Wu, Y. Dou, J. Li, and P.S. Yu. Knowledge-preserving incremental social event detection via heterogeneous GNNs. In Proceedings of The Web Conference, pages 3383-3395, 2021.

* H. Peng, R. Zhang, S. Li, et al. Reinforced, incremental and cross-lingual event detection from social messages. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(1):980-998, 2022.

* J. Ren, L. Jiang, H. Peng, Y. Cao, J. Wu, P.S. Yu, and L. He. From known to unknown: Quality-aware self-improving graph neural network for open set social event detection. In Proceedings of CIKM, pages 1696-1705, 2022.

* J. Ren, L. Jiang, H. Peng, Z. Liu, J. Wu, and P.S. Yu. Evidential temporal-aware graph-based social event detection via Dempster-Shafer theory. In IEEE ICWS, pages 331-336, 2022.

* Y. Guo, Z. Zang, H. Gao, X. Xu, R. Wang, L. Liu, and J. Li. Unsupervised social event detection via hybrid graph contrastive learning and reinforced incremental clustering. Knowledge-Based Systems, 284:111225, 2024.

* J. Ren, L. Jiang, H. Peng, Z. Liu, J. Wu, and P.S. Yu. Uncertainty-guided boundary learning for imbalanced social event detection. IEEE Transactions on Knowledge and Data Engineering, 2023.

* P. Li, X. Yu, H. Peng, Y. Xian, L. Wang, L. Sun, J. Zhang, and P.S. Yu. Relational prompt-based pre-trained language models for social event detection. ACM Transactions on Information Systems, 43(1):1-43, 2024.

* Y. Cao, H. Peng, Z. Yu, and P.S. Yu. Hierarchical and incremental structural entropy minimization for unsupervised social event detection. In Proceedings of AAAI, 38(8):8255-8264, 2024.

* Z. Yang, Y. Wei, H. Li, et al. Adaptive Differentially Private Structural Entropy Minimization for Unsupervised Social Event Detection. In Proceedings of CIKM, pages 2950-2960, 2024.

* X. Yu, Y. Wei, S. Zhou, Z. Yang, L. Sun, H. Peng, L. Zhu, and P.S. Yu. Towards effective, efficient and unsupervised social event detection in the hyperbolic space. In Proceedings of AAAI, 2025.
