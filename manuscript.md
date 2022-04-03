---
title: Computational Methods for Single-Cell Multi-Omics Integration and Alignment
keywords:
- Single-cell
- Multi-omics
- Machine learning
- Integration
lang: en-US
date-meta: '2022-04-03'
author-meta:
- Stefan Stanojevic
- Yijun Li
- Lana Garmire
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="Computational Methods for Single-Cell Multi-Omics Integration and Alignment" />
  <meta name="citation_title" content="Computational Methods for Single-Cell Multi-Omics Integration and Alignment" />
  <meta property="og:title" content="Computational Methods for Single-Cell Multi-Omics Integration and Alignment" />
  <meta property="twitter:title" content="Computational Methods for Single-Cell Multi-Omics Integration and Alignment" />
  <meta name="dc.date" content="2022-04-03" />
  <meta name="citation_publication_date" content="2022-04-03" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Stefan Stanojevic" />
  <meta name="citation_author_institution" content="Department of Computational Medicine and Bioinformatics, University of Michigan" />
  <meta name="citation_author" content="Yijun Li" />
  <meta name="citation_author_institution" content="Department of Biostatistics, University of Michigan" />
  <meta name="citation_author_orcid" content="0000-0003-0513-9565" />
  <meta name="twitter:creator" content="@jenny589446011" />
  <meta name="citation_author" content="Lana Garmire" />
  <meta name="citation_author_institution" content="Department of Computational Medicine and Bioinformatics, University of Michigan" />
  <meta name="citation_author_orcid" content="0000-0003-1672-6917" />
  <meta name="twitter:creator" content="@GarmireGroup" />
  <link rel="canonical" href="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta property="og:url" content="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta property="twitter:url" content="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta name="citation_fulltext_html_url" content="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta name="citation_pdf_url" content="https://codingpoppy.github.io/multiomics_review_manubot/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://codingpoppy.github.io/multiomics_review_manubot/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://codingpoppy.github.io/multiomics_review_manubot/v/409a41161091da8d14116425d6ce3e4d606f4e9f/" />
  <meta name="manubot_html_url_versioned" content="https://codingpoppy.github.io/multiomics_review_manubot/v/409a41161091da8d14116425d6ce3e4d606f4e9f/" />
  <meta name="manubot_pdf_url_versioned" content="https://codingpoppy.github.io/multiomics_review_manubot/v/409a41161091da8d14116425d6ce3e4d606f4e9f/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <meta property="og:image" content="https://github.com/codingpoppy/multiomics_review_manubot/raw/409a41161091da8d14116425d6ce3e4d606f4e9f/content/images/thumbnail.png" />
  <meta property="twitter:image" content="https://github.com/codingpoppy/multiomics_review_manubot/raw/409a41161091da8d14116425d6ce3e4d606f4e9f/content/images/thumbnail.png" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://codingpoppy.github.io/multiomics_review_manubot/v/409a41161091da8d14116425d6ce3e4d606f4e9f/))
was automatically generated
from [codingpoppy/multiomics_review_manubot@409a411](https://github.com/codingpoppy/multiomics_review_manubot/tree/409a41161091da8d14116425d6ce3e4d606f4e9f)
on April 3, 2022.
</em></small>

## Authors



+ **Stefan Stanojevic**<br><br>
  <small>
     Department of Computational Medicine and Bioinformatics, University of Michigan
  </small>

+ **Yijun Li**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0003-0513-9565](https://orcid.org/0000-0003-0513-9565)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [codingpoppy](https://github.com/codingpoppy)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [jenny589446011](https://twitter.com/jenny589446011)<br>
  <small>
     Department of Biostatistics, University of Michigan
  </small>

+ **Lana Garmire**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0003-1672-6917](https://orcid.org/0000-0003-1672-6917)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [lanagarmire](https://github.com/lanagarmire)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [GarmireGroup](https://twitter.com/GarmireGroup)<br>
  <small>
     Department of Computational Medicine and Bioinformatics, University of Michigan
  </small>



## Abstract {.page_break_before}

Recently developed technologies to generate single-cell genomic data have made a revolutionary impact in the field of biology. Multi-omics assays offer even greater opportunities to understand cellular states and biological processes. However, the problem of integrating different -omics data with very different dimensionality and statistical properties remains quite challenging. A growing body of computational tools are being developed for this task, leveraging ideas ranging from machine translation to the theory of networks and representing a new frontier on the interface of biology and data science. Our goal in this review paper is to provide a comprehensive, up-to-date survey of computational techniques for the integration of multi-omics and alignment of multiple modalities of genomics data in the single cell research field.


## Introduction

Single-cell sequencing technologies have opened the door to investigating biological processes at an unprecedentedly high resolution. Techniques such as DROP-seq [@doi:10.1016/j.cell.2015.05.002] and 10x Genomics assays are capable of measuring single-cell gene expression, or scRNA-seq, in tens of thousands of single cells simultaneously. Measurements of other data modalities are also increasingly available. For example, single-cell ATAC-seq (scATAC-seq) assesses chromatin accessibility, and single-cell bisulfite sequencing captures DNA methylation, all from single cells. However, many of such techniques are designed to measure a single modality and do not lend themselves to multi-omics measurements. The way to combine information from such measurements is then to assay different -omics from different subsets of the same samples. By assuming that cells assayed by different techniques share similar properties, one can then use alignment methods to computationally aggregate similar cells across different omics assays and draw consensus biological inference.

Recently, however, a number of experimental techniques capable of assaying multiple modalities simultaneously from the same set of single cells have been developed. CITE-seq [@doi:10.1038/nmeth.4380] and REAP-seq [@doi:10.1038/nbt.3973] measure proteins and gene expression. SNARE-seq [@doi:10.1038/nbt.3973; @doi:10.1038/s41587-019-0290-0], SHARE-seq [@doi:10.1038/s41576-020-00308-6] and sci-CAR [@doi:10.1126/science.aau0730] measure gene expression and chromatin accessibility, while scGEM [@doi:10.1038/nmeth.3961] measures gene expression and DNA methylation. For triple-omics data generation, scNMT [@doi:10.1038/s41467-018-03149-4] measures gene expression, chromatin accessibility and DNA methylation, and scTrio-seq [@doi:10.1038/nmeth.3961; @doi:10.1126/science.aao3791] captures SNPs, gene expression and DNA methylation simultaneously. Integrative analysis of such data obtained from the same cells remains a challenging computational task due to a combination of reasons, such as the noise and sparsity in the assays, and different statistical distributions for different modalities. For clarity, we distinguish between integration methods that combine multiple -omics data from the set of the same single cells (Section I), from alignment methods designed to work with multi-modal data coming from the same tissue but different cells (Section II). The difference in their approaches is shown in Figure {@fig:1}.

![Multi-omics data can sometimes be sequenced from the same set of single cells (left); at other times, only the data sequenced from the same/similar sample, but different single cells are available (right). In the former case, we have the task of integrating the different data modalities (left); in the latter case, we need to first identify similar cells across the samples (right) - this is the computational task of alignment.](images/Fig_1.png){#fig:1 width="75%" height="75%"}

The application of data fusion algorithms for multi-omics sequencing data predates the single-cell technologies; bulk-level data have been integrated using a variety of computational tools as reviewed in [@doi:10.3389/fgene.2017.00084]. In this review, we aim to give a comprehensive, up-to-date summary of existing computational tools of multi-omics data integration and alignment in the single-cell field, for researchers in the field of computational biology. For more general surveys, the readers are encouraged to check other single-cell multi-omics reviews [@doi:10.1016/j.coisb.2018.01.003; @doi:10.1016/j.tibtech.2020.02.013; @doi:10.1093/bib/bbaa042; @doi:10.1038/s41587-021-00895-7; @doi:10.1016/j.csbj.2021.04.060; @doi:10.1038/s41581-021-00463-x].


## Integration methods handling multi-omics data generated from the same single cells

The integration methods for multi-modal data assayed from the same set of single cells can be broadly categorized into at least three main types by methodology: mathematical matrix factorization methods, AI (eg. neural-network) based methods and network-based methods. The scheme of these methods is illustrated in Figure {@fig:2}. Additional less diversified approaches include a Bayesian statistical method and a metric learning method. The list of the currently implemented methods is summarized in Table @tbl:1.

![Illustration of some common integration approaches for single-cell multi-omics: matrix factorization, neural network and network-based approaches.](images/Fig_2.png){#fig:2 width="75%" height="75%"}

| Methodology  Category | Method               | Data                                  | Algorithm                                                        | Reference |
|-----------------------|----------------------|---------------------------------------|------------------------------------------------------------------|-----------|
| Matrix Factorization  | MOFA+                | Transcriptomic, Epigenetic            | Matrix Factorization with Automatic Relevance Determination      | [@doi:10.1038/nmeth.3961]       |
|                       | scAI                 | Transcriptomic, Epigenetic            | Matrix factorization, with custom aggregation of epigenetic data | [@doi:10.3389/fgene.2017.00084]      |
| Neural Network        | totalVI              | Transcriptomic, Proteomic             | Variational autoencoder                                          | [@doi:10.1016/j.tibtech.2020.02.013]      |
|                       | scMVAE               | Transcriptomic, Epigenetic            |                                                                  | [@doi:10.1093/bib/bbaa042]      |
|                       | DCCA                 | Transcriptomic, Epigenetic            |                                                                  | [@doi:10.1093/bioinformatics/btab403]      |
|                       | LIBRA                | Transcriptomic, Proteomic, Epigenetic | Split-brain autoencoder                                          | [@doi:10.1038/s41581-021-00463-x]      |
|                       | BABEL                | Transcriptomic, Proteomic, Epigenetic | Autoencoder translating between modalities                       | [@doi:10.1101/2020.11.09.375550]      |
|                       | DeepMAPS             | Transcriptomic, Epigenetic, Proteomic | Graph Neural Network                                             | [@doi:10.1101/2021.10.31.466658]      |
| Network - Based       | citeFUSE             | Transcriptomic, Proteomic             | Similarity network fusion                                        | [@doi:10.15252/msb.20178124]      |
|                       | Seurat v4            | Transcriptomic, Proteomic             | Weighted averaging of nearest neighbor graphs                    | [@doi:10.1186/s13059-020-1932-8]      |
|                       | Integrated Diffusion | Transcriptomic                        | Joint Manifold Learning through Integrated Diffusion             | [@doi:10.48550/arXiv.2102.06757]      |
| Other                 | BREM-SC              | Transcriptomic, Proteomic             | Bayesian mixture model                                           | [@doi:10.1093/nar/gkaa314]      |
|                       | SCHEMA               | Transcriptomic, Epigenetic            | Metric Learning                                                  | [@doi:10.1186/s13059-021-02313-2]      |

Table: Summary of the methods for integrating multi-omics data from the same cells. {#tbl:1}

### Matrix Factorization based methods

Matrix factorization methods aim to describe each cell as the product between a vector that describes each -omics element (genes, epigenetic loci, proteins, etc.) and a vector of reduced and common features ("factors") capturing its basic properties (Figure {@fig:2}A). Mathematically, if we represent each -omics as matrix $X_{i (i=1,2,\cdots)}$ then matrix factorization decomposes it as the product of a shared matrix H across all omics data types, and -omics specific matrix $W_{i (i=1,2,\cdots)}$, together with random noise $\epsilon_{i (i=1,2,\cdots)}$ as
$$X_1=W_1H+\epsilon_1, X_2=W_2H+\epsilon_2, \cdots, X_i=W_iH+\epsilon_i$$
Such methods are simple and easily interpretable since the cell and -omics factors both carry clearly discernible biological meaning, but may lack the ability to capture nonlinear effects. We describe the variations in this type of methods below:

**MOFA+** [@doi:10.1186/s13059-020-02015-1] is a sequel to the MOFA (Multi-Omics Factor Analysis) [@doi:10.15252/msb.20178124]. Both studies perform factor analysis, equipped with sparsity-inducing Bayesian elements including Automatic Relevance Determination [@isbn:9780387947242]. MOFA+ integrates data over both views (corresponding to different modalities) and groups (corresponding to different experimental conditions). The model scales easily to large datasets. MOFA+ was applied to integrate gene expression, chromatin accessibility and DNA methylation data assayed using scNMT from mouse embryos, as well as to integrate several datasets over different experimental conditions rather than different -omics. After performing factor analysis on the mouse dataset, the most relevant factors are related to biological processes shaping embryo development. MOFA+ provides an elegant and successful general framework for integration, which could potentially be superseded in specific cases by more specialized models designed for integrating specific -omics layers.

**scAI** ("single-cell aggregation and inference") [@doi:10.1186/s13059-020-1932-8] features a twist on matrix factorization and is designed specifically for integration of epigenetic (chromatin accessibility, DNA methylation) and transcriptomic data. It addresses the sparsity of epigenetic data by aggregating (averaging) such data between similar cells. This requires a notion of cell-cell similarity which is learned as a part of the model, rather than being postulated prior to the integration. Their model solves the following optimization problem
$$\min_{W_1, W_2, H, Z}\alpha||X_1-W_1H||^2_F + ||X_2(Z\cdot R-W_2H)||^2_F + \lambda||Z_H^TH||^2_F +\gamma\sum_j||H_{\cdot j}||_1^2$$
where $X_1$ represents the transcriptomic data, $X_2$ the epigenomic data, $H$ are the common (cell-specific) factors, $W_1$, $W_2$ are the assay-specific factors, $Z$ is the cell-cell similarity matrix, and entries of are Bernoulli-distributed random variables. The twist on the usual matrix factorization is made by factoring aggregated epigenetic data $X_2(Z\cdot R)$, rather than directly factoring the epigenetic data $X_2$. After the learning is complete, the matrix of cell factors is used to cluster the cells and the importance of genes and epigenetic marks is ranked using the magnitude of the values in loading matrices. In order to jointly visualize different factors, scAI implements a novel VscAI algorithm utilizing Sammon mappings [@doi:10.1109/T-C.1969.222678]. The relationships between epigenetics and gene expression can be explored using correlation analysis and nonnegative least square regression. The model was tested on simulations using MOSim [@doi:10.1101/421834], and several real world datasets, and performed better than the earlier MOFA version, in terms of identifying natural clusters and condensing epigenetic data into meaningful factors.

### Neural Network based methods

While neural networks are generally well-suited for supervised tasks, a class of neural networks called autoencoders is commonly used for unsupervised learning, such as the multi-omics integration problem in single cells. Deep autoencoders perform nonlinear dimensionality reduction by squeezing the input through a lower-dimensional hidden layer ("bottle neck") and attempting to reconstruct the original input as the output of the neural network (Figure {@fig:2}B). They consist of two parts: the "encoder" network performing the dimensionality reduction and the "decoder" network reconstructing based on the dimensionally reduced data. In principle, autoencoders generalize the principal component analysis by allowing for nonlinear transformations. Many variations of autoencoder models exist, and among them variational autoencoders
have proven useful for analyzing single-cell data. Rather than directly encoding the data in a dimensionally reduced ("latent") space, variational autoencoders sample from a probability distribution (usually Gaussian) in the latent space, and use the encoder network to produce the parameters of this distribution. As such, they combine deep learning and Bayesian inference to produce generative models, which not only dimensionally reduce the original data but also produce realistic synthetic data points. Below we review the methods using certain variations of the autoencoder architecture to integrate single-cell multi-omics data.

**scMVAE** ("Single Cell Multimodal Variational Autoencoder") [@doi:10.1093/bib/bbaa287] was designed to integrate transcriptomic and chromatin accessibility data, using a version of a variational autoencoder. The key question in multi-omics integration is how to encode the multi-omics data into a single latent space representation. In the case of scMVAE, a combination of 3 different methods was used for this task, including a neural network acting on the concatenated input data, neural networks encoding transcriptomic and chromatin accessibility data separately prior to merging, and a "Product of Experts" technique for combining different representations [@doi:10.1162/089976602760128018]. At the same time, cell-specific scales used to normalize expression across cells are learned (called "library factors"). The input data are reconstructed by processing the latent representations via decoder neural networks, which calculate the probabilities of gene dropouts and predict the expression of measured genes modelled as a negative binomial distribution.

This model incorporates the task of constructing shared representations of the multi-modal data with clustering. Namely, one of the latent variables is constructed to correspond to the clustering label $c$. Furthermore, the model incorporates tools to deal with tasks such as data imputation, and can be used for studying the association between epigenetics and gene expression. scMVAE was applied to integrate two real datasets assaying mRNA and chromatin accessibility using SNARE-seq method, as well as simulated data generated by "Splatter" [@doi:10.1186/s13059-017-1305-0]. It takes into account the known relationships between appropriately located transcription factors and gene expression, and uses them to test the imputed (denoised) data. According to the authors, scMVAE performed better than MOFA in terms of clustering and enhancing the consistency between different -omics layers on several real and simulated datasets.

**DCCA**, denoting "Deep cross-omics cycle attention model", is another method in this category for joint analysis of single-cell multi-omics data [@doi:10.1093/bioinformatics/btab403]. It uses variational autoencoders to integrate multi-omics data, and builds on the scMVAE algorithm described above. However, DCCA diverges from scMVAE in one important aspect: DCCA uses separate but coupled autoencoders to dimensionally reduce different -omics layers, while scMVAE constructs a shared dimensionally reduced representation of transcriptomic and
epigenetic data. This strategy is inspired by the theory of machine translation, notably the so-called "attention transfer"; in this case, the "teacher network" working with the scRNA-seq data guides the learning of the "student network" working with scATAC-seq data. Their model compares favorably to scAI and MOFA+ on metrics such as clustering accuracy, denoising quality and consistency between different -omics.

**totalVI** [@doi:10.1101/791947] combines Bayesian inference and a neural network to create a generative model for data integration. It was created to handle gene expression and protein data. Joint latent space representations are learned via an encoder network and used to reconstruct the original data while accounting for the difference between the original data modalities. The model generates latent representations capturing both -omics, and at the same time models experimental conditions through an additional set of latent variables. The gene expression data are sampled from a negative binomial distribution, and the parameters are obtained as outputs of a decoder neural network. The protein data are sampled from a mixture model with two negative binomial distributions simulating the experimental background and the actual signal respectively. The model was applied to two datasets containing transcriptomic and proteomic measurements, and generated shared representations of cells with interpretable components.

**LIBRA** [@doi:10.1101/2021.01.27.428400] uses an autoencoder-like neural network to "translate" between different omics. Motivated by "split-brain autoencoder"[@{https://ieeexplore.ieee.org/document/8099559}], and "machine translation" approach, the model consists of two separate neural networks. The first network takes as input elements of the first dataset and aims to reconstruct a corresponding element of the second dataset. The second network performs an inverse task. Taken together, the bottlenecks of two networks aim to convert the two datasets into the same latent space. This method is quite general and can be applied to various pairs of -omics data. It produced clusters of similar quality compared to Seurat v4.

**BABEL** [@doi:10.1101/2020.11.09.375550] also uses autoencoder-like neural networks to translate between gene expression (modeled by Negative Binomial distribution) and binarized chromatin accessibility data. There are two encoder and two decoder neural networks, each encoder/decoder handles one data type of gene expression or chromatin accessibility. As a result, four combinations between encoders and decoders are formed, and the loss function is optimized to minimize reconstruction error for four combinations of encoders and decoders. In this approach, the two encoders are prone to produce similar representations, as the encoded gene accessibility is decoded as chromatin accessibility and vice versa.
BABEL provides a promising generic framework to multi-omics inference at a single-cell level from single-omics data, by using the model that was previously trained on multi-omics data sequenced from the same single cells. The modular nature of BABEL provides additional flexibility, as the model can be extended to work with additional modalities when the corresponding data becomes available. Despite the potential for generalization, one should be cautioned that if the training is conducted on cell types that are very different, the transfer learning using BABEL is not very successful.

**DeepMAPS** [@doi:10.1101/2021.10.31.466658] integrates different data modalities by a graph transformer neural network architecture for interpretable representation learning. The data is represented using a heterogenous graph in which some of the nodes represent cells and others represent genes. An autoencoder-like graph neural network architecture is used for representation learning, with an attention mechanism. The attention mechanism learns the weights by the contribution of the neighbors to the node of interest. This not only achieves better performance, but also enhances the interpretability to identify genes most relevant to cell state differences. DeepMAPS method learns relevant gene-gene interaction networks and cell-cell similarities, which can be used for downstream steps such as clustering to infer novel cell types. It compared favorably on clustering, compared to state-of-the art techniques such as MOFA+ and totalVI.

### Network-based methods

Network-based methods represent the relationships between different cells using a weighted graph, where cells serve as nodes (Figure {@fig:2}C). Integration is then accomplished by manipulating such graph representation. This approach emphasizes the neighborhood structure and sometimes pools the information between neighbors, leading to additional robustness against the noise. Below are the currently available methods.

**citeFUSE** [@doi:10.1101/854299] integrates transcriptomic and proteomic CITE-seq data using network fusion of similarity graphs corresponding to different modalities. This idea traces back to computer science work [@{https://ieeexplore.ieee.org/document/7348699}] on fusing multi-view networks through cross-diffusion, and to the follow-up SNF method [@doi:10.1038/nmeth.2810] that was used to integrate bulk level multi-omics data. The algorithm adjusts the graph connectivities by a process of diffusion, which allows for the distance information to be aggregated between neighbors. Namely, the algorithm consists of two iterative steps: separate diffusion on different -omics layers and fusion across the -omics layers. It results in a fused consensus matrix of distances between cells, borrowing information from multiple -omics. citeFUSE used spectral clustering to identify cell types, and showed an improvement over single-modality based clusters. Additional benefits of the method include inference of ligand-receptor interactions and a novel tool for doublet detection.

**Joint Diffusion** [@doi:10.48550/arXiv.2102.06757] constructs graph representations of different -omics and then performs a joint diffusion process on the two graphs in order to denoise and integrate the data. This approach builds upon MAGIC [@doi:10.1016/j.cell.2018.05.061], a method for denoising scRNA-seq data, and generalizes it to multi-modal data. Diffusion can be conceptualized as a random walk process. In a graph diffusion algorithm, random walking on the graph can help discover the intrinsic structure of the data hidden behind the noise. In Joint Diffusion random walks are performed while allowing for transitions from one graph to another. A key idea in this work is to quantify the amount of noise in different datasets, through a spectral entropy of the corresponding graphs, and adjust the time one spends on different graphs in accordance with their relative levels of noise. In this way, the transcriptomic and epigenetic data will not be weighted equally, as the transcriptomic data is generally of better quality. This method excels at denoising and visualizations, and was shown to present an improved clustering performance compared to single-modality clustering and the one based on a more naive alternating diffusion process.

**Seurat v4** [@doi:10.1101/2020.10.12.335331] aims to represent the data as a WNN (weighted nearest neighbor) graph in which cells that are similar according to the consensus of both modalities are connected. In the process of constructing a WNN graph, a set of cell-specific weights dictating the relative importance of different -omics data is learned. Such weights often carry important biological meaning. Specifically, Seurat v4 pipeline has the following steps: first, data corresponding to different -omics are dimensionally reduced using PCA to the same number of dimensions. Then, kNN (k nearest neighbor) graphs corresponding to different -omics are constructed. In a kNN graph, each datapoint (a node of this graph) is connected to nearest neighboring nodes. Cell-specific coefficients determining the relative importance of different -omics are then learned by considering the accuracy of inter-modality and cross-modality predictions by nearest neighbor graphs. Lastly, a linear combination of data from different omics is done, using the coefficients learned in the previous step. The nearest neighbors with respect to those linear combinations are then connected to build the WNN graph. Seurat v4 was applied to a CITE-seq based transcriptomic and proteomic dataset, and several other datasets involving mRNA, proteins and chromatin accessibility. The authors compared this method with MOFA+ and totalVI, using correlations (Pearson and Spearman) between the data corresponding to a cell and the average of its nearest latent space neighbors, and claimed that it performed better than MOFA+ or totalVI.

### Other Models

**BREMSC** [@doi:10.1093/nar/gkaa314] is a Bayesian mixture method. It integrates single-cell gene expression and protein data by modeling them as a mixture of probability distributions that share the same underlying set of parameters. The model is useful for performing joint clustering, where confidence in cluster assignments can be quantified via posterior probabilities. It performed favorably compared to single-omics clustering methods. While the MCMC procedure used to train the model can be computationally intensive, the model provides an effective way of integration by accounting the differences between the two -omics layers using probability distributions.

**SCHEMA** [@doi:10.1186/s13059-021-02313-2] is a different metric learning approach that aims to construct a notion of distances on the space of samples, taking into account different -omics data. One of the -omics (usually, scRNA-seq) is considered the primary base for distance, additional omics are then used to modify this distance. This is formulated as optimization of the quadratic function using quadratic programming. The scRNA-seq and scATAC-seq data can thus be integrated, yielding downstream insights into cell developmental trajectories. This method showed a better clustering performance than those based on clustering different modalities separately or integrating them using canonical correlation analysis. It is a useful method for asymmetrically integrating data modalities of different qualities, such as the case of scRNA-seq and scATAC-seq data.


## Alignment methods handling multiple genomics data generated from different single cells of the same tissue

Compared to multi-omics data, it is experimentally much easier to obtain multiple modalities of data where each modality is obtained from similar but different cells of the same tissue. The task to harmonize these data is called alignment (Figure {@fig:1}). The body of literature applying machine learning and statistical methods to this task is rich, including manifold learning, neural-network based methods, and Bayesian methods, as summarized in Table @tbl:2 and depicted in Figure {@fig:3}. Note that some of the methods developed for batch-correct different scRNA-seq datasets, could in principle be repurposed for single-cell multiple omics alignment; we refer readers to previous benchmark studies [@doi:10.1186/s13059-019-1850-9].

![Illustration of some common approaches for alignment of multi-omics single-cell data: Bayesian methods, manifold alignment methods and neural network based models.](images/Fig_3.png){#fig:3 width="75%" height="75%"}

| Methodology Category | Method      | Algorithm                                                            | Data                                              | Reference |
|----------------------|-------------|----------------------------------------------------------------------|---------------------------------------------------|-----------|
| Manifold Alignment   | UNION - Com | Topological Alignment                                                | Transcriptomic, Epigenetic                        | [@doi:10.1093/bioinformatics/btaa443]      |
|                      | MATCHER     | Pseudotime Reconstruction and Manifold Alignment                     | Transcriptomic, Epigenetic                        | [@doi:10.1186/s13059-017-1269-0]      |
|                      | MMD-MA      | Manifold Alignment                                                   | Transcriptomic, Epigenetic (DNAme)                | [@doi:10.1101/644310]      |
|                      | SCOT        | Gromov-Wasserstein optimal transport                                 | Transcriptomic, Epigenetic (DNAme, accessibility) | [@doi:10.1101/2020.04.28.066787]      |
|                      | Pamona      | Partial Gromov-Wasserstein optimal transport                         | Transcriptomic, Epigenetic                        | [@doi:10.1093/bioinformatics/btab594]      |
| Neural Network       | MAGAN       | Generative Adversarial Network                                       | Transcriptomic, Proteomic                         | [@doi:10.48550/arXiv.1803.00385]      |
|                      | SCIM        | Adversarial autoencoder                                              | Transcriptomic, Proteomic (CyTOF)                 | [@doi:10.1093/bioinformatics/btaa843]      |
|                      | Multigrate  | Variational Autoencoder                                              | Transcriptomic, Proteomic                         | [@doi:10.1101/2022.03.16.484643]      |
| Bayesian             | clonealign  | Bayesian latent variable model                                       | RNA-seq, DNA                                      | [@doi:10.1186/s13059-019-1645-z]      |
|                      | MUSIC       | Topic models                                                         | RNA, CRISPR                                       | [@doi:10.1038/s41467-019-10216-x]      |
|   Other              | Seurat v3   | Canonical Correlation Analysis and Mutual Nearest Neighbors analysis | RNA-seq, ATAC-seq                                 | [@doi:10.1016/j.cell.2019.05.031]      |
|                      | bindSC      |  Canonical Correlation Analysis                                      | RNA-seq, ATAC-seq                                 | [@doi:10.1101/2020.12.11.422014]      |
|                      | MAESTRO     |                                                                      | RNA-seq, ATAC-seq                                 | [@doi:10.1186/s13059-020-02116-x]      |
|                      | LIGER       | Matrix factorization                                                 | RNA-seq, methylation                              | [@doi:10.1016/j.cell.2019.05.031; @doi:10.1016/j.cell.2019.05.006]   |

Table: Summary of the computational methods for aligning multiple omics data from different single cells. {#tbl:2}

### Bayesian Methods

**Clonealign** [@doi:10.1186/s13059-019-1645-z] integrates single-cell RNA and DNA sequencing data from heterogeneous populations by assigning cells measured by RNA-seq to clones derived from DNA-seq data. Clonealign is based on a Bayesian latent variable model, where a categorical variable is used to specify cell assignment. The model maps the copy number of a gene to its expression value by introducing a copy number dosage effect on the gene expression. The model is also flexible enough to allow for additional covariates such as batch effects or biological information that can be inferred from the gene expression (cell cycle, etc.). In addition to simulation studies that demonstrated robustness, Clonealign was also applied on real cancer datasets to discover novel clone-specific dysregulated biological pathways.

**MUSIC** [@doi:10.1038/s41467-019-10216-x] is an unsupervised topic modeling method for integrative analysis of single-cell RNA data and pooled CRISPR screening data [@doi:10.1214/07-AOAS114]. The model links the gene expression profile of the cells and specific biological function by delineating perturbation effects,, allowing for better understanding of perturbation functions in single cell CRISPR data. In the perturbation effect prioritizing step, MUSIC utilizes the output from the topic model and estimates individual gene perturbation effects on cell phenotypes. It takes three different schemes in modeling combined single-cell and CRISPR data: an overall perturbation effect which represents the gene perturbation effect, a topic model which specifies the function of perturbation effectsway, and with respect to relationships between different perturbation effects. MUSIC was applied to 14 real single-cell CRISPR screening datasets and accurately quantified and prioritized the individual gene perturbation effect on cell phenotypes, with tolerance for substantial noise.

### Manifold Alignment Methods

Manifold alignment methods aim to infer a lower-dimensional structure within multiple complex datasets (Figure {@fig:3}B). Once this is done, points can be matched across the datasets. This is a very broad class of algorithms, and we here review several representative ones based on distinct ideas, such as the use of pseudotime trajectories, Kernel methods and distance-based matching of cells. The distance-based matching (Figure {@fig:4}) is a general idea containing several different realizations, such as **UNION-Com** [@doi:10.1093/bioinformatics/btaa443], **SCOT** [@doi:10.1101/2020.04.28.066787] and **Pamona** [@doi:10.1093/bioinformatics/btab594], which are reviewed below, among other methods.

![Summary of the distance-based alignment algorithm: cells are represented by nodes in two different graph representations and matched in order to preserve a notion of the distance on the graph.](images/Fig_4.png){#fig:4 width="75%" height="75%"}

**MATCHER** [@doi:10.1186/s13059-017-1269-0] is the first manifold alignment technique to align different forms of single-cell data. Their approach builds on trajectory inference [@doi:10.1038/nbt.2859]. It constructs pseudotime trajectories corresponding to cellular processes for each omic first, and then aligns them between different -omics. Pseudotime trajectory models the corresponding cellular process as a Gaussian process and infers the latent variable corresponding to pseudotime. This results in a set of curves capturing the biological processes, one for each -omics layer. Such curves are then projected onto a reference line so that different cells can be matched across -omics. The model makes a strong assumption that there is only one common biological process to be modeled.

**MMD-MA** [@doi:10.1101/644310], or Maximum Mean Discrepancy - Manifold Alignment, is a completely unsupervised method. The alignment is performed by matching low-dimensional representations of different -omics, constructed through a kernel-based technique that minimizes the MMD (Maximum Mean Discrepancy) [@{https://dl.acm.org/doi/10.5555/2188385.2188410}] between the two datasets. Additionally, the representations are constructed by taking into account the distortion of the distances in the original data while keeping the transformation as simple as possible. The model was evaluated on data containing gene expression and methylation values from the same single cells; the known cell correspondence information was hidden and MMD-MA was able to successfully reconstruct this information.

**UNION-Com** [@doi:10.1093/bioinformatics/btaa443] performs unsupervised alignment of different -omics datasets by matching the structure of the datasets. The idea is that, if different -omics layers indeed correspond to similar samples of cells, then the distance matrices of any two -omics layers will become very similar after rearranging the cell indices. A matching matrix connecting points across datasets is learned by optimizing the similarity of distance matrices after cell permutation. This approach of matching is an extension of GUMA ("Generalized Unsupervised Manifold Alignment") [@{https://dl.acm.org/doi/10.5555/2969033.2969098}] with newly allowed soft matchings. Subsequently, this method performs a version of t-SNE [@{http://www.jmlr.org/papers/v9/vandermaaten08a.html}] adopted for multi-modal data represented in the same latent space. This approach takes the overall structure of all datasets into account while matching the cells, without the requirement of identical distributions of different modalities. UNION-Com compared favorably with Seurat v3 and MMD-MA when evaluated on the quality of labels transferred between gene expression, methylation and chromatin accessibility data.

**SCOT** [@doi:10.1101/2020.04.28.066787] is similar to UNION-Com in terms of distance comparison across the -omics layers. However, it is formulated as a different optimization problem per the theory of optimal transport. It starts by considering k-nearest neighbor graphs in different -omics layers and uses those to compute distances between cells from different -omics, like UNION-Com. The soft matchings are applied here as well, with points matched probabilistically across datasets. Unlike UNION-Com, such matchings are obtained by considering a version of optimal transport given by the Gromov-Wasserstein distance, which generalizes the "earth-mover" Wasserstein distance to optimal transport between different spaces [@doi:10.1007/s10208-011-9093-5]. SCOT compared favorably to MMD-MA and UNION-Com on several real and simulated datasets containing transcriptomic and epigenetic (DNAme or chromatin accessibility) data. The model contains only two hyperparameters, making it particularly simple to tune.

**Pamona** [@doi:10.1093/bioinformatics/btab594] uses a similar approach to SCOT, but with a modification of optimal transport based on Partial Gromov-Wasserstein distance [@doi:10.48550/arXiv.2002.08276], which accounts for data points that do not have appropriate matches across datasets. By doing so, the authors can allow for possible imperfect alignment between the datasets, tolerating cell types present in one dataset only. After the alignment is found, the data corresponding to different modalities is projected down to a dimensionally reduced space using Laplacian Eigenmaps [@doi:10.1162/089976603321780317]. Benchmarked on several datasets containing transcriptomic and epigenetic data, their model outperformed SCOT, MMD-MA and Seurat v3.

### Neural Network-Based Methods

Neural networks, including autoencoders and generative adversarial networks (GAN), have been used for the unsupervised task of the alignment of -omics datasets. Autoencoders have been described earlier. GANs typically consist of two parts: the generator network and the discriminator network. While the generator tries to produce outputs of a form resembling a certain target dataset, the discriminator learns the difference between the generator's outputs and the elements of the target dataset. In this section, we summarize the relevant neural network methods below.

**SCIM** [@doi:10.1093/bioinformatics/btaa843] builds on a multi-domain translation approach [@doi:10.48550/arXiv.1902.03515] to integrate multi-omics data in an unsupervised fashion. It uses a separate variational autoencoder for each modality in order to map the data onto reduced latent space representations. Such representations are then aligned to have a similar structure, by using a discriminator network in addition to autoencoders which learns to distinguish between the latent space representations of different -omics. The two autoencoders and the discriminator network are trained simultaneously, resulting in the two latent spaces being maximally alike. Once both datasets are encoded into approximately corresponding representations, the points with similar latent representations are matched across the datasets. This model was tested on simulations from PROSSTT ("Probabilistic Simulation of Single-Cell RNA-seq Tree-Like Topologies") [@doi:10.1093/bioinformatics/btz078] as well as datasets containing gene expression and proteins, and performed favorably to MATCHER when applied to simulated data exhibiting a complex cellular differentiation process.

**MULTIGRATE** [@doi:10.1101/2022.03.16.484643] uses a multi-modal variational autoencoder structure to project multi-omics data onto
a shared latent space. While somewhat similar to the scMVAE model [@doi:10.1093/bib/bbaa287], this framework brings additional flexibility and can be used for integration of the paired and unpaired single-cell data. Furthermore, this model can integrate data from a multi-omics assay such as CITE-seq with data from a single-omics assay such as scRNA-seq. Data corresponding to different -omics are first passed through separate neural networks, before being combined by the Product of Experts technique [@doi:10.1162/089976602760128018] to form the latent distribution. The decoder networks then aim to reconstruct all of the -omics from this unified representation. To better align cells, Maximum Mean Discrepancy is added to the loss function, penalizing the misalignment between the point clouds belonging to different assays. Their model was used for the creation of multi-modal atlases, and mapping a COVID-19 single-cell dataset onto a multi-modal reference.

**MAGAN** [@doi:10.48550/arXiv.1803.00385] utilizes generative adversarial networks (GANs) to align data from different domains. MAGAN uses two tied GANs to translate between the -omics layers, while tying their parameters and requiring that their combination maps any point onto itself. Namely, if the first generator maps data point A to data point B, then the second generator should map B back to A. It is conceptually very similar to the CycleGAN [@doi:10.48550/arXiv.1703.10593] model from computer vision, but with a key innovation that allowed it to more efficiently align and integrate single-cell data. The novelty here was noting that while the CycleGAN framework was very good at aligning the datasets in aggregate, it would not necessarily correctly match individual points. This is a particularly important problem for single-cell data. To address this problem, MAGAN is augmented with a correspondence loss measuring the difference between points before and after being mapped by generators. This model was tested on a variety of datasets, ranging from a simulated dataset to MNIST handwritten digits to molecular data. The method was applied to combine transcriptomic and proteomic data in single cells. The model was shown to meaningfully align the datasets even when the correspondence information was not available.

### Other Methods

**CCA** (Canonical Correlation Analysis) based methods reduce the dimensionality of data by selecting for the degrees of freedom that are correlated between the datasets. Seurat v3 [@doi:10.1016/j.cell.2019.05.031] combines CCA with network concepts in order to align and integrate single-cell multi-omics data. After performing the CCA, the algorithm identifies anchors between the datasets and scores the quality of those anchors. Anchors are identified by MNNs (mutual nearest neighbors), and their quality is scored by considering the overlap between the neighborhoods of anchors. Similar to Seurat v3, MAESTRO [@doi:10.1186/s13059-020-02116-x] also utilized canonical correlation analysis for the integration of transcriptomic and epigenetic data, and provided a comprehensive analysis pipeline. bindSC [@doi:10.1101/2020.12.11.422014] also uses canonical correlation analysis to construct shared representations of the data, iteratively optimized using a custom procedure.

**LIGER** [@doi:10.1016/j.cell.2019.05.006] performs an iNMF (integrative non-negative matrix factorization) to learn factors explaining the variation within and across datasets. Data such as DNA methylation are first aggregated over genes. Cells corresponding to different datasets are described by separate sets of cell-specific factors. Gene factors consist of two components: one that is shared across datasets and one that is dataset specific; the model aims to make the dataset-specific portion as small as possible. After performing the matrix factorization, the shared factor neighborhood graph is formed, in which cells are connected based on the similarity of their factors, and used for aligning the cells across modalities. Recently, this nonnegative matrix factorization approach has been extended to incorporate the idea of online learning. It iteratively updates the model in real-time, and leads to better scalability and computational efficiency [@doi:10.1038/s41587-021-00867-x].


## Concluding Remarks

The landscape of experimental techniques for -omics sequencing and analyzing the data has grown significantly last few years. Accompanying the thrust of technological advancement, an increasing body of computational methods to handle multi-omics data integration or alignment have been proposed. Geared towards computational biologists and genomics scientists, here we reviewed in-depth and extensively these computational methods by their working principles. Among these methods, AI and machine learning based methods account for the majority, demonstrating the influence in single cell computational biology. Other approaches using matrix factorization and or Bayesian’s methods have also been proposed. As demonstrated in a range of methods, the integration of multi-omics data at the single-cell level improves the quality of downstream biological interpretation steps, such as clustering. With the advent of technologies for sequencing multi-omics data from the same single cells, efficient multi-omics integration methods to provide further biological and medical insights at larger scales will be of continued demand.


Meanwhile the rapidly growing number of computational methods pose an urgent need for benchmarking studies on their performances, in order to provide guidelines to choose appropriate methods for specific datasets. Current comparisons are either incomplete, or using a small set of benchmark datasets, with inconsistent metrics in various studies, impeding the selection of appropriate methods for the dataset to analyze. This is made more difficult by the generally unsupervised nature of the integration task, where commonly required ground truths are not known for certain. Moreover, different methods have different prerequisites regarding preprocessing steps, normalization, etc and as a result, careful consideration of these steps and their impacts on the model performances is needed. Oftentimes, the integration methods were developed with one specific application/assay in mind, generalization of these methods with the emergence of new technologies needs to be demonstrated. Fortunately, some benchmarking studies have been conducted in other sub-fields of single cell computational biology for reference, such as those focused on the integration of data from different cells and atlas study [@doi:10.1101/2020.05.22.111161], cell-type annotation [@doi:10.1016/j.gpb.2020.07.004], and integration algorithms to spatial transcriptomics [@doi:10.1101/2021.08.27.457741]. Creating standardized high-quality benchmarking datasets would aid such efforts, as proposed in [@doi:10.1101/2021.12.08.471773] for scRNA-seq data. Finally, comprehensive and flexible benchmarking pipelines that can accommodate the ever-increasing body of integration methods will be extremely useful, in keeping the field up-to-date on multi-omics integration. One such example is the dynverse [@{https://dynverse.org/}].

## Competing Interests

The authors declare no competing interests.

## Acknowledgements

This work was supported by R01 LM012373 and LM012907 awarded by NLM, and R01 HD084633 awarded by NICHD to L.X. Garmire.


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
