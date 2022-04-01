---
title: Manuscript Title
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2022-04-01'
author-meta:
- John Doe
- Jane Roe
header-includes: |-
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta name="dc.title" content="Manuscript Title" />
  <meta name="citation_title" content="Manuscript Title" />
  <meta property="og:title" content="Manuscript Title" />
  <meta property="twitter:title" content="Manuscript Title" />
  <meta name="dc.date" content="2022-04-01" />
  <meta name="citation_publication_date" content="2022-04-01" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="John Doe" />
  <meta name="citation_author_institution" content="Department of Something, University of Whatever" />
  <meta name="citation_author_orcid" content="XXXX-XXXX-XXXX-XXXX" />
  <meta name="twitter:creator" content="@johndoe" />
  <meta name="citation_author" content="Jane Roe" />
  <meta name="citation_author_institution" content="Department of Something, University of Whatever" />
  <meta name="citation_author_institution" content="Department of Whatever, University of Something" />
  <meta name="citation_author_orcid" content="XXXX-XXXX-XXXX-XXXX" />
  <link rel="canonical" href="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta property="og:url" content="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta property="twitter:url" content="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta name="citation_fulltext_html_url" content="https://codingpoppy.github.io/multiomics_review_manubot/" />
  <meta name="citation_pdf_url" content="https://codingpoppy.github.io/multiomics_review_manubot/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://codingpoppy.github.io/multiomics_review_manubot/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://codingpoppy.github.io/multiomics_review_manubot/v/3660c1cb8fea0cc111c4b9beaba25c38968917d9/" />
  <meta name="manubot_html_url_versioned" content="https://codingpoppy.github.io/multiomics_review_manubot/v/3660c1cb8fea0cc111c4b9beaba25c38968917d9/" />
  <meta name="manubot_pdf_url_versioned" content="https://codingpoppy.github.io/multiomics_review_manubot/v/3660c1cb8fea0cc111c4b9beaba25c38968917d9/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
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
([permalink](https://codingpoppy.github.io/multiomics_review_manubot/v/3660c1cb8fea0cc111c4b9beaba25c38968917d9/))
was automatically generated
from [codingpoppy/multiomics_review_manubot@3660c1c](https://github.com/codingpoppy/multiomics_review_manubot/tree/3660c1cb8fea0cc111c4b9beaba25c38968917d9)
on April 1, 2022.
</em></small>

## Authors



+ **John Doe**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [XXXX-XXXX-XXXX-XXXX](https://orcid.org/XXXX-XXXX-XXXX-XXXX)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [johndoe](https://github.com/johndoe)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [johndoe](https://twitter.com/johndoe)<br>
  <small>
     Department of Something, University of Whatever
     · Funded by Grant XXXXXXXX
  </small>

+ **Jane Roe**<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [XXXX-XXXX-XXXX-XXXX](https://orcid.org/XXXX-XXXX-XXXX-XXXX)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [janeroe](https://github.com/janeroe)<br>
  <small>
     Department of Something, University of Whatever; Department of Whatever, University of Something
  </small>



## Abstract {.page_break_before}
Recently developed technologies to generate single-cell genomic data have made a revolutionary impact in the field of biology. Multi-omics assays offer even greater opportunities to understand cellular states and biological processes. However, the problem of integrating different -omics data with very different dimensionality and statistical properties remains quite challenging. A growing body of computational tools are being developed for this task, leveraging ideas ranging from machine translation to the theory of networks and representing a new frontier on the interface of biology and data science. Our goal in this review paper is to provide a comprehensive, up-to-date survey of computational techniques for the integration of multi-omics and alignment of multiple modalities of genomics data in the single cell research field.


## Introduction {.page_break_before}
Single-cell sequencing technologies have opened the door to investigating biological processes at an unprecedentedly high resolution. Techniques such as DROP-seq [@doi:10.1016/j.cell.2015.05.002] and 10x Genomics assays are capable of measuring single-cell gene expression, or scRNA-seq, in tens of thousands of single cells simultaneously. Measurements of other data modalities are also increasingly available. For example, single-cell ATAC-seq (scATAC-seq) assesses chromatin accessibility, and single-cell bisulfite sequencing captures DNA methylation, all from single cells. However, many of such techniques are designed to measure a single modality and do not lend themselves to multi-omics measurements. The way to combine information from such measurements is then to assay different -omics from different subsets of the same samples. By assuming that cells assayed by different techniques share similar properties, one can then use alignment methods to computationally aggregate similar cells across different omics assays and draw consensus biological inference.

Recently, however, a number of experimental techniques capable of assaying multiple modalities simultaneously from the same set of single cells have been developed. CITE-seq [@doi:10.1038/nmeth.4380] and REAP-seq [@doi:10.1038/nbt.3973] measure proteins and gene expression. SNARE-seq [@doi:10.1038/nbt.3973; @doi:10.1038/s41587-019-0290-0], SHARE-seq [@doi:10.1038/s41576-020-00308-6] and sci-CAR [@doi:10.1126/science.aau0730] measure gene expression and chromatin accessibility, while scGEM [@doi: 10.1038/nmeth.3961] measures gene expression and DNA methylation. For triple-omics data generation, scNMT [@doi:10.1038/s41467-018-03149-4] measures gene expression, chromatin accessibility and DNA methylation, and scTrio-seq [@doi:10.1038/nmeth.3961; @doi:10.1126/science.aao3791] captures SNPs, gene expression and DNA methylation simultaneously. Integrative analysis of such data obtained from the same cells remains a challenging computational task due to a combination of reasons, such as the noise and sparsity in the assays, and different statistical distributions for different modalities. For clarity, we distinguish between integration methods that combine multiple -omics data from the set of the same single cells (Section I), from alignment methods designed to work with multi-modal data coming from the same tissue but different cells (Section II). The difference in their approaches is shown in Figure. {@fig-1}.

![Multi-omics data can sometimes be sequenced from the same set of single cells (left); at other times, only the data sequenced from the same/similar sample, but different single cells are available (right). In the former case, we have the task of integrating the different data modalities (left); in the latter case, we need to first identify similar cells across the samples (right) - this is the computational task of alignment.](images/Fig_1.png){#fig-1 tag="F1" width="100%" height="100%"}

The application of data fusion algorithms for multi-omics sequencing data predates the single-cell technologies; bulk-level data have been integrated using a variety of computational tools as reviewed in [@doi:10.3389/fgene.2017.00084]. In this review, we aim to give a comprehensive, up-to-date summary of existing computational tools of multi-omics data integration and alignment in the single-cell field, for researchers in the field of computational biology. For more general surveys, the readers are encouraged to check other single-cell multi-omics reviews [@doi:10.1016/j.coisb.2018.01.003; @doi:10.1016/j.tibtech.2020.02.013; @doi:10.1093/bib/bbaa042; @doi:10.1038/s41587-021-00895-7; @doi:10.1016/j.csbj.2021.04.060; @doi:10.1038/s41581-021-00463-x].


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>
