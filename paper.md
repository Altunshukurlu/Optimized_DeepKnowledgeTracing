---
title: 'Modernizing Deep Knowledge Tracing: A PyTorch Package with Gated Architectures and Adaptive Optimization'
tags:
  - deep learning
  - knowledge tracing
  - LSTM
  - GRU
  - optimization algorithms
authors:
  - name: Altun Shukurlu
    affiliation: 1
    orcid: 0009-0004-0419-5586
    corresponding: true
affiliations:
  - index: 1
    name: University of Virginia, United States
date: 20 June 2025
bibliography: paper.bib
---

# Summary

**Modern Deep Knowledge Tracing (DKT)** is an open-source Python package for modeling student learning behavior over time using deep neural networks. It revisits and improves the widely-cited DKT model, which uses recurrent neural networks (RNNs) to predict students' future responses based on past interactions with learning content. The original DKT implementation was developed using the now-deprecated Lua Torch framework and standard RNNs, limiting its usability in modern AI research and applications.

Our software provides a fully re-implemented DKT framework in PyTorch with two major improvements:
1. Replacing standard RNNs with more effective gated architectures (LSTM and GRU), which better model long-term dependencies.
2. Integrating a modular optimization benchmarking suite that supports SGD, RMSProp, Adagrad, Adam, and AdamW for more stable and efficient training.

This package is designed for education researchers, data scientists, and EdTech practitioners looking to build adaptive learning systems, evaluate temporal modeling strategies, or benchmark deep learning models on educational datasets. Full research article and findings can be found on [Arxiv](https://arxiv.org/abs/2504.20070)

# Statement of need

Student modeling and personalized education require accurate, scalable tools to predict student performance over time. Deep Knowledge Tracing, first introduced by [@piech2015deep], marked a shift from probabilistic models to neural architectures in educational data mining. However, its original implementation lacks modern infrastructure and does not exploit advances in gated networks or optimization techniques.

Our package fills this gap by providing:
- A PyTorch-based, extensible, and reproducible codebase.
- Support for gated recurrent units (LSTM/GRU) to improve performance and training stability.
- Optimization benchmarking for choosing effective training strategies under different data regimes.
- Built-in datasets (Synthetic-5) and training pipelines for rapid experimentation.

Compared to alternatives like DKVMN or SAKT, this tool provides a lightweight, baseline-friendly setup specifically tailored for educational data, while allowing extension to attention- or graph-based variants.

# Acknowledgements

We acknowledge the original DKT authors and the open-source contributors behind PyTorch.

# References

