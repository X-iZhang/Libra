<h1 align="center">
    <a href="https://arxiv.org/abs/2411.19378">Libra: Leveraging Temporal Images for Biomedical Radiology Analysis</a>
</h1>

[![arXiv](https://img.shields.io/badge/Arxiv-2411.19378-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19378) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg?)](https://github.com/X-iZhang/Libra/blob/main/LICENSE) 
![architecture](./assets/libra_architecture.png)

## Overview 🔬
We propose **Libra** (**L**everaging Temporal **I**mages for **B**iomedical **R**adiology **A**nalysis), a novel framework tailored for radiology report generation (RRG) that incorporates temporal change information to address the challenges of interpreting medical images effectively.

Libra leverages RAD-DINO, a pre-trained visual transformer, as its image encoder to generate robust and scalable image features. These features are further refined by a **Temporal Alignment Connector (TAC)**, a key innovation in Libra's architecture. The TAC comprises:
* **Layerwise Feature Extractor (LFE)**: Captures high-granularity image feature embeddings from the encoder.
* **Temporal Fusion Module (TFM)**: Integrates temporal references from prior studies to enhance temporal awareness and reasoning.

These refined features are fed into Meditron, a specialised medical large language model (LLM), to generate comprehensive, temporally-aware radiology reports. Libra’s modular design seamlessly integrates state-of-the-art open-source pre-trained models for both image and text, aligning them through a temporal-aware adapter to ensure robust cross-modal reasoning and understanding.

Through a two-stage training strategy, Libra demonstrates the powerful potential of multimodal large language models (MLLMs) in specialised radiology applications. Extensive experiments on the **MIMIC-CXR dataset** highlight Libra's performance, setting a new state-of-the-art benchmark among models of the same parameter scale.

## Contributions 🛠

* **Temporal Awareness**: Libra captures and synthesises temporal changes in medical images, addressing the challenge of handling prior study citations in RRG tasks.
* **Innovative Architecture**: The Temporal Alignment Connector (TAC) ensures high-granularity feature extraction and temporal integration, significantly enhancing cross-modal reasoning capabilities.
* **State-of-the-Art Performance**: Libra achieves outstanding results on the MIMIC-CXR dataset, outperforming existing MLLMs in both accuracy and temporal reasoning.
 
## Project Status 🚀

The code is currently being organised and will be available soon. **Please check back later for updates!**

We are actively preparing the repository to ensure a seamless experience for contributors and users. Stay tuned for the initial release and future enhancements.

