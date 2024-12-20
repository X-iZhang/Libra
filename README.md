<h1 align="center">
    Libra: Leveraging Temporal Images for Biomedical Radiology Analysis
</h1>

[![arXiv](https://img.shields.io/badge/Arxiv-2411.19378-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19378) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg?)](https://github.com/X-iZhang/Libra/blob/main/LICENSE)
[![Views](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FX-iZhang%2FLibra&count_bg=%2300C0FF&title_bg=%23004080&icon=&icon_color=%23FFFFFF&title=Views)](https://hits.seeyoufarm.com)

<details open><summary>💡 More Than Radiology: Features You’ll Love! ⚒️ </summary><p>
    
>  * **LLaVA-Type & LLaMA3 Support**: Deploy and train advanced models effortlessly.
>  * **Resume Training**: Resume training from checkpoints at any stage, whether for pre-training or fine-tuning.  
>  * **Validation Dataset**: Track model performance in real-time on validation datasets during training. 
>  * **Custom Metrics**: Go beyond `eval_loss` with metrics like `BLEU`, `ROUGE-L`, `RadGraph-F1` or define your own criteria for flexible evaluation.    
>  * **Smart Saving**: Automatically save the best model based on validation loss or custom evaluation scores.

</p></details>

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

## Acknowledgements 🙏

We sincerely thank the following projects for their contributions to **Libra**:

* [LLaVA](https://github.com/haotian-liu/LLaVA): A Large Language and Vision Assistant, laying the groundwork for multimodal understanding.
* [FastChat](https://github.com/lm-sys/FastChat): An Open Platform for Training, Serving, and Evaluating Large Language Model based Chatbots.
* [LLaMA](https://github.com/facebookresearch/llama): Open and efficient foundation language models that inspired our core language processing capabilities.
* [RAD-DINO](https://huggingface.co/microsoft/rad-dino): An open and efficient biomedical image encoder, enabling robust radiological analysis.

## Citation ✒️

If you find our paper and code useful in your research and applications, please cite using this BibTeX:
```BibTeX
@misc{zhang2024libraleveragingtemporalimages,
      title={Libra: Leveraging Temporal Images for Biomedical Radiology Analysis}, 
      author={Xi Zhang and Zaiqiao Meng and Jake Lever and Edmond S. L. Ho},
      year={2024},
      eprint={2411.19378},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.19378}, 
}
```
