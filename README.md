<div align="center">

<h1> ECG-Agent: On-Device Tool-Calling Agent for ECG Multi-Turn Dialogue </h1>

<h5 align="center"> 

<a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='https://huggingface.co/gustmd0121'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>

<br>

Hyunseung Chung<sup>1</sup>,
Jungwoo Oh<sup>1</sup>,
Daeun Kyung<sup>1</sup>,
Jiho Kim<sup>1</sup>,
Yeonsu Kwon<sup>1</sup>,
Min-Gyu Kim<sup>2</sup>,
Edward Choi<sup>1</sup>

[Hyunseung Chung](https://sites.google.com/view/thschung)<sup>1</sup>,
[Jungwoo Oh](https://github.com/Jwoo5)<sup>1</sup>,
[Daeun Kyung](https://dek924.github.io/)<sup>1</sup>,
[Jiho Kim](https://jiho283.github.io/)<sup>1</sup>,
[Yeonsu Kwon](https://sites.google.com/view/yeonsukwon)<sup>1</sup>,
[Min-Gyu Kim](https://mingyuk.github.io/)<sup>2</sup>
[Edward Choi](https://mp2893.com/)<sup>1</sup>

<sup>1</sup>KAIST <sup>2</sup>Ajou University School of Medicine

<p align="center">
    <img src="figs/overall_figure.png" width="95%">
</p>

</h5>
</div>

## Introduction

Recent advances in Multimodal Large Language Models have rapidly expanded to electrocardiograms, focusing on classification, report generation, and single-turn QA tasks. However, these models fall short in real-world scenarios, lacking multi-turn conversational ability, on-device efficiency, and precise understanding of ECG measurements such as the PQRST intervals.

To address these limitations, we introduce **ECG-Agent**, the first LLM-based tool-calling agent for multi-turn ECG dialogue. To facilitate its development and evaluation, we also present the **ECG-Multi-Turn-Dialogue (ECG-MTD) dataset**, a collection of realistic user-assistant multi-turn dialogues for diverse ECG lead configurations. We develop ECG-Agents in various sizes, from on-device capable (1B, 3B) to larger agents (8B, 32B).

Experimental results show that ECG-Agents outperform baseline ECG-LLMs in response accuracy. Furthermore, on-device agents achieve comparable performance to larger agents in various evaluations that assess response accuracy, tool-calling ability, and hallucinations, demonstrating their viability for real-world applications.
