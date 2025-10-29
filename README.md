<div align="center">

<h1> ECG-Agent: On-Device Tool-Calling Agent for ECG Multi-Turn Dialogue </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

<h5 align="center"> 

[Homepage-TBA] &nbsp;
[Paper-TBA] &nbsp;
[Models-TBA] &nbsp;
[Dataset-TBA]

<br>

Hyunseung Chung<sup>1</sup>,
Jungwoo Oh<sup>1</sup>,
Daeun Kyung<sup>1</sup>,
Jiho Kim<sup>1</sup>,
Yeonsu Kwon<sup>1</sup>,
Min-Gyu Kim<sup>2</sup>,
Edward Choi<sup>1</sup>

<sup>1</sup>KAIST <sup>2</sup>Ajou University School of Medicine

</h5>

<p align="center">
    <img src="figures/overall_figure_0917_final.png" width="90%">
</p>
</div>

## Introduction

Recent advances in Multimodal Large Language Models have rapidly expanded to electrocardiograms, focusing on classification, report generation, and single-turn QA tasks. However, these models fall short in real-world scenarios, lacking multi-turn conversational ability, on-device efficiency, and precise understanding of ECG measurements such as the PQRST intervals.

To address these limitations, we introduce **ECG-Agent**, the first LLM-based tool-calling agent for multi-turn ECG dialogue. To facilitate its development and evaluation, we also present the **ECG-Multi-Turn-Dialogue (ECG-MTD) dataset**, a collection of realistic user-assistant multi-turn dialogues for diverse ECG lead configurations. We develop ECG-Agents in various sizes, from on-device capable (1B, 3B) to larger agents (8B, 32B).

Experimental results show that ECG-Agents outperform baseline ECG-LLMs in response accuracy. Furthermore, on-device agents achieve comparable performance to larger agents in various evaluations that assess response accuracy, tool-calling ability, and hallucinations, demonstrating their viability for real-world applications.

## Setup

```shell
# Clone the repository
git clone [https://github.com/YOUR-USERNAME/ECG-Agent.git](https://github.com/YOUR-USERNAME/ECG-Agent.git)
cd ECG-Agent

# Install dependencies
pip install -r requirements.txt
