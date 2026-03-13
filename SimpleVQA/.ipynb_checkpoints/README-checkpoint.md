---
license: apache-2.0
task_categories:
- visual-question-answering
pretty_name: SimpleVQA
---
# SimpleVQA
### SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models

**Dataset:** https://huggingface.co/datasets/m-a-p/SimpleVQA

## Abstract
The increasing application of multi-modal large language models (MLLMs) across various sectors have spotlighted the essence of their output reliability and accuracy, particularly their ability to produce content grounded in factual information (e.g. common and domain-specific knowledge). In this work, we introduce SimpleVQA, the first comprehensive multi-modal benchmark to evaluate the factuality ability of MLLMs to answer natural language short questions. SimpleVQA is characterized by six key features: it covers multiple tasks and multiple scenarios, ensures high quality and challenging queries, maintains static and timeless reference answers, and is straightforward to evaluate. Our approach involves categorizing visual question-answering items into 9 different tasks around objective events or common knowledge and situating these within 9 topics. Rigorous quality control processes are implemented to guarantee high-quality, concise, and clear answers, facilitating evaluation with minimal variance via an LLM-as-a-judge scoring system. Using SimpleVQA, we perform a comprehensive assessment of leading 18 MLLMs and 8 text-only LLMs, delving into their image comprehension and text generation abilities by identifying and analyzing error cases.

## Dataset Building

![](images/benchmarks.png)

![image_list](images/image_list.png)

![](images/dataset_statistics.png)

## Main Results

![](images/llm_res.png)

![mllm1](images/mllm1.png)

![mllm2](images/mllm2.png)

![trace](images/trace.png)

## Citation

Please consider citing this work in your publications if it helps your research.
```tex
@misc{cheng2025simplevqamultimodalfactualityevaluation,
      title={SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models}, 
      author={Xianfu Cheng and Wei Zhang and Shiwei Zhang and Jian Yang and Xiangyuan Guan and Xianjie Wu and Xiang Li and Ge Zhang and Jiaheng Liu and Yuying Mai and Yutao Zeng and Zhoufutu Wen and Ke Jin and Baorui Wang and Weixiao Zhou and Yunhong Lu and Tongliang Li and Wenhao Huang and Zhoujun Li},
      year={2025},
      eprint={2502.13059},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.13059}, 
}
```

## Acknowledgements

- [https://openai.com/index/introducing-simpleqa/](https://openai.com/index/introducing-simpleqa/)
- [https://openstellarteam.github.io/ChineseSimpleQA/](https://openstellarteam.github.io/ChineseSimpleQA/)