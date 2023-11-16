# Awesome-Robotics-Foundation-Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![alt text](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models/blob/main/survey_tree.png)

This is the partner repository for the survey paper "Foundation Models in Robotics: Applications, Challenges, and the Future". The authors hope this repository can act as a quick reference for roboticists who wish to read the relevant papers and implement the associated methods. The organization of this readme follows Figure 1 in the paper (shown above) and is thus divided into foundation models that have been applied to robotics and those that are relevant to robotics in some way.

We welcome contributions to this repository to add more resources. Please submit a pull request if you want to contribute!

## Table of Contents

- [Survey](#survey)
- [Robotics](#robotics)
- [Robot Policy Learning for Decision Making and Controls](#robot-policy-learning-for-decision-making-and-controls)
- [Language-Image Goal-Conditioned Value Learning](#language-image-goal-conditioned-value-learning)
- [Robot Task Planning Using Large Language Models](#robot-task-planning-using-large-language-models)
- [In-context Learning for Decision-Making](#in-context-learning-for-decision-making)
- [Robot Transformers](#robot-transformers)
- [Open-Vocabulary Robot Navigation and Manipulation](#open-vocabulary-robot-navigation-and-manipulation)
- [Relevant to Robotics](#relevant-to-robotics)
- [Open-Vocabulary Object Detection and 3D Classification](#open-vocabulary-object-detection-and-3D-classification)
- [Open-Vocabulary Semantic Segmentation](#open-vocabulary-semantic-segmentation)
- [Open-Vocabulary 3D Scene Representations](#open-vocabulary-3D-scene-representations)
- [Open-Vocabulary Object Representations](#open-vocabulary-object-representations)
- [Affordance Information](#affordance-information)
- [Predictive Models](#predictive-models)
- [Generalist AI](#generalist-AI)
- [Simulators](#simulators)


## Survey

This repository is largely based on the following paper:

**[Foundation Models in Robotics: Applications, Challenges, and the Future]()**
<br />
Roya Firoozi
Jiankai Sun,
Johnathan Tucker,
Anirudha Majumdar,
Yuke Zhu,
Shuran Song,
Ashish Kapoor,
Weiyu Liu,
Stephen Tian,
Karol Hausman,
Brian Ichter,
Danny Driess,
Jiajun Wu,
Cewu Lu,
Mac Schwager
<br />

If you find this repository helpful, please consider citing:

## Robotics

### Robot Policy Learning for Decision-Making and Controls
#### Language-Conditioned Imitation Learning
* CLIPort: What and Where Pathways for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2109.12098)[[Project]](https://cliport.github.io/)[[Code]](https://github.com/cliport/cliport)
* Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2209.05451)[[Project]](https://peract.github.io/)[[Code]](https://github.com/peract/peract)
* Play-LMP: Learning Latent Plans from Play [[Project]](https://learning-from-play.github.io/)
* Multi-Context Imitation: Language-Conditioned Imitation Learning over Unstructured Data [[Project]](https://language-play.github.io)
#### Language-Assisted Reinforcement Learning
* Towards A Unified Agent with Foundation Models [[Paper]](https://arxiv.org/abs/2307.09668)
* Reward Design with Language Models [[Paper]](https://arxiv.org/abs/2303.00001)
* Learning to generate better than your llm [[Paper]](https://arxiv.org/pdf/2306.11816.pdf)[[Code]](https://github.com/Cornell-RL/tril)
* Guiding Pretraining in Reinforcement Learning with Large Language Models [[Paper]](https://arxiv.org/abs/2302.06692)[[Code]](https://github.com/yuqingd/ellm)

### Language-Image Goal-Conditioned Value Learning

* SayCan: Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[Paper]](https://arxiv.org/abs/2204.01691)[[Project]](https://say-can.github.io/)[[Code]](https://github.com/google-research/google-research/tree/master/saycan)
* Zero-Shot Reward Specification via Grounded Natural Language[[Paper]](https://proceedings.mlr.press/v162/mahmoudieh22a/mahmoudieh22a.pdf)
* VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models[[Project]](https://voxposer.github.io)
* VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training [[Paper]](https://arxiv.org/abs/2210.00030)[[Project]](https://sites.google.com/view/vip-rl)
* 
* 
* VIMA: General Robot Manipulation with Multimodal Prompts [[Paper]](https://arxiv.org/abs/2210.03094)[[Project]](https://vimalabs.github.io/)[[Code]](https://github.com/vimalabs/VIMA)
* Text2Motion: From Natural Language Instructions to Feasible Plans [[Paper]](https://arxiv.org/abs/2303.12153)[[Project]](https://sites.google.com/stanford.edu/text2motion)
* ChatGPT for Robotics: Design Principles and Model Abilities [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf)[[Blog]](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)[[Code]](https://github.com/microsoft/PromptCraft-Robotics)
* Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents [[Paper]](https://arxiv.org/pdf/2201.07207.pdf)[[Project]](https://wenlong.page/language-planner/)[[Code]](https://github.com/huangwl18/language-planner)
* Planning with Large Language Models via Corrective Re-prompting [[Paper]](https://arxiv.org/abs/2211.09935) 
* ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding [[Paper(ULIP 1)]](https://arxiv.org/abs/2212.05171)[[Paper(ULIP 2)]](https://arxiv.org/abs/2305.08275)[[Project]](https://tycho-xue.github.io/ULIP/)[[Code]](https://github.com/salesforce/ULIP)

### LLM-Based Code Generation
* Paper: ProgPrompt: Generating Situated Robot Task Plans using Large Language Models [[Paper]](https://arxiv.org/abs/2209.11302)[[Project]](https://progprompt.github.io/) 



### Robot Task Planning Using Large Language Models

* NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models [[Paper]](https://arxiv.org/pdf/2305.07766.pdf)[[Project]](https://yongchao98.github.io/MIT-realm-NL2TL/)[[Code]](https://github.com/yongchao98/NL2TL)

### In-context Learning for Decision-Making

### Robot Transformers

* RT-1: Robotics Transformer for Real-World Control at Scale [[Paper]](https://robotics-transformer.github.io/assets/rt1.pdf)[[Project]](https://robotics-transformer.github.io/)[[Code]](https://github.com/google-research/robotics_transformer)

### Open-Vocabulary Robot Navigation and Manipulation
* CoWs on PASTURE: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation [[Paper]](https://arxiv.org/pdf/2203.10421.pdf)[[Project]](https://cow.cs.columbia.edu/)[[Code]]()
* LSC: Language-guided Skill Coordination for Open-Vocabulary Mobile Pick-and-Place [[Paper]]()[[Project]](https://languageguidedskillcoordination.github.io/)
* L3MVN: Leveraging Large Language Models for Visual Target Navigation [[Project]](https://arxiv.org/abs/2304.05501)
* Open-World Object Manipulation using Pre-trained Vision-Language Models [[Paper]](https://robot-moo.github.io/assets/moo.pdf)[[Project]](https://robot-moo.github.io/) 

## Relevant to Robotics

### Open-Vocabulary Object Detection and 3D Classification
*  Grounded Language-Image Pre-training [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.pdf)[[Code]]()

### Open-Vocabulary Semantic Segmentation
* Language-driven Semantic Segmentation [[Paper]](https://arxiv.org/abs/2201.03546)[[Code]]()
* Emerging Properties in Self-Supervised Vision Transformers [[Paper]](https://arxiv.org/abs/2104.14294)[[Code]](https://github.com/facebookresearch/dino)

### Open-Vocabulary 3D Scene Representations
* LERF: Language Embedded Radiance Fields [[Paper]](https://arxiv.org/abs/2303.09553)[[Project]](https://www.lerf.io/)[[Code]](https://github.com/kerrj/lerf)


### Open-Vocabulary Object Representations

### Affordance Information


### Predictive Models

### Embodied AI

* Voyager: An Open-Ended Embodied Agent with Large Language Models [[Paper]](https://arxiv.org/abs/2305.16291)[[Project]](https://voyager.minedojo.org/)[[Code]](https://github.com/MineDojo/Voyager)
* Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory [[Paper]](https://arxiv.org/abs/2305.17144)[[Code]]()
* EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought [[Paper]](https://arxiv.org/abs/2305.15021)[[Project]](https://embodiedgpt.github.io/)

### Generalist AI

### Simulators
