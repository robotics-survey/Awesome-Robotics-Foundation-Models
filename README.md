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

### Robot Policy Learning for Decision Making and Controls

* Paper: Text2Motion: From Natural Language Instructions to Feasible Plans [[Paper]](https://arxiv.org/abs/2303.12153)[[Project]](https://sites.google.com/stanford.edu/text2motion)
* 
* Grounded Language-Image Pre-training [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.pdf)[[Code]]()
* Open-World Object Manipulation using Pre-trained Vision-Language Models [[Paper]](https://robot-moo.github.io/assets/moo.pdf)[[Project]](https://robot-moo.github.io/) 
* ConceptFusion: Open-set Multimodal 3D Mapping [[Paper]](https://concept-fusion.github.io/assets/pdf/2023-ConceptFusion.pdf)[[Project]](https://concept-fusion.github.io/)[[Code]](https://github.com/concept-fusion/concept-fusion)
* LERF: Language Embedded Radiance Fields [[Paper]](https://arxiv.org/abs/2303.09553)[[Project]](https://www.lerf.io/)[[Code]](https://github.com/kerrj/lerf)
* RT-1: Robotics Transformer for Real-World Control at Scale [[Paper]](https://robotics-transformer.github.io/assets/rt1.pdf)[[Project]](https://robotics-transformer.github.io/)[[Code]](https://github.com/google-research/robotics_transformer)
* BusyBot: Learning to Interact, Reason, and Plan in a BusyBoard Environment [[Paper]](https://arxiv.org/abs/2207.08192)[[Project]](https://busybot.cs.columbia.edu/)[[Code]](https://github.com/columbia-ai-robotics/BusyBot)
* NeuPSL: Neural Probabilistic Soft Logic [[Paper]](https://arxiv.org/pdf/2205.14268.pdf) 
* Any-to-Any Generation via Composable Diffusion [[Paper]](https://arxiv.org/abs/2305.11846)[[Project]](https://codi-gen.github.io/)[[Code]](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)
* SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion [[Paper]](https://arxiv.org/pdf/2209.03855.pdf)[[Project]](https://sites.google.com/view/se3dif)[[Code]](https://github.com/TheCamusean/grasp_diffusion)
* Text2NeRF: Text-Driven 3D Scene Generation with Neural Radiance Fields [[Paper]](https://arxiv.org/abs/2305.11588) 
* Voyager: An Open-Ended Embodied Agent with Large Language Models [[Paper]](https://arxiv.org/abs/2305.16291)[[Project]](https://voyager.minedojo.org/)[[Code]](https://github.com/MineDojo/Voyager)
* Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory [[Paper]](https://arxiv.org/abs/2305.17144)[[Code]]()
* NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models [[Paper]](https://arxiv.org/pdf/2305.07766.pdf)[[Project]](https://yongchao98.github.io/MIT-realm-NL2TL/)[[Code]](https://github.com/yongchao98/NL2TL)
* CLIPort: What and Where Pathways for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2109.12098)[[Project]](https://cliport.github.io/)[[Code]](https://github.com/cliport/cliport)
* Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2209.05451)[[Project]](https://peract.github.io/)[[Code]](https://github.com/peract/peract)
* Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[Paper]](https://arxiv.org/abs/2204.01691)[[Project]](https://say-can.github.io/)[[Code]](https://github.com/google-research/google-research/tree/master/saycan)
* VIMA: General Robot Manipulation with Multimodal Prompts [[Paper]](https://arxiv.org/abs/2210.03094)[[Project]](https://vimalabs.github.io/)[[Code]](https://github.com/vimalabs/VIMA)
* ChatGPT for Robotics: Design Principles and Model Abilities [[Paper]](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf)[[Blog]](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)[[Code]](https://github.com/microsoft/PromptCraft-Robotics)
* Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents [[Paper]](https://arxiv.org/pdf/2201.07207.pdf)[[Project]](https://wenlong.page/language-planner/)[[Code]](https://github.com/huangwl18/language-planner)
* L3MVN: Leveraging Large Language Models for Visual Target Navigation [[Project]](https://arxiv.org/abs/2304.05501)
* Planning with Large Language Models via Corrective Re-prompting [[Paper]](https://arxiv.org/abs/2211.09935) 
* EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought [[Paper]](https://arxiv.org/abs/2305.15021)[[Project]](https://embodiedgpt.github.io/)
* LSC: Language-guided Skill Coordination for Open-Vocabulary Mobile Pick-and-Place [[Paper]]()[[Project]](https://languageguidedskillcoordination.github.io/)
* ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding [[Paper(ULIP 1)]](https://arxiv.org/abs/2212.05171)[[Paper(ULIP 2)]](https://arxiv.org/abs/2305.08275)[[Project]](https://tycho-xue.github.io/ULIP/)[[Code]](https://github.com/salesforce/ULIP) 

### Language-Image Goal-Conditioned Value Learning

### Robot Task Planning Using Large Language Models

### In-context Learning for Decision-Making

### Robot Transformers

### Open-Vocabulary Robot Navigation and Manipulation

## Relevant to Robotics

### Open-Vocabulary Object Detection and 3D Classification

### Open-Vocabulary Semantic Segmentation

### Open-Vocabulary 3D Scene Representations

### Open-Vocabulary Object Representations

### Affordance Information

### Predictive Models

### Generalist Ai

### Simulators
