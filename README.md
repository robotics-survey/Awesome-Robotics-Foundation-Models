# Awesome-Robotics-Foundation-Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![alt text](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models/blob/main/survey_tree.png)

This is the partner repository for the survey paper "Foundation Models in Robotics: Applications, Challenges, and the Future". The authors hope this repository can act as a quick reference for roboticists who wish to read the relevant papers and implement the associated methods. The organization of this readme follows Figure 1 in the paper (shown above) and is thus divided into foundation models that have been applied to robotics and those that are relevant to robotics in some way.

We welcome contributions to this repository to add more resources. Please submit a pull request if you want to contribute!

## Table of Contents

- [Survey](#survey)
- [Robotics](#robotics)
- [Neural Scaling Laws for embodied AI](#neural-scaling-laws-for-embodied-AI)
- [Robot Policy Learning for Decision Making and Controls](#robot-policy-learning-for-decision-making-and-controls)
- [Language-Image Goal-Conditioned Value Learning](#language-image-goal-conditioned-value-learning)
- [Robot Task Planning Using Large Language Models](#robot-task-planning-using-large-language-models)
- [Robot Transformers](#robot-transformers)
- [In-context Learning for Decision-Making](#in-context-learning-for-decision-making)
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
Roya Firoozi,
Johnathan Tucker,
Stephen Tian,
Anirudha Majumdar,
Jiankai Sun,
Weiyu Liu,
Yuke Zhu,
Shuran Song,
Ashish Kapoor,
Karol Hausman,
Brian Ichter,
Danny Driess,
Jiajun Wu,
Cewu Lu,
Mac Schwager
<br />

If you find this repository helpful, please consider citing:
```bibtex
@article{firoozi2024foundation,
  title={Foundation Models in Robotics: Applications, Challenges, and the Future},
  author={Firoozi, Roya and Tucker, Johnathan and Tian, Stephen and Majumdar, Anirudha and Sun, Jiankai and Liu, Weiyu and Zhu, Yuke and Song, Shuran and Kapoor, Ashish and Hausman, Karol and others},
  journal={The International Journal of Robotics Research},
  year={2024}, 
  doi= {https://doi.org/10.1177/02783649241281508}
}
```


```bibtex
@article{firoozi2023foundation,
  title={Foundation Models in Robotics: Applications, Challenges, and the Future},
  author={Firoozi, Roya and Tucker, Johnathan and Tian, Stephen and Majumdar, Anirudha and Sun, Jiankai and Liu, Weiyu and Zhu, Yuke and Song, Shuran and Kapoor, Ashish and Hausman, Karol and others},
  journal={arXiv preprint arXiv:2312.07843},
  year={2023}
}
```


## Robotics

### Neural Scaling Laws
* Neural Scaling Laws for Embodied AI: Neural Scaling Laws for Embodied AI [[Paper]](https://arxiv.org/abs/2405.14005)


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
* Motif: Intrinsic Motivation from Artificial Intelligence Feedback [[Paper]](https://arxiv.org/abs/2310.00166)[[Code]](https://github.com/facebookresearch/motif)

### Language-Image Goal-Conditioned Value Learning
* SayCan: Do As I Can, Not As I Say: Grounding Language in Robotic Affordances [[Paper]](https://arxiv.org/abs/2204.01691)[[Project]](https://say-can.github.io/)[[Code]](https://github.com/google-research/google-research/tree/master/saycan)
* Zero-Shot Reward Specification via Grounded Natural Language [[Paper]](https://proceedings.mlr.press/v162/mahmoudieh22a/mahmoudieh22a.pdf)
* VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models [[Project]](https://voxposer.github.io)
* VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training [[Paper]](https://arxiv.org/abs/2210.00030)[[Project]](https://sites.google.com/view/vip-rl)
* LIV: Language-Image Representations and Rewards for Robotic Control [[Paper]](https://arxiv.org/abs/2306.00958)[[Project]](https://penn-pal-lab.github.io/LIV/)
* LOReL: Learning Language-Conditioned Robot Behavior from Offline Data and Crowd-Sourced Annotation [[Paper]](https://arxiv.org/abs/2109.01115)[[Project]](https://sites.google.com/view/robotlorel)
* Text2Motion: From Natural Language Instructions to Feasible Plans [[Paper]](https://arxiv.org/abs/2303.12153)[[Project]](https://sites.google.com/stanford.edu/text2motion)
* MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control [[Paper]](https://arxiv.org/pdf/2403.12037.pdf)[[Project]](https://sites.google.com/view/minedreamer/main)[[Code]](https://github.com/Zhoues/MineDreamer)

### Robot Task Planning Using Large Language Models
* Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents [[Paper]](https://arxiv.org/abs/2201.07207)[[Project]](https://wenlong.page/language-planner/)
* Open-vocabulary Queryable Scene Representations for Real World Planning (NLMap) [[Paper]](https://arxiv.org/pdf/2209.09874.pdf)[[Project]](https://nlmap-saycan.github.io/)
* NL2TL: Transforming Natural Languages to Temporal Logics using Large Language Models [[Paper]](https://arxiv.org/pdf/2305.07766.pdf)[[Project]](https://yongchao98.github.io/MIT-realm-NL2TL/)[[Code]](https://github.com/yongchao98/NL2TL)
* AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers[[Paper]](https://arxiv.org/abs/2306.06531)[[Project]](https://yongchao98.github.io/MIT-REALM-AutoTAMP/)
* LATTE: LAnguage Trajectory TransformEr [[Paper]](https://arxiv.org/abs/2208.02918)[[Code]](https://github.com/arthurfenderbucker/LaTTe-Language-Trajectory-TransformEr)
* Planning with Large Language Models via Corrective Re-prompting [[Paper]](https://arxiv.org/abs/2211.09935)
* Describe, explain, plan and select: interactive planning with LLMs enables open-world multi-task agents [[Paper]](https://arxiv.org/pdf/2302.01560.pdf)[[Code]](https://github.com/CraftJarvis/MC-Planner)
* JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models [[Paper]](https://arxiv.org/pdf/2311.05997.pdf)[[Project]](https://craftjarvis.github.io/JARVIS-1/)[[Code]](https://github.com/CraftJarvis/JARVIS-1)
* An Embodied Generalist Agent in 3D World [[Paper]](https://arxiv.org/pdf/2311.12871.pdf)[[Project]](https://embodied-generalist.github.io/)[[Code]](https://github.com/embodied-generalist/embodied-generalist)
* LLM+P: Empowering Large Language Models with Optimal Planning Proficiency [[Paper]](https://arxiv.org/pdf/2304.11477.pdf)[[Code]](https://github.com/Cranial-XIX/llm-pddl)
* MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception [[Paper]](https://arxiv.org/pdf/2312.07472.pdf)[[Project]](https://iranqin.github.io/MP5.github.io/)[[Code]](https://github.com/IranQin/MP5)
* Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection [[Paper]](https://arxiv.org/abs/2412.04455) [[Project]](https://zhoues.github.io/Code-as-Monitor/)

### LLM-Based Code Generation
* ProgPrompt: Generating Situated Robot Task Plans using Large Language Models [[Paper]](https://arxiv.org/abs/2209.11302)[[Project]](https://progprompt.github.io/)
* Code as Policies: Language Model Programs for Embodied Control [[Paper]](https://arxiv.org/abs/2209.07753)[[Project]](https://code-as-policies.github.io/)
* ChatGPT for Robotics: Design Principles and Model Abilities [[Paper]](https://arxiv.org/abs/2306.17582)[[Project]](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)[[Code]](https://github.com/microsoft/PromptCraft-Robotics)
* Voyager: An Open-Ended Embodied Agent with Large Language Models [[Paper]](https://arxiv.org/abs/2305.16291)[[Project]](https://voyager.minedojo.org/)
* Visual Programming: Compositional visual reasoning without training [[Paper]](https://arxiv.org/abs/2211.11559)[[Project]](https://prior.allenai.org/projects/visprog)[[Code]](https://github.com/allenai/visprog)
* Deploying and Evaluating LLMs to Program Service Mobile Robots [[Paper]](https://arxiv.org/abs/2311.11183)[[Project]](https://amrl.cs.utexas.edu/codebotler/)[[Code]](https://github.com/ut-amrl/codebotler)

### Robot Transformers
* MotionGPT: Finetuned LLMs are General-Purpose Motion Generators [[Paper]](https://arxiv.org/abs/2306.10900)[[Project]](https://qiqiapink.github.io/MotionGPT/)
* RT-1: Robotics Transformer for Real-World Control at Scale [[Paper]](https://robotics-transformer.github.io/assets/rt1.pdf)[[Project]](https://robotics-transformer.github.io/)[[Code]](https://github.com/google-research/robotics_transformer)
* Masked Visual Pre-training for Motor Control [[Paper]](https://arxiv.org/abs/2203.06173)[[Project]](https://tetexiao.com/projects/mvp)[[Code]](https://github.com/ir413/mvp)
* Real-world robot learning with masked visual pre-training [[Paper]](https://arxiv.org/abs/2210.03109)[[Project]](https://tetexiao.com/projects/real-mvp)
* R3M: A Universal Visual Representation for Robot Manipulation [[Paper]](https://arxiv.org/abs/2203.12601)[[Project]](https://sites.google.com/view/robot-r3m/)[[Code]](https://github.com/facebookresearch/r3m)
* Robot Learning with Sensorimotor Pre-training [[Paper]](https://arxiv.org/abs/2306.10007)[[Project]](https://robotic-pretrained-transformer.github.io/)
* Rt-2: Vision-language-action models transfer web knowledge to robotic control [[Paper]](https://arxiv.org/abs/2307.15818)[[Project]](https://robotics-transformer2.github.io/)
* PACT: Perception-Action Causal Transformer for Autoregressive Robotics Pre-Training [[Paper]](https://arxiv.org/abs/2209.11133)
* GROOT: Learning to Follow Instructions by Watching Gameplay Videos [[Paper]](https://arxiv.org/pdf/2310.08235.pdf)[[Project]](https://craftjarvis.github.io/GROOT/)[[Code]](https://github.com/CraftJarvis/GROOT)
* Behavior Transformers (BeT): Cloning k modes with one stone [[Paper]](https://arxiv.org/abs/2206.11251)[[Project]](https://mahis.life/bet/)[[Code]](https://github.com/notmahi/bet)
* Conditional Behavior Transformers (C-BeT), From Play to Policy: Conditional Behavior Generation from Uncurated Robot Data [[Paper]](https://arxiv.org/abs/2210.10047)[[Project]](https://play-to-policy.github.io/)[[Code]](https://github.com/jeffacce/play-to-policy)
* MAGIC<sup>VFM</sup>: Meta-learning Adaptation for Ground Interaction Control with Visual Foundation Models [[Paper]](https://arxiv.org/abs/2407.12304)

### In-context Learning for Decision-Making
* A Survey on In-context Learning [[Paper]](https://arxiv.org/abs/2301.00234)
* Large Language Models as General Pattern Machines [[Paper]](https://arxiv.org/abs/2307.04721)
* Chain-of-Thought Predictive Control [[Paper]](https://arxiv.org/abs/2304.00776)
* ReAct: Synergizing Reasoning and Acting in Language Models [[Paper]](https://arxiv.org/abs/2210.03629)
* ICRT: In-Context Imitation Learning via Next-Token Prediction [[Paper]](https://arxiv.org/abs/2408.15980) [[Project]](https://icrt.dev/) [[Code]](https://github.com/Max-Fu/icrt)

### Open-Vocabulary Robot Navigation and Manipulation
* CoWs on PASTURE: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation [[Paper]](https://arxiv.org/pdf/2203.10421.pdf)[[Project]](https://cow.cs.columbia.edu/)[[Code]]()
* Open-vocabulary Queryable Scene Representations for Real World Planning (NLMap) [[Paper]](https://arxiv.org/pdf/2209.09874.pdf)[[Project]](https://nlmap-saycan.github.io/)
* LSC: Language-guided Skill Coordination for Open-Vocabulary Mobile Pick-and-Place [[Paper]]()[[Project]](https://languageguidedskillcoordination.github.io/)
* L3MVN: Leveraging Large Language Models for Visual Target Navigation [[Project]](https://arxiv.org/abs/2304.05501)
* Open-World Object Manipulation using Pre-trained Vision-Language Models [[Paper]](https://robot-moo.github.io/assets/moo.pdf)[[Project]](https://robot-moo.github.io/)
* VIMA: General Robot Manipulation with Multimodal Prompts [[Paper]](https://arxiv.org/abs/2210.03094)[[Project]](https://vimalabs.github.io/)[[Code]](https://github.com/vimalabs/VIMA)
* Diffusion-based Generation, Optimization, and Planning in 3D Scenes [[Paper]](https://arxiv.org/pdf/2301.06015.pdf)[[Project]](https://scenediffuser.github.io/)[[Code]](https://github.com/scenediffuser/Scene-Diffuser)
* LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery [[Paper]](http://arxiv.org/abs/2311.02058) [[Project]](https://ut-austin-rpl.github.io/Lotus/)
* Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World [[Paper]](https://arxiv.org/abs/2312.02976) [[Project]](https://spoc-robot.github.io/)
* ThinkBot: Embodied Instruction Following with Thought Chain Reasoning [[Paper]](https://arxiv.org/abs/2312.07062) [[Project]](https://guanxinglu.github.io/thinkbot/)
* CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory [[Paper]](https://arxiv.org/abs/2210.05663) [[Project]](https://mahis.life/clip-fields) [[Code]](https://github.com/notmahi/clip-fields)
* USA-Net: Unified Semantic and Affordance Representations for Robot Memory [[Paper]](https://arxiv.org/abs/2304.12164) [[Project]](https://usa.bolte.cc/) [[Code]](https://github.com/codekansas/usa)

## Relevant to Robotics (Perception)

### Open-Vocabulary Object Detection and 3D Classification
* Simple Open-Vocabulary Object Detection with Vision Transformers [[Paper]](https://arxiv.org/pdf/2205.06230.pdf)[[Code]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
*  Grounded Language-Image Pre-training [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.pdf)[[Code]](https://github.com/microsoft/GLIP)
*  Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection [[Paper]](https://arxiv.org/abs/2303.05499)[[Code]](https://github.com/IDEA-Research/GroundingDINO)
*  PointCLIP: Point Cloud Understanding by CLIP [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.pdf)[[Code]](https://github.com/ZrrSkywalker/PointCLIP)
*  Point-bert: Pre-training 3d point cloud transformers with masked point modeling [[Paper]](https://arxiv.org/abs/2111.14819)[[Code]](https://github.com/lulutang0608/Point-BERT)
*  ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding [[Paper]](https://arxiv.org/abs/2212.05171)[[Project]](https://tycho-xue.github.io/ULIP/)[[Code]](https://github.com/salesforce/ULIP)
*  Ulip-2: Towards scalable multimodal pre-training for 3d understanding [[Paper]](https://arxiv.org/pdf/2305.08275.pdf)[[Code]](https://github.com/salesforce/ULIP)
*  3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment [[Paper]](https://arxiv.org/pdf/2308.04352.pdf)[[Project]](https://3d-vista.github.io/)[[Code]](https://github.com/3d-vista/3D-VisTA)

### Open-Vocabulary Semantic Segmentation
* Language-driven Semantic Segmentation [[Paper]](https://arxiv.org/abs/2201.03546)[[Code]](https://github.com/isl-org/lang-seg)
* Emerging Properties in Self-Supervised Vision Transformers [[Paper]](https://arxiv.org/abs/2104.14294)[[Code]](https://github.com/facebookresearch/dino)
* Segment Anything [[Paper]](https://arxiv.org/abs/2304.02643)[[Project]](https://segment-anything.com/)
* Fast segment anything [[Paper]](https://arxiv.org/abs/2306.12156)[[Code]](https://github.com/CASIA-IVA-Lab/FastSAM)
* Faster Segment Anything: Towards Lightweight SAM for Mobile Applications [[Paper]](https://arxiv.org/abs/2306.14289)[[Code]](https://github.com/ChaoningZhang/MobileSAM)
* Track anything: Segment anything meets videos [[Paper]](https://arxiv.org/abs/2304.11968)[[Code]](https://github.com/gaomingqi/Track-Anything)

### Open-Vocabulary 3D Scene Representations
* Open-vocabulary Queryable Scene Representations for Real World Planning (NLMap) [[Paper]](https://arxiv.org/pdf/2209.09874.pdf)[[Project]](https://nlmap-saycan.github.io/)
* Clip-NeRF: Text-and-image driven manipulation of neural radiance fields [[Paper]](https://arxiv.org/abs/2112.05139)[[Project]](https://cassiepython.github.io/clipnerf/)
* CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory [[Paper]](https://arxiv.org/abs/2210.05663) [[Project]](https://mahis.life/clip-fields) [[Code]](https://github.com/notmahi/clip-fields)
* LERF: Language Embedded Radiance Fields [[Paper]](https://arxiv.org/abs/2303.09553)[[Project]](https://www.lerf.io/)[[Code]](https://github.com/kerrj/lerf)
* Decomposing nerf for editing via feature field distillation [[Paper]](https://arxiv.org/abs/2205.15585)[[Project]](https://pfnet-research.github.io/distilled-feature-fields)

### Object Representations
* FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/abs/2312.08344)[[Project]](https://nvlabs.github.io/FoundationPose/)
* BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects [[Paper]](https://arxiv.org/abs/2303.14158)[[Project]](https://bundlesdf.github.io//)
* Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation [[Paper]](https://arxiv.org/abs/2112.05124)[[Project]](https://yilundu.github.io/ndf/)
* Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation [[Paper]](https://arxiv.org/abs/2308.07931)[[Project]](https://f3rm.github.io/)
* You Only Look at One: Category-Level Object Representations for Pose Estimation From a Single Example [[Paper]](https://arxiv.org/abs/2305.12626)
* Zero-Shot Category-Level Object Pose Estimation [[Paper]](https://arxiv.org/abs/2204.03635)[[Code]](https://github.com/applied-ai-lab/zero-shot-pose)
* VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors [[Paper]](https://arxiv.org/abs/2210.11339)[[Project]](https://ut-austin-rpl.github.io/VIOLA/)[[Code]](https://github.com/UT-Austin-RPL/VIOLA)
* Learning Generalizable Manipulation Policies with Object-Centric 3D Representations [[Paper]](http://arxiv.org/abs/2310.14386)[[Project]](https://ut-austin-rpl.github.io/GROOT/)[[Code]](https://github.com/UT-Austin-RPL/GROOT)

### Affordance Information
* Affordance Diffusion: Synthesizing Hand-Object Interactions [[Paper]](https://arxiv.org/abs/2303.12538)[[Project]](https://judyye.github.io/affordiffusion-www/)
* Affordances from Human Videos as a Versatile Representation for Robotics [[Paper]](https://arxiv.org/abs/2304.08488)[[Project]](https://robo-affordances.github.io/)

### Predictive Models
* Adversarial Inverse Reinforcement Learning With Self-Attention Dynamics Model [[Paper]](https://ieeexplore.ieee.org/document/9361118)
* Connected Autonomous Vehicle Motion Planning with Video Predictions from Smart, Self-Supervised Infrastructure [[Paper]](https://arxiv.org/pdf/2309.07504.pdf)
* Self-Supervised Traffic Advisors: Distributed, Multi-view Traffic Prediction for Smart Cities [[Paper]](https://arxiv.org/abs/2204.06171)
* Planning with diffusion for flexible behavior synthesis [[Paper]](https://arxiv.org/abs/2205.09991)
* Phenaki: Variable-length video generation from open domain textual description [[Paper]](https://arxiv.org/abs/2210.02399)
* Robonet: Large-scale multi-robot learning [[Paper]](https://arxiv.org/abs/1910.11215)
* GAIA-1: A Generative World Model for Autonomous Driving [[Paper]](https://arxiv.org/abs/2309.17080)
* Learning universal policies via text-guided video generation [[Paper]](https://arxiv.org/abs/2302.00111)
* Video language planning [[Paper]](https://arxiv.org/abs/2310.10625)
* MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control [[Paper]](https://arxiv.org/pdf/2403.12037.pdf)[[Project]](https://sites.google.com/view/minedreamer/main)[[Code]](https://github.com/Zhoues/MineDreamer)

## Relevant to Robotics (Embodied AI)
* Inner Monologue: Embodied Reasoning through Planning with Language Models [[Paper]](https://arxiv.org/abs/2207.05608)[[Project]](https://innermonologue.github.io/)
* Statler: State-Maintaining Language Models for Embodied Reasoning [[Paper]](https://arxiv.org/abs/2306.17840)[[Project]](https://statler-lm.github.io/)
* EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought [[Paper]](https://arxiv.org/pdf/2305.15021.pdf)[[Project]](https://embodiedgpt.github.io/)
* MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge [[Paper]](https://openreview.net/forum?id=rc8o_j8I8PX)[[Code]](https://github.com/MineDojo/MineDojo)
* Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos [[Paper]](https://arxiv.org/abs/2206.11795)
* Open-World Multi-Task Control Through Goal-Aware Representation Learning and Adaptive Horizon Prediction [[Paper]](https://arxiv.org/pdf/2301.10034.pdf)[[Code]](https://github.com/CraftJarvis/MC-Controller)
* Describe, explain, plan and select: interactive planning with LLMs enables open-world multi-task agents [[Paper]](https://arxiv.org/pdf/2302.01560.pdf)[[Code]](https://github.com/CraftJarvis/MC-Planner)
* Voyager: An Open-Ended Embodied Agent with Large Language Models [[Paper]](https://arxiv.org/abs/2305.16291)[[Project]](https://voyager.minedojo.org/)[[Code]](https://github.com/MineDojo/Voyager)
* Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory [[Paper]](https://arxiv.org/abs/2305.17144)[[Project]](https://github.com/OpenGVLab/GITM)
* Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents [[Paper]](https://arxiv.org/pdf/2201.07207.pdf)[[Project]](https://wenlong.page/language-planner/)[[Code]](https://github.com/huangwl18/language-planner)
* GROOT: Learning to Follow Instructions by Watching Gameplay Videos [[Paper]](https://arxiv.org/pdf/2310.08235.pdf)[[Project]](https://craftjarvis.github.io/GROOT/)[[Code]](https://github.com/CraftJarvis/GROOT)
* JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models [[Paper]](https://arxiv.org/pdf/2311.05997.pdf)[[Project]](https://craftjarvis.github.io/JARVIS-1/)[[Code]](https://github.com/CraftJarvis/JARVIS-1)
* SQA3D: Situated Question Answering in 3D Scenes [[Paper]](https://arxiv.org/pdf/2210.07474.pdf)[[Project]](https://sqa3d.github.io/)[[Code]](https://github.com/SilongYong/SQA3D)
* MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception [[Paper]](https://arxiv.org/pdf/2312.07472.pdf)[[Project]](https://iranqin.github.io/MP5.github.io/)[[Code]](https://github.com/IranQin/MP5)
* MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control [[Paper]](https://arxiv.org/pdf/2403.12037.pdf)[[Project]](https://sites.google.com/view/minedreamer/main)[[Code]](https://github.com/Zhoues/MineDreamer)

### Generalist AI
* Generative Agents: Interactive Simulacra of Human Behavior [[Paper]](https://arxiv.org/abs/2304.03442)
* Towards Generalist Robots: A Promising Paradigm via Generative Simulation [[Paper]](https://arxiv.org/abs/2305.10455)
* A generalist agent [[Paper]](https://arxiv.org/abs/2205.06175)
* An Embodied Generalist Agent in 3D World [[Paper]](https://arxiv.org/pdf/2311.12871.pdf)[[Project]](https://embodied-generalist.github.io/)[[Code]](https://github.com/embodied-generalist/embodied-generalist)

### Simulators
* Gibson Env: real-world perception for embodied agents [[Paper]](https://arxiv.org/abs/1808.10654)
* iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks [[Paper]](https://arxiv.org/abs/2108.03272)[[Project]](https://svl.stanford.edu/igibson/)
* BEHAVIOR-1k: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation [[Paper]](https://openreview.net/forum?id=_8DoIe8G3t)[[Project]](https://behavior.stanford.edu/behavior-1k)
* Habitat: A Platform for Embodied AI Research [[Paper]](https://arxiv.org/abs/1904.01201)[[Project]](https://aihabitat.org/)
* Habitat 2.0: Training home assistants to rearrange their habitat [[Paper]](https://arxiv.org/abs/2106.14405)
* Robothor: An open simulation-to-real embodied ai platform [[Paper]](https://arxiv.org/abs/2004.06799)[[Project]](https://ai2thor.allenai.org/robothor/)
* VirtualHome: Simulating Household Activities via Programs [[Paper]](https://arxiv.org/abs/1806.07011)
* ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes [[Paper]](https://arxiv.org/pdf/2304.04321.pdf)[[Project]](https://arnold-benchmark.github.io/)[[Code]](https://github.com/arnold-benchmark/arnold)
* ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks [[Paper]](https://arxiv.org/abs/1912.01734)[[Project]](https://askforalfred.com/)[[Code]](https://github.com/askforalfred/alfred)
* LIBERO: Benchmarking Knowledge Transfer in Lifelong Robot Learning [[Paper]](https://arxiv.org/pdf/2306.03310.pdf)[[Project]](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/overview.html)[[Code]](https://github.com/Lifelong-Robot-Learning/LIBERO)
* ProcTHOR: Large-Scale Embodied AI Using Procedural Generation [[Paper]](https://arxiv.org/abs/2206.06994)[[Project]](https://procthor.allenai.org/)[[Code]](https://github.com/allenai/procthor)
