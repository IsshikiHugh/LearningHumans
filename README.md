# LearningHuman

This repository aims to provide code snippets and examples for common human motion utilities, for example, varies kinds of motion expression from skeletons to parameters models.

> [!TIP]
>
> 👋 Welcome! As a newbie to this field, I'm not sure I can provide a comprehensive guide for this field. 📖 But I think it's a good way to learn better by writing down what I've learned and I will try my best to make this project wonderful. So, if you find any mistakes (including typos) or have any suggestions, please feel free to open an issue or PR. Thanks!

## Preparation

### Environment

```shell
conda create -n lh python=3.8
conda activate lh

pip install -r requirements.txt
pip install -e .
```

### Data

Some notebooks may need some extra data downloaded from the internet. They can be checkpoints of models or datasets. The instruction of how to download them will be provided in notebooks that need them.

## Usage

The support codes are placed in `lib` folder. They are mainly some utility functions for visualization or logging or data organization or something else. You can just ignore them if you are not interested in them.

All the snippets are in the jupyter notebooks under the `notebooks` folder.

Here is the recommended order to read the notebooks if you are totally new to this field:

1. [Rotation Representation](notebooks/rotation_representation.ipynb)
2. \(🛠️ *WIP*\) [Skeletons](notebooks/skeletons.ipynb)
3. [SMPL Basic](notebooks/SMPL_basic.ipynb)
4. SMPLH and SMPLX
5. SMPL Details
6. T.B.C.


## Todo List / Help Wanted

### About the Main Content

- [ ] Skeletons (rough now)
- [ ] Parameters Models
    - [x] SMPL basic
    - [ ] SMPL-H basic
    - [ ] SMPL-X basic
    - [ ] SMPL details
- [ ] Common Metrics
    - {PA, WA, W2A, normal} x {MPJPE, MPVE}
    - Accel, AccelErr, Jitter
    - FootSliding
    - ...

### Something Less Important

- [ ] Colab Support
- [ ] Chinese Translation
- [ ] Better Display

## Acknowledgements

This project benefits a lot from the following resources:

- Motivations
    - [learning_research](https://github.com/pengsida/learning_research)
    - [learning_nerf](https://github.com/pengsida/learning_nerf)
    - [LearningMotion](https://github.com/phj128/LearningMotion)
- Contents Details
    - [ ] TODO