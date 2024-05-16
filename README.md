# LearningHuman

This repository aims to provide code snippets and examples for common human motion utilities, for example, varies kinds of motion expression from skeletons to parameters models.

> [!TIP] Preface
>
> 👋 Welcome! As a newbie to this field, I'm not sure I can provide a comprehensive guide for this field.
>
> 📖 But I think it's a good way to learn better by writing down what I've learned and I will try my best to make this project wonderful. So, if you find any mistakes (including typos) or have any suggestions, please feel free to open an issue or PR. Thanks!
>
> ✨ Things you will meet actually will be more complex then I present here. It's quite important for you to learn how to solve the specific problem you meet. For example, many researcher will use their own version of SMPL-like model instead to meet their specific needs.

## Preparation

### Environment

```shell
conda create -n lh python=3.8
conda activate lh

pip install -r requirements.txt  # install the required packages
pip install -e .                 # install the local package
```

### Data

Some notebooks may need some extra data downloaded from the internet. They can be checkpoints of models or datasets or something else. The instruction of how to download them will be provided in notebooks that need them.

## Usage

The support codes are placed in `lib` folder. They are mainly some utility functions for visualization or logging or data organization or something else. You can just ignore them if you are not interested in them.

All the snippets are in the jupyter notebooks under the `notebooks` folder.

Here is the recommended order to read the notebooks if you are totally new to this field:

1. [Rotation Representation](notebooks/rotation_representation.ipynb)
2. \(🛠️ *WIP*\) [Skeletons](notebooks/skeletons.ipynb)
3. [SMPL Basic](notebooks/SMPL_basic.ipynb)
4. SMPLH Basic
5. [SMPLX Basic](notebooks/SMPLX_basic.ipynb)
6. SMPL Details
7. T.B.C.


## Todo List

### About the Main Content

- [ ] Skeletons (quite rough now)
- [ ] Parameters Models
    - [x] SMPL basic
    - [ ] SMPL-H basic (compared to SMPL)
    - [x] SMPL-X basic (compared to SMPL)
    - [ ] SMPL details
- [ ] Msic
    - [ ] useful visualization tools
        - [ ] Wis3D
        - [ ] matplotlib
        - ...
    - [ ] useful data processing tools (path walking, file dumping, etc.)
        - [ ] pathlib.Path, ...
        - [ ] shutil, ...
        - [ ] torch.save, np.save, np.savez, joblib.dump...

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