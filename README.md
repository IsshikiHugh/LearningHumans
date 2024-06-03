# LearningHumans

This repository aims to provide code snippets and examples for common human motion utilities, for example, varies kinds of motion expression from skeletons to parameters models. You might think of this project as a simple starting tutorial rather than an advanced tutorial.

> [!TIP]
>
> 👋 Welcome! I hope this project will help you!
>
> 📖 As a newbie to this field, I can't ensure a comprehensive guide here. But I think it's a good way to improve for the better through sharing what I've learned. I will try my best to improve this project. If you find any mistakes (anything) or have any suggestions, please feel free to open issues or PRs. Thanks!
>
> ✨ Things you will meet actually will be more complex then I present here. It's quite important for you to learn how to solve specific problems you meet. For instance, many researcher will use their own variation of SMPL model instead to meet their specific needs.

## Preparation

### Environment

```shell
conda create -n lh python=3.8
conda activate lh

pip install -r requirements.txt  # install the required packages
pip install -e .                 # install the local package
```

### Data

Some sections require extra data downloaded from the internet. They can be checkpoints of models or datasets or something else. The instructions of how to download them will be provided in notebooks in which you need those data.

## Usage

The supporting codes are placed in `lib` folder. They are mainly some utility functions for visualization or logging or data organization or something else. You can just ignore them.

All the snippets are in the jupyter notebooks under the `notebooks` folder.

Here is one recommended order to read the notebooks:

1. [Rotation Representation](notebooks/rotation_representation.ipynb)
2. \(🛠️ *WIP*\) [Skeletons](notebooks/skeletons.ipynb)
3. [SMPL Basic](notebooks/SMPL_basic.ipynb)
4. SMPLH Basic
5. [SMPLX Basic](notebooks/SMPLX_basic.ipynb)
6. \(🛠️ *WIP*\) [SMPL Details](notebooks/SMPL_details.ipynb)
7. Skel Basic
8. T.B.C.


## Todo List

### About the Main Content

- [ ] Skeletons (quite rough now)
- [ ] Parameters Models
    - [x] SMPL basic
    - [ ] SMPL-H basic (compared to SMPL)
    - [x] SMPL-X basic (compared to SMPL)
    - [ ] SMPL details
    - [ ] Skel basic
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