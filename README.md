# LearningHumans

This repository aims to provide code snippets and examples for common human motion utilities, for example, varies kinds of motion expression from skeletons to parameters models. You might think of this project as a simple starting tutorial rather than an advanced tutorial. And it's also my notebook.

> [!TIP]
>
> 👋 Welcome! I hope this project will help you!
>
> 📖 As a newbie to this field, I can't ensure a comprehensive guidance here. But I think it's a good way to improve for the better through sharing what I've learned. I will try my best to improve this project. If you find any mistakes (anything) or have any suggestions, please feel free to open issues or PRs. Thanks!
>
> ✨ Problems you will actually meet will be more complex then I present here. It's quite important for you to learn how to solve the specific problems. For instance, many researcher will use their own variation of SMPL model instead, as they may have specific needs.

## Preparation

### Environment

```shell
conda create -n lh python=3.8
conda activate lh

pip install -r requirements.txt  # install the required packages
pip install -e .                 # install the local package
```

### Data

Some sections require extra data to be downloaded. (They can be checkpoints of models or datasets or something else.) The instructions about how to download them will be provided in specific notebooks in which you need those data.

## Usage

The supporting codes are placed in `lib` folder. They are mainly some utilities for visualization or logging or data organization or something else. You can just ignore them.

All the snippets are in the jupyter notebooks under the `notebooks` folder.

Here is one recommended order to read the notebooks:

1. [Rotation Representation](notebooks/rotation_representation.ipynb)
2. \(🛠️ *WIP*\) [Skeletons](notebooks/skeletons.ipynb)
3. [SMPL Basic](notebooks/SMPL_basic.ipynb)
4. [SMPLH Basic](notebooks/SMPLH_basic.ipynb)
5. [SMPLX Basic](notebooks/SMPLX_basic.ipynb)
6. \(🛠️ *WIP*\) [SMPL Details](notebooks/SMPL_details.ipynb)
7. \(🛠️ *WIP*\) [SKEL Basic](notebooks/SKEL_basic.ipynb)
8. T.B.C.

I also provide some code snippets of some common tasks:

1. [Parallel Tasks](notebooks/parallel.ipynb)
2. T.B.C.

---

## Todo List

### About the Main Content

- [ ] Skeletons (quite rough now)
- [ ] Parameters Models
    - [x] SMPL basic
    - [x] SMPL-H basic (compared to SMPL)
    - [x] SMPL-X basic (compared to SMPL)
    - [ ] SMPL details
    - [ ] SKEL basic
- [ ] Msic
    - [x] parallel tasks
    - [ ] useful visualization tools
        - [ ] `Wis3D`
        - [ ] `matplotlib`
        - ...
    - [ ] useful data processing tools (path walking, file dumping, etc.)
        - [ ] `pathlib.Path`, ...
        - [ ] `shutil`, ...
        - [ ] `torch.save`, `np.save`, `np.savez`, `joblib.dump`...

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
    - Check the references in the notebooks.