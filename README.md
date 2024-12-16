# LearningHumans

This repository aims to provide code snippets and examples for common virtual human utilities, for example, varies kinds of representation from skeletons to parameters models. You can regard this project as just a simple starting tutorial.

> [!TIP]
>
> 👋 Welcome! I hope this project will help you!
>
> 📖 As a newbie to this field, I can't ensure a comprehensive guidance here. But I think it's a good way to improve for the better through sharing what I've learned. I will try my best to improve this project. If you find **any** mistakes or have **any** suggestions, please feel free to open issues or PRs. Thanks!
>
> ✨ Problems you will meet in practice will be more complex. I think it's important to learn how to solve a specific problem without full set of guides. For instance, some researcher will use their own variation of SMPL model.

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
2. \(🛠️ *WIP*\) [Abstract Skeletons](notebooks/abstract_skeletons.ipynb)
3. [SMPL Basic](notebooks/SMPL_basic.ipynb)
4. [SMPLH Basic](notebooks/SMPLH_basic.ipynb)
5. [SMPLX Basic](notebooks/SMPLX_basic.ipynb)
6. \(🛠️ *WIP*\) [SMPL Details](notebooks/SMPL_details.ipynb)
7. SKEL Basic
8. T.B.C.

I also provide some code snippets for some common tasks:

1. [Parallel Tasks](notebooks/parallel.ipynb)
2. T.B.C.

---

## Todo List

- [ ] Abstract Skeletons (quite rough now)
- [ ] Parameter Models
    - [x] SMPL basic
    - [x] SMPL-H basic (compared to SMPL)
    - [x] SMPL-X basic (compared to SMPL)
    - [ ] SMPL details
    - [ ] SKEL basic
- [ ] Misc
    - [x] parallel tasks
    - [ ] useful visualization tools
        - [ ] `wis3d`
        - [ ] `viser`
        - [ ] `matplotlib`
        - [ ] `pyrender`
        - [ ] `pytorch3d` renderer
        - [ ] ...
    - [ ] useful data processing tools (path walking, file dumping, etc.)
        - [ ] `pathlib.Path`, ...
        - [ ] `shutil`, ...
        - [ ] `torch.save`, `np.save`, `np.savez`, `joblib.dump`...


## Acknowledgements

This project benefits a lot from the following resources:

- Motivations
    - [learning_research](https://github.com/pengsida/learning_research)
    - [learning_nerf](https://github.com/pengsida/learning_nerf)
    - [LearningMotion](https://github.com/phj128/LearningMotion)
- Contents Details
    - Check the references in the notebooks.