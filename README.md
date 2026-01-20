# LearningHumans

This repository aims to provide code snippets and examples for common virtual human utilities, for example, various kinds of representation from skeletons to parametric models. You can regard this project as just a simple starting tutorial.

> [!TIP]
>
> 👋 Welcome! I hope this project will help you!
>
> 📖 If you find **any** mistakes or have **any** suggestions, please feel free to open issues or PRs. Thanks!
>
> ✨ Problems you will meet in practice will be more complex. I think it's important to learn how to solve a specific problem without a full set of guides. For instance, some researchers will use their own variation of SMPL model.

## Preparation

### Environment

```shell
conda create -n lh python=3.8
conda activate lh

pip install -r requirements.txt  # install the required packages
pip install -e .                 # install the local package
```

Then, you need to install `pytorch3d`:

- For macOS user, please use `pip install pytorch3d==0.7.4`.
- For Linux user, according to [this](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), if you know your environment version, I recommend you to [install wheels directly](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-wheels-for-linux), otherwise, [installing from GitHub](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#1-install-from-github) would be easier (but slower).

### Data

Some sections require additional data to be downloaded (such as model checkpoints, datasets, or other resources). The instructions on how to download them will be provided in the specific notebooks where you need those data.

## Usage

The supporting codes are provided in the `lib` folder and the [`ez4d`](https://github.com/IsshikiHugh/ez4d) library. Codes in `lib` are mainly for notebook supports, and things in `ez4d` should be helpful for general human motion research.

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
    - [x] SKEL basic
- [ ] Misc
    - [x] parallel tasks

## Acknowledgements

This project greatly benefits from the following resources:

- Motivations
    - [learning_research](https://github.com/pengsida/learning_research)
    - [LearningMotion](https://github.com/phj128/LearningMotion)
- Contents Details
    - Check the references in the notebooks.