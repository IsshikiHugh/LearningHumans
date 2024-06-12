# Prepare SKEL Model

SKEL provide three repositories, and we only need one of them. You can check the official instruction [here](https://github.com/MarilynKeller/SKEL?tab=readme-ov-file#installation).

```shell
# Preparation.
cd third_party  # just make sure you are in the right directory
conda create --name lh-skel --clone lh  # clone the original `lh` env to prevent conflict
conda activate lh-skel

# Start setting up.
git clone https://github.com/MarilynKeller/SKEL
pip install git+https://github.com/mattloper/chumpy
pip install -e .
```

Then, download necessary files from [project page](https://skel.is.tue.mpg.de/), click `Download/SKEL and BSM models` to download the results. You can put the results where ever you like, you can just put them under `data_inputs/body_models` folder. And then, copy the path of the `skel_models_v1.0` folder, and fill it into the `skel_folder = <skel-folder-path>` in `SKEL/skel/config.py`.