# Pose Estimation Project (CV Final Project)

**Authors**: Eric Song, Yoohyuk Chang, Ryan McGovern, Jacob Choi  
**Course**: EN.601.461 Computer Vision Final Project

Follow these steps to set up your environment and get started.

---

## 🚀 Quick Start Guide

For this project, it’s recommended to use a virtual environment to manage dependencies. For macOS users, first install `virtualenv` using `pip3 install virtualenv`. Then, create a virtual environment by running `virtualenv env` in your terminal. Once created, activate it with `source env/bin/activate`. With the environment active, install the required packages by running `pip install -r requirements.txt`. When you are done working, you can deactivate the environment by using the `deactivate` command.

For Windows users, you can also use `virtualenv` to create a virtual environment. First, install it using `pip install virtualenv`. Then, create the environment by running `virtualenv env` in your command prompt or PowerShell. To activate the virtual environment, use `.\env\Scripts\activate`. Once activated, install the necessary packages using `pip install -r requirements.txt`. To deactivate, simply run `deactivate`.

Before running any models, it’s essential to set the correct root path. Make sure you are in the `final_project` directory (the root directory of this project). For macOS, set the root path by running `export PYTHONPATH=$(pwd)`. For Windows, set the path using `set PYTHONPATH=%cd%`. This will ensure that the models can locate the necessary files and run correctly.

If you are using macOS and encounter any SSL certificate issues, you can resolve this by running the command `/Applications/Python\ 3.11/Install\ Certificates.command` in your terminal. This step will update your system’s SSL certificates and help avoid connection issues.

Additionally, some annotations are required to run the models effectively. First, create a new directory named `annotations` inside the `scripts` directory. Then, download the COCO dataset annotations by visiting [COCO Dataset annotations page](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). After downloading, go to your downloads folder, manually extract the files, and place `person_keypoints_train2017.json` and `person_keypoints_val2017.json` in the `scripts/annotations` directory.

With everything set up, you’re ready to run the models. Navigate to the `run_models` directory and run the model you want by using `python {file_name}.py`, replacing `{file_name}` with the specific model’s filename you wish to execute. 
