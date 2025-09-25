<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This project uses the [Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) from Kaggle to train
and evaluate a Convolutional Neural Network (CNN). The dataset contains images of cats and dogs, suitable for binary
classification tasks and deep learning experiments. This repository contains a Streamlit application for training,
testing, and visualizing a Convolutional Neural Network (CNN) for binary image classification (Cats vs Dogs). It
supports a full workflow from data preprocessing to model evaluation and single image prediction.

**DATA DESCRIPTION**
---

- **Training Set (train)**  
  Contains two subsets: cats and dogs. After removing duplicates, each subset has 4,000 images.  
  Filenames are like `cat.0.jpg` and `dog.1.jpg`, and labels can be inferred directly from filenames.

- **Test Set (test)**  
  Contains two subsets: cats and dogs. After removing duplicates, each subset has 1,000 images.  
  Filenames follow the same format as the training set and can be used to infer labels.

**FEATURES**
---

- **Data Preparation & Augmentation**: Preprocess images using rotation, width/height shift, shear, zoom, and horizontal
  flip.
- **Batch Loading**: Load training and testing images in batches, navigate and inspect individual samples.
- **CNN Training**: Simple yet effective CNN architecture with multiple Conv2D, MaxPooling2D, Flatten, and Dense layers.
- **Real-time Metrics Logging**: Custom Keras callbacks update Streamlit placeholders for live metrics (loss, accuracy)
  during training.
- **Model Testing & Evaluation**: Evaluate model with accuracy, precision, recall, AUC, and F1 score.
- **Single Image Prediction**: Select a test image and predict its label instantly.
- **Save & Load Model**: Save trained models and reload them for inference.
- **Streamlit Integration**: Interactive sidebar settings, sliders, and buttons for smooth workflow.

**QUICK START**
---

1. Clone the repository to your local machine.
2. Install the required dependencies with the command `pip install -r requirements.txt`.
3. Run the application with the command `streamlit run main.py`.
4. You can also try the application by visiting the following
   link:  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://cnn-cat-n-dog.streamlit.app/)

**LARGE FILE STORAGE (LFS)**
---
This project uses Git Large File Storage (LFS) to manage large files, such as datasets, models, and binary files. The
instructions as follows are only used to upload the large file to the remote repository.

1. Install Git LFS with the command `brew install git-lfs`.
2. Initialise Git LFS in the repository with the command `git lfs install`. **ONLY ONCE**.
3. Track the large files with the command `git lfs track "*.jpg"` (you can replace `*.jpg` with the appropriate file
   extension).
4. Add the `.gitattributes` file to version control with the command `git add .gitattributes` or using the UI interface.
5. Add the `data/` file to version control with the command `git add data/` or using the UI interface.
6. Commit the changes with the command `git commit -m "Track large files with Git LFS"` or using the UI interface.
7. Use the command `git lfs ls-files` to list all files being tracked by Git LFS.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.
9. If you change the name of remote while initialising the repository, you need to change the `origin` to your remote
   name, such as `GitHub` or `xxx`, in the command `git push -u GitHub main` or `git push -u GitHub main`. Besides, if
   you change the branch name, you also need to change the `main` to your branch name, such as `master` or `xxx`, in the
   command `git push -u GitHub master` or `git push -u xxx master`. Therefore, it is better to keep the default names of
   remote and branch.
10. If you fail to push the large files, you might have used 2FA authentication. The normal push of the button of the
    UI interface is invalid. You can try to use a **personal access token (PAT)** instead of accessing the GitHub
    repository. If you have had the token, run the command `git push origin main` first. Then, enter the `username` and
    the `token` as the password.
11. When you push with `username` and `token` successfully first, you can continue to use the button of the UI interface
    to push the changes.
12. If you use `username` and `password` to initialise the repository, and you use the `personal access token (PTA)` to
    push the large files, you might fail to push the future changes with the `push` button of the UI. In this case, you
    can close the LFS push function by running the following command:
    ```bash
    git config lfs.<remote-url>/info/lfs.locksverify
    ```
    Then, you can use the `push` button of the UI to push the changes.
13. You must **install Git LFS locally** before you clone the repository if you plan to get the
    **full size of the data**. Otherwise, you will only get the pointer files. you can run the following command to
    install Git LFS:
    ```bash
    git lfs uninstall
    ```
14. (Optional) If you have already cloned the repository without Git LFS installed, you can run the following command to
    fetch the actual large files:
    ```bash
    git lfs pull
    ```

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.
3. Run the command `streamlit run app.py` to start the web application.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause Licence](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.