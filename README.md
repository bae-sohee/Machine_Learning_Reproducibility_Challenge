
# Machine Learning Reproducibility Challenge (MLRC) of PT4AL 

## ▶ Project Objective
  - Review the reproducibility of previously published papers.
  - We experimented by adding a dataset that was not in the official code of the selected paper and adding the code required for the pretext task.

## ▶ Selected Papers and Official codes
  -  [PT4AL: Using Self-Supervised Pretext Tasks for Active Learning (ECCV2022)](https://arxiv.org/abs/2201.07459)
  - [Official Pytorch Implementation](https://github.com/johnsk95/PT4AL)

## ▶ Team Member

<table style="border: 0.5px solid gray">
 <tr>
    <td align="center"><a href="https://github.com/Moominci"><img src="https://avatars.githubusercontent.com/u/68638190?v=4" width="130px;" alt=""></td>
    <td align="center"><a href="https://github.com/bae-sohee"><img src="https://avatars.githubusercontent.com/u/123538321?v=4" width="130px;" alt=""></td>
    <td align="center" style="border-right : 0.5px solid gray"><a href="���� �ڵ�"><img src="https://avatars.githubusercontent.com/u/118954283?v=4" width="130px;" alt=""></td>

  </tr>
  <tr>
    <td align="center"><a href="https://github.com/Moominci"><b>MinSeok Yoon</b></td>
    <td align="center"><a href="https://github.com/bae-sohee"><b>Sohee Bae</b></td>
    <td align="center" style="border-right : 0.5px solid gray"><a href="https://github.com/Sunni-yoon"><b>Seonyoung Yoon</b></td>
  </tr>
</table>
<br/>

## Overview
  ![PT4AL](https://github.com/bae-sohee/Machine_Learning_Reproducibility_Challenge/assets/123538321/2e804b12-846f-4424-8fca-7e7599092c1e)

  - This paper proposes an Active learning method using Self-Supervised Pretext Tasks.
  - First, they learn the representation of unlabeled data using Pretext Task, which initializes the main task model.
  - Since then, they have proposed an Active learning method that utilizes the loss of Pretext Task to select difficult and representative data.

## Environmnet Setting

- Required Computer Specifications 
  | GPU | CUDA | CUDNN |  
  | --- | --- | --- |
  | GTX-2080TI \* 8 | 11.4 | 8.2 |

- Python version is 3.9.
- Installing all the requirements may take some time. After installation, you can run the codes.
- Please notice that we used 'PyTorch' and device type as 'GPU'.
- We utilized 8 GPUs in our implementation. If the number of GPUs differs, please adjust the code accordingly based on the specific situation.
- [```requirements.txt```] file is required to set up the virtual environment for running the program. This file contains a list of all the libraries needed to run your program and their versions.

    #### In **Anaconda** Environment,

    ```
    $ conda create -n [your virtual environment name] python=3.9
    
    $ conda activate [your virtual environment name]
    
    $ pip install -r requirements.txt
    ```

- Create your own virtual environment.
- Activate your Anaconda virtual environment where you want to install the package. If your virtual environment is named 'testasal', you can type **conda activate test**.
- Use the command **pip install -r requirements.txt** to install libraries.

## Prerequisites
To generate train and test dataset:
```
python make_data.py
```
- Create the Cifar10, Imbalanced_Cifar10, Caltech101 folder required for the experiment

To train the pretext task (rotation, colorization) on the unlabeled set:
```
python rotation.py --dataset Cifar10
python rotation.py --dataset Imbalanced_Cifar10
python colorization.py --dataset Caltech101
```
- The dataset and pretext task can be set as an argument, the available datasets include *Cifar10, Imbalanced_Cifar10, Caltech101*, and the available pretext tasks include *rotation, colorization*.

To extract pretext task losses and create batches:
  ```
  python make_batches.py --task rotation --dataset Caltech101
  ```
- The dataset and the prext task are set as arguments to extract the loss of the prext task and divide the batches required for learning.
- The available datasets are *Cifar10, Imbalanced_Cifar10, Caltech101*, and the available pretext tasks are *rotation, colorization*.
- Sort the loss of the pretext task of each data in ascending order, then divide it into 10 batches and save it as a text file.
- `loss_{dataset}_{task}` folder : You can check each batch.
- `{task}_loss_{dataset}.txt`: You can check the loss for the task of each dataset.

  ```bash
  ├── loss_{dataset}_{task}
  │   ├── batch_0.txt
  │   │    ...
  │   ├── batch_9.txt
  ```

- Each text file contains, in sequential order, the paths of the corresponding data. An excerpt from an example text file is provided below(ex.loss_Cifar10_rotation/batch_0.txt):

  ```bash
  ./Cifar10/DATA/train/5/38343.png
  ./Cifar10/DATA/train/0/29348.png
  ./Cifar10/DATA/train/5/22390.png
  ...
  ```

## Running the Code

To evaluate on PT4AL task:
```
python main.py --dataset Cifar10 --task rotation
```
- Experiments on PT4AL proposed by the paper.
- Configure the initial sample with the batch of pretext task loss built in advance, and then conduct active learning by sampling data from the loss batch in subsequent cycles.
- The available datasets are *Cifar10, Imbalanced_Cifar10, Caltech101*, and the available pretext tasks are *rotation, colorization*.
- Results for the experiment are found in **main_best_{dataset}_{task}.txt**.

To evaluate on random sampling active learning task:
```
python main_random.py --dataset Cifar10
```
- Experiment on random sampling for comparison with the proposed method.
- In the case of initial samples, the data sampled at the time of the cycle is also randomly extracted and active learning is carried out.
- The available datasets are *Cifar10, Imbalanced_Cifar10, Caltech101*.
- Results for the experiment are found in **main_best_{dataset}_random.txt**.

## Correlation Plot

To extract a classification loss with superivsed learning:
```
python main_supervised.py --dataset Cifar10
```
- In order to verify the assumption of this paper, **'H1: Pretext task loss is correlated with the main task loss'**, we extract the loss of the downstream task.
- The available datasets are *Cifar10, Imbalanced_Cifar10, Caltech101*.

To determine the correlation between main task loss and pretext task loss:
```
python correlation_plot.py --dataset Cifar10 --task rotation
```
- It is possible to check the correlation plot between classification loss, which is the main task, and each pretext task loss. (Correlation plot for the entire dataset, and correlation plot sampled 1000 by random)
- The available datasets are *Cifar10, Imbalanced_Cifar10, Caltech101*, and the available pretext tasks are *rotation, colorization*.

## Result
### Active learning result
- 이미지
- 설명
### Correlation result
- 이미지
- 설명

## Citation
```
@inproceedings{yi2022using,
  title = {Using Self-Supervised Pretext Tasks for Active Learning},
  author = {Yi, John Seon Keun and Seo, Minseok and Park, Jongchan and Choi, Dong-Geol},
  booktitle = {Proc. ECCV},
  year = {2022},
}
```
