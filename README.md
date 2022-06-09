# Comp472_Project

Our goal is to train a CNN with the ability to classify different types of masks in images.

## The Team (NS_04)
- Maxime Johnson (Compliance Specialist)
- Visweshwaran Balasubramanian (Training Specialist)
- Matthew Greco (Evaluation Specialist)
- Hayden Ross (Data Specialist)

## File Descriptions:
- <b>MaskDetection.ipynb:</b> Final combined jupyter notebook responsible for handling the preprocessing, training, and evaluation of our model. Saves the model to be imported.
- <b>MaskModelEvaluation.ipynb:</b> Loads and evaluates the preformance of the loaded model.
- <b>Dataset:</b> All of our images organised by class into subfolders.
- <b>TrainedModel_V1:</b> Our best trained model.
- <b>Project_Report:</b> Report of project.
- <b>runMe.bat:</b> Installs all dependencies.

## Dependencies
- Tensorflow
- PyTorch
- TorchVision
- sklearn
- itertools
- matplotlib
- numpy


## Execution
1. Run `runMe.bat` to install all dependencies.
2. Download `MaskDetection.ipynb`, `MaskModelEvaluation.ipynb`, `Dataset`.
3. Make sure that the `Dataset` is in the execution directory of `MaskDetection.ipynb` and `MaskModelEvaluation.ipynb`.
4. Run `MaskDetection.ipynb` or `MaskModelEvaluation.ipynb`.
