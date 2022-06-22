# Comp472_Project

Our goal is to train a CNN with the ability to classify different types of masks in images.

## The Team (NS_04)
- Maxime Johnson (Compliance Specialist)
- Visweshwaran Balasubramanian (Training Specialist)
- Matthew Greco (Evaluation Specialist)
- Hayden Ross (Data Specialist)

## File Descriptions:
- <b>MaskDetection.py:</b> Final combined python code responsible for handling the preprocessing, training, and evaluation of our model. Saves the model to a file.
- <b>MaskModelEvaluation.py:</b> Loads and evaluates the performance of the loaded model.
- <b>Dataset:</b> All of our images organised by class into sub folders.
- <b>Project_Report:</b> Report of project.
- <b>runMe.bat:</b> Run to install all dependencies within `Dependencies.txt`.
- <b>Dependencies.txt:</b> List of all dependencies.

## Dependencies
- Tensorflow
- PyTorch
- TorchVision
- sklearn
- itertools
- matplotlib
- numpy
- splitfolders


## Execution
1. Install all dependencies using `runMe.bat` or independently.
2. Download the trained model from [Google Drive](https://drive.google.com/file/d/1L27D-IVS6tbxKTNns-PcQKbUYX6L7JH7/view)
3. Download `MaskDetection.py`, `MaskModelEvaluation.py`, `Dataset`.
4. Make sure that the `Dataset` is in the execution directory of `MaskDetection.py` and `MaskModelEvaluation.py`.
5. Open `MaskDetection.py` or `MaskModelEvaluation.py` with Jupyter Notebook and run all cells.
