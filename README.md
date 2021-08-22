# Image Classifier

This is a command line application that predicts and classifies various images of flowers with an accuracy of 85.5%. It is built using the `PyTorch` library.

# Installation

Clone this github repository and install all python libraries fom the `requirements.txt`

Run the following code in your `Python` terminal
```
pip install -r requirements.txt

$   git clone https://github.com/github/Image_Classifier.git
$   cd ImageClassifier
```

# Usage

The `predict.py` requires a number of arguments for runtime. The following examples
demonstrates how to run it manually from the command line.
```
python train.py  --data_dir floweres --arch vgg13 --gpu true
python predict.py  --img_path floweres/test/1/img_076543.jpg --topk 5 --gpu true category_names cat_to_names.json
```
