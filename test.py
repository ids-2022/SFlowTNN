#使用之前训练好的模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from text_cnn import MyTextCnn
from datetime import datetime
import sys


def predict(input_path, model_path):
    input_path=input_path
    model_path=model_path
    test_data_dir1 = input_path
    data_transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]
    )
    raw_data = ImageFolder(test_data_dir1, transform=data_transform)
    test_loader = Data.DataLoader(raw_data, batch_size=20, shuffle=False,num_workers=4)
    mymodel=torch.load(model_path)
    classlabel=raw_data.classes
    # time1 = datetime.now()
    final_prelab=list()
    final_trulab=list()
    for step, (b_x, b_y) in enumerate(test_loader):
        mymodel.eval()
        output = mymodel(b_x)
        prelab = torch.argmax(output, 1)
        final_prelab.extend(prelab)
        final_trulab.extend(b_y)
    return final_trulab,final_prelab

if __name__ == '__main__':
    input_path=sys.argv[1]
    model_path=sys.argv[2]
    predict(input_path, model_path)









