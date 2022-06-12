# SFlow for traffic detection
We propose a Multi-Chaneel Convolutional Neural Network approach for encryption intrusion traffic detection and malware traffic detection in order to bread the limitation of traffic online real-time detction and end devices with constraints  of computational resource. To achieve this, evaluate SFlow on open dataset ISCXVPN2016, ISCXTor2016, Kitsune2018 and CTU-13.

## Directory Structure
The repository provides .pkl files, test sets and test code, which can be used for reproducing the results of experiments in intrusion detection, malware traffic dection and multi-label classification for ecnrypted traffic. The following table presents the directory structure.

<table>
    <tr>
        <td>Directory</td>
        <td>Categories</td>
        <td>Description</td>
    </tr>
    <tr>
        <td rowspan="6">model</td>
        <td>Intrusion Traffic Detection</td>
        <td>pkl file for traffic detection of six categories  intrusion and one benign traffic</td>
    </tr>
    <tr>
        <td>Malware Traffic Detection</td>
        <td>pkl file for five maleware traffic classification</td>
    </tr>
    <tr>
        <td>Applications Classification</td>
        <td>pkl file for five vido applications and five audio applications identification</td>
    </tr>
    <tr>
        <td>Traffic Categories over Tor</td>
        <td>pkl file for five traffic categories classification over Tor</td>
    </tr>
    <tr>
        <td>Traffic Categories over VPN</td>
        <td>pkl file for four traffic categories classification over VPN</td>
    </tr>
    <tr>
        <td>Encryption Technologies </td>
        <td>pkl file for VPN, No-VPN and Tor identification</td>
    </tr>
    <tr>
        <td rowspan="2">test_set</td>
        <td>manual_test</td>
        <td>test set  for training</td>
    </tr>
    <tr>
        <td>test_for_evaluation</td>
        <td>the test set for 'Evaluation_of_models'.</td>
    </tr>
    <tr>
</table>

## Quickly Run
next to do
## Reference
This repo provides the pre-trained models with test scripts for the experiments in the following paper (under peer reviewing).

>@article{"USENIX2023",
title={Fast and Efficacious:
A DL-based Performance Sensitive Intrusion Detection System},
author={M.M. GE, R. Feng, D.W. Kong, X.Z. Yu, Y.W. Zheng, Y. Liu},
year=2022
}