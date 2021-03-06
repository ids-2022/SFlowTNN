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
        <td>test_set</td>
        <td>test_for_evaluation</td>
        <td>the test set for 'Evaluation_of_models'.</td>
    </tr>
</table>

## Dependency
- python version:3.8.7
- pip install tourch==1.18.0
- pip install numpy==1.16.4
- pip install dpkt== 1.9.2
- pip install scapy==2.4.5
- pip install -U scikit-learn
- pip install Pillow

## Quickly Run
`python predict.py input_path model_path`<br>
- input_path: your input file path
- model_path: your predict model path

