# REBERT: Domain-Specific BERT-based Multi-Task Learning

REBERT is a Python-based project designed for domain-specific multi-task learning leveraging BERT-based models. It supports customizable training and evaluation pipelines for tasks involving multiple domains.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running `default.py`](#running-defaultpy)
  - [Running `main.py`](#running-mainpy)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Output](#output)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction

REBERT leverages a BERT-based architecture for multi-task learning across multiple domains. It enables domain-specific fine-tuning and supports tasks such as classification and regression. With customizable options, it provides flexibility for training on specific datasets and evaluating across diverse domains.

---

## Features

- Multi-domain support for training and testing.
- BERT-based architecture (`bert-base-uncased` or other pre-trained variants).
- Customizable parameters for fine-tuning.
- Handles multi-task learning across domains.
- Outputs stored in a structured format for downstream analysis.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Duddu-Hriday/REBERT.git
   cd REBERT
   ```

## Usage

To run the model training and evaluation process with the default configurations, simply execute the `default.py` script as follows:

### Running the Script

  ```bash
  python default.py
  ```
### Running main.py
main.py provides a customizable interface for training and testing. Example usage:
```bash
python main.py --train_batch_size 16 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 3 --data_dir data/fdu-mtl/ --model_type cobe --model_name_or_path bert-base-uncased --train_domains books dvd --test_domains electronics kitchen_housewares --do_train --do_test
```
### Running default.py
default.py provides all the default paramters. Example usage:
```bash
python default.py
```
## Parameters:

- `--train_batch_size`: Batch size for training.
- `--eval_batch_size`: Batch size for evaluation.
- `--learning_rate`: Learning rate for optimization.
- `--num_train_epochs`: Number of epochs for training.
- `--data_dir`: Path to the dataset directory.
- `--model_type`: Model type (e.g., cobe).
- `--model_name_or_path`: Pre-trained model name or path.
- `--train_domains`: Domains for training (e.g., books dvd).
- `--test_domains`: Domains for testing (e.g., electronics kitchen_housewares).
- `--do_train`: Flag to enable training.
- `--do_test`: Flag to enable testing.


## Dependencies
Python 3.x
PyTorch
Transformers
Other dependencies are listed in the requirements.txt file.

## Configuration
All configurable parameters, including batch size, learning rate, and model paths, can be adjusted by modifying the command-line arguments when running main.py. Default configurations are available in default.py if no parameters are passed.

## Output
Training and evaluation logs will be output to the console and saved in specified directories.

## Contributors
- D Hriday - 210010016@iitdh.ac.in,
- KSN Manikanta - 210010050@iitdh.ac.in,
- M Suresh - 210010030@iitdh.ac.in

## License
This project is licensed under the MIT License - see the LICENSE file for details.
