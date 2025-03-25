# ArtExtract Projects @ HumanAI Umbrella Organization

## Google Summer of Code 2025 - Evaluation Tasks

This repository contains the solutions for the evaluation tasks for the ArtExtract project under the HumanAI Umbrella Organization as part of Google Summer of Code (GSoC) 2025.

```
Evaluation Test: ArtExtract
```

## Repository Structure

```
|-- Data_WikiData.ipynb     # Notebook for data processing and exploration
|-- task1.ipynb             # Solution for Task 1: Convolutional-Recurrent Architectures
|-- task2.ipynb             # Solution for Task 2: Similarity in Paintings
|-- app_task1.py            # Application script for Task 1
|-- app_task2.py            # Application script for Task 2
|-- README.md               # Project Documentation
```

---

## Task 1: Convolutional-Recurrent Architectures

### Task Description
The goal of this task is to build a **Convolutional-Recurrent model** for classifying paintings based on **Style, Artist, Genre, and other attributes**. The chosen dataset is **ArtGAN's WikiArt Dataset** ([Link](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)).

### Files
- `Data_WikiData.ipynb` – Contains the processing of csv files of WikiArt dataset.
- `task1.ipynb` – Contains the model implementation and evaluation.
- `app_task1.py` – A script to run the trained model for classification.

---

## Task 2: Similarity in Paintings

### Task Description
The goal is to build a **painting similarity model** to find **portraits with a similar face or pose** using the **National Gallery Of Art open dataset** ([Link](https://github.com/NationalGalleryOfArt/opendata)).

### Files
- `task2.ipynb` – Contains the similarity model implementation and evaluation.
- `app_task2.py` – A script to find similar paintings given an input image.

