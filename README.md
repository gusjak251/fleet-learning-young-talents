# Fleet learning - Eskilstuna

Fleet learning is a project in collaboration with Zenseact and Volvo. They use federated learning technology to train ML models on cars all around the world.

## Table of contents

- [ Setup ](#setup)
- [ Sectors ]()
    1. [ Metadata visualization ](#metadata-visualization)
    2. [ Client selection strategies ](#selection)
    3. [ Novelty detection ](#outliers)

## Setup
Installation
    
    pip install -r requirements.txt

Running the model

    python3 main.py

Running the dashboard

    python3 metadata_visualization.py

## Metadata visualization

### Dashboard 

The dashboard was originally developed by our competitors in Örebro. We took inspiration from their solution and made our own dashboard, to showcase other aspects and features of the data.

Our version consists of three sections:

#### The multi-selection graph

The multi-selection graph, which allows users to combine different features from the metadata, so that the information from chosen clients can be seen from new perspectives.

#### Outlier map

A world map, which displays all the data points and their locations. Points which have been classified as anomalies are highlighted in blue, and the remaining ones are yellow.

The outliers have been identified by the isolation forest algorithm from the scikit-learn library.

Ideas:

Color countries/cities according to how much data they contain



#### Performance graph 

Visualization of the training loss, validation loss & test loss from the model

To do: 

Make it possible to isolate training loss from testing loss

Show parameters used for the selected training session

Show mean values + min & max values