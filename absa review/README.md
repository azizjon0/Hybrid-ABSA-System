# ABSA Review App

This directory contains a Streamlit application developed to manually validate and correct ABSA predictions.

The application was created to accelerate the construction of a high-quality **gold-standard dataset** for Aspect-Based Sentiment Analysis.

## Features

- Review one sample at a time
- Display the original review text
- Show the predicted aspect and sentiment
- Edit incorrect aspect spans
- Correct sentiment labels
- Remove invalid predictions
- Add annotation comments
- Save corrections directly to the dataset
- Resume annotation from the last processed sample

## Purpose

The application was used during the manual validation stage of this project.

Using this interface, every aspect prediction from the Farmicia dataset (200 Yelp reviews) was manually inspected and corrected when necessary, resulting in a fully validated gold-standard dataset used for evaluation.

The interface significantly reduced the time required for manual annotation compared with editing the dataset directly in a spreadsheet.