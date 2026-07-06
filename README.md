# Wine Review Analytics

End-to-end Data Analytics and Machine Learning pipeline for exploring, modeling, and generating insights from over 150,000 wine reviews. This project performs comprehensive exploratory data analysis, statistical profiling, predictive modeling, clustering, anomaly detection, and automatically generates a professional analytical PDF report with rich visualizations.

## Features

- Automated data cleaning and preprocessing
- Comprehensive exploratory data analysis (EDA)
- Statistical summary generation
- Wine rating and price distribution analysis
- Country, province, winery, and variety analytics
- Feature engineering
- Value score computation
- Correlation analysis
- Predictive machine learning models
- Feature importance analysis
- Wine clustering using K-Means
- Anomaly detection using Isolation Forest
- Automated professional PDF report generation
- Publication-quality visualizations
- Fully reproducible analysis pipeline

## Dataset

The project uses the Wine Reviews dataset containing approximately **150,930 wine reviews**.

Dataset includes information such as:

- Country
- Province
- Region
- Winery
- Variety
- Price
- Rating (Points)
- Description
- Designation
- Taster Information

## Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy
- ReportLab

## Machine Learning Models

### Regression

- Random Forest Regressor
- Gradient Boosting Regressor

### Clustering

- K-Means Clustering

### Anomaly Detection

- Isolation Forest

## Analytical Components

### Exploratory Data Analysis

- Missing value analysis
- Dataset profiling
- Distribution analysis
- Outlier handling
- Correlation exploration

### Business Analytics

- Country performance
- Variety performance
- Winery rankings
- Price analysis
- Rating trends
- Premium wine identification
- Value score computation

### Predictive Analytics

- Wine rating prediction
- Feature importance analysis
- Model evaluation
- Performance comparison

### Unsupervised Learning

- Customer-independent wine segmentation
- Cluster visualization
- Cluster statistics

### Anomaly Detection

- Identification of unusual wine reviews
- Premium pricing anomalies
- High-value outlier detection

## Generated Outputs

The pipeline automatically generates:

- Executive Summary
- Statistical Tables
- Distribution Charts
- Scatter Plots
- Country Analysis
- Variety Analysis
- Winery Rankings
- Feature Importance Graphs
- Cluster Analysis
- Anomaly Detection Results
- Professional PDF Report

## Project Structure

```
Wine_Review_Data_Analysis/
│
├── Dataset/
│   └── winemag-data_first150k.csv
│
├── Output/
│   ├── Charts
│   ├── Tables
│   ├── PDF Report
│   └── Processed Data
│
├── src/
│   ├── preprocessing.py
│   ├── analytics.py
│   ├── visualization.py
│   ├── machine_learning.py
│   ├── clustering.py
│   ├── anomaly_detection.py
│   └── report_generator.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository

```bash
git clone https://github.com/sagnik10/wine-review-analytics.git
```

Navigate into the project

```bash
cd wine-review-analytics
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the complete analytics pipeline

```bash
python main.py
```

The program automatically performs:

1. Data preprocessing
2. Exploratory analysis
3. Statistical profiling
4. Machine learning
5. Clustering
6. Anomaly detection
7. Visualization
8. PDF report generation

## Sample Results

The generated report includes:

- 150,930 analyzed wine reviews
- Average wine rating analysis
- Price distribution
- Country rankings
- Variety rankings
- Winery performance
- Machine learning prediction results
- Cluster summaries
- Anomaly detection findings
- Professional analytical report

## Applications

- Data Science Portfolio
- Machine Learning Demonstration
- Exploratory Data Analysis
- Statistical Learning
- Business Intelligence
- Predictive Analytics
- Data Visualization
- Educational Projects

## Future Improvements

- Interactive dashboard
- Streamlit web application
- Deep learning models
- Recommendation engine
- Time-series analysis
- Geographic visualization
- REST API
- Docker deployment
- Cloud integration

## License

This project is licensed under the Apache License 2.0.

You may freely use, modify, distribute, and build upon this project in accordance with the terms of the Apache License, Version 2.0.

See the LICENSE file for details.

## Citation

If you use this project in research, publications, or educational work, please cite this repository.

## Author

**Sagnik Sen**

Project Scientist-I  
India Meteorological Department

GitHub: https://github.com/sagnik10

## Acknowledgements

- Wine Reviews Dataset
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- ReportLab
- Open Source Python Community
