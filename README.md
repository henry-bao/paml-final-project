# PAML Final Project - Predictive ML Models on US Accidents (2016-2023)

## About the Dataset

This is a countrywide car accident dataset that covers 49 states of the USA. The accident data were collected from February 2016 to March 2023, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by various entities, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road networks. The dataset currently contains approximately 7.7 million accident records. For more information about this dataset, please visit [here](https://smoosavi.org/datasets/us_accidents).

## About the Project

### Problem Addressed

The project focuses on building an application that analyzes and predicts traffic accidents across the United States, aiming to understand the factors leading to accidents and help reduce their occurrence through predictive modeling.

### Importance of the Problem

Traffic accidents are a major public safety concern, leading to significant human and economic losses annually. Predictive modeling can enhance road safety measures by providing insights that assist in preventive planning and real-time decision-making. Accurate predictions can also facilitate better urban planning and infrastructure development, aligning road safety enhancements with identified high-risk areas.

### Expected Findings

We anticipate that the integration of diverse data types and advanced machine learning algorithms will significantly improve the accuracy of accident predictions compared to existing models. This could profoundly impact public safety initiatives by providing governments and agencies with a tool to dynamically adjust to evolving road conditions and potentially hazardous situations in real-time.


## Instruction to Run the Streamlit App

1. Clone the repository
2. Install the required packages
   - `pip3 install -r requirements.txt`
3. Download the dataset from [Kaggle](https://www.kaggle.com/sobhanmoosavi/us-accidents)
4. Move the dataset to the `data` folder
5. Download the pre-trained models from [Box](https://cornell.box.com/s/td370oe6hnh03541hza0al5fxsvhu8xy)
6. Move the pre-trained models to the `models` folder
7. Run the Streamlit app
   - `python3 -m streamlit run app.py`
   - or `streamlit run app.py`
   - The app will be running on `http://localhost:8501`

## Acknowledgement

1. Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.
2. Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
