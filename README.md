# Content

## Introduction:
This project is supposed to showcase the results of our discussion of the question:
'What personal information is most commonly exposed during data breaches, and how does this exposure vary across different industries?' as part of the programming project: data science in python and R at the SoSe 25 at TU-Berlin.

## Data
Our analysis of the 4 hypothesis that make up our question is based on 2 complementary datasets:
a local, detailed and well structured dataset from the washington attorney generals office
and a broader, less detailed one compiled by volunteers given by the David McCandless Dataset that
we used to verify trends and draw comparisons.

Most of our analysis use the sector of the breached organization as an independent variable for both datasets.
The further dependent variables do differ however:
- In the attorney generals' dataset, we draw conclusions from the type of personal information exposed as well as the affected number of individuals
- And in the global dataset, we use data sensitivity, the number of records lost as well as the method used for the breach

## Hypothesis:
Here is an overview of the hypothesis, in order:

**Hypothesis 1**: 'The relationship between industry sector and exposed data follows clear, identifiable patterns.' 

**Hypothesis 2**: 'Certain industries tend to expose specific types of data'

**Hypothesis 3**: 'Breach size varies by industry, with some sectors facing consistently larger incidents.'

**Hypothesis 4**: 'The sector and number of records lost can help predict the breach method.'

The results of the analysis in all cases is yielded by the appropriate visualizations which can then be independently analysed by a sufficiently 
competent user. In the following, you will find a guide to set up this repo for yourself:

# Setup
## Project Overview
As a quick glance at the project will reveal, all visualization has been performed using python and standard data-science libraries like pandas and matplotlib, among others. It is organized by hypothesis which, as with the data-preprocessing script, all bear their own sub-directory. 
The outputs can then be found in a subdirectory of each hypothesis.

## Project Structure

datascience-pr/
│
├── Datasets_Cleaning.py # Script for cleaning and preprocessing datasets
├── Kaggle_DB.csv # Original Kaggle dataset
├── Kaggle_DB_updated.csv # Cleaned/processed Kaggle dataset
├── Washington_DB.csv # Washington dataset
│
├── Hypothesis1/
│ ├── Hypothesis1.py # Analysis for Hypothesis 1
│ └── Hypothesis1_Plots/ # Visualizations for Hypothesis 1
│
├── Hypothesis2/
│ ├── DataExplosure.py # Exposure data analysis
│ ├── ExposureIndustry.py # Industry-wise exposure analysis
│ └── Hypothesis2_Plots/ # Visualizations for Hypothesis 2
│
├── Hypothesis3/
│ ├── kaggle.py # Kaggle-related analysis
│ ├── WA.py # Washington-related analysis
│ └── Hypothesis3_Plots/ # Visualizations for Hypothesis 3
│
├── Hypothesis4/
│ ├── DecisionTree.py # Decision tree model
│ ├── Visualization.py # Visualizations for Hypothesis 4
│ └── Hypothesis4_Plots/ # Plots for Hypothesis 4
│
├── generate_all_plots.py # Script to generate all plots across hypotheses
├── requirements.txt # Python dependencies
├── .gitignore # Files/directories to ignore in version control
└── README.md # Project overview and documentation

## Setting up the environment

Its strongly recommended to set-up this project through a python virtual environment (conda or uv is obviously fine as well, though we won't include intructions for that here). This especially prevents issues with the version control of certain libraries between projects.

In order to run this, you will need Python 3.7 and upwards installed. If you don't have that, there are numerous instructions on how to install the most popular programming language in your operating system online.

To set up a new virtual environment, you can simply move into the project directory in the command line and run

` python3 -m venv ./venv`

Now, to activate the environment (you will have to always do this in the terminal after initializing a new session to run the scripts) simply run (in the same directory) the command 

`source ./venv/bin/activate`

in your linux operating system of choice. For other operating systems, please refer to other ressources to accomplish the same goal.

Now, since our project does rely on some dependencies in the form of python libraries, you can install those to then seemlessly use without further adjustments by running (inside the activated environment):

`pip3 install -r requirements.txt`

# How-to run
Note that this project does not come with all the visualizations pre-generated (as they are blocked from versioning by git in the .gitignore by their subdirectory). Thus, if you want to generate them all and then view them in the aforementioned subdirectories and after the previous setup, you can run

`python3 generate_all_plots.py`

in the main directory to automatically generate all files from the command line.

However, it is also possible to only generate plots associated with one of the sub-analysis in any given hypothesis. To do that for a given hypothesis, you can just run 

`python3 /HypothesisName/file_name.py`

which will then also interactively visualize the results in a seperate window as they are generated.

## Data preprocessing

You will note that we already have the (generally preprocessed) data from the global dataset included here, which is referenced in its updated form in all applicable scripts. However, in order to understand the cleanup from the non-tidy original file, you can refer to the 
`/data/Datasets_Cleaning.py` 
script for more information.

If you have any further questions, suggestions or want to contribute, write me a message or create an issue :)