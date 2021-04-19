# Global-Data-Analysis-for-Social-Unrest-Prediction


## Team ExaMine : Raksha Harish, Tavleen Sahota, Harman Jit Singh

### Video: https://youtu.be/Dl3AdkKj-sI

## Problem Statement  
In this project, we integrated different datasets in order to analyse different factors like world food prices, Gross Domestic Product (GDP) rates, crime and poverty rates, armed conflicts and historical event patterns, and dynamic Twitter streams. Using temporal burst patterns in historical events data helped in the prediction of impactful social unrest events across the world by capturing the social, political and economic contexts of different countries over different timelines. The main aim of this project is:    
● To help world governments and social scientists make better policies by considering all the factors for establishing peaceful governance    
● To help in predicting future social unrest, thereby giving enough time for the governments to implement actions to either handle the unrest events or prevent it altogether.    


## Project Execution Steps  

### Datasets used:  
1. GDELT Global Events dataset (using Google BigQuery)  
2. ACLED (Armed Conflict Data)  
3. World Food Program (WFP) data  
4. UN Global Criminal Rates data  
5. Latest Twitter Data using Twitter API  

### Download the Following python packages  
1. numpy, pandas  
2. seaborn, matplotlib  
3. nltk, dash, flask
4. sklearn, plotly, pickle  
5. mchmm, pomegranate, graphtool  
6. tweepy, geocoder  
7. google.cloud, bigquery, pyarrow  

Create a free Google Cloud Platform Account, and download the json credentials file to connect to GCP BigQuery.  
Setup API keys and Access Tokens in order to acess the free version of Twitter API for live twitter streams.  

### Execution of the Backend Framework 
1. To extract relevant data : "Data_Extraction.ipynb"  
2. To perform Exploratory Data Analysis : "Final-EDA.ipynb"  
3. To build Linear Regression model, Random Forest Regressor and simulate Markov Chain for Event history : "GDELT - LReg + RandomF + MarkovCh.ipynb"  
4. To build Gradient Boosting Machine and Isolated Forest Model using H2O ai framework : "H2O - GBM, IFM models.ipynb"  
5. To get the probability of simulated event sequence and test the model for random event sequence : "Event Sequence Prediction Using HMM.ipynb"  
6. To get trending hastags on Twitter, perform analysis and draw network graphs : Go to the "Twitter" folder and execute "Twitter.ipynb"  

### Save all the executed models as Pickle Files

### Execution of the Final Data Product - Project Dashboard  
1. To prepare the extracted data for UI : "DataPrep_UI.ipynb"
2. Change to the "web_dev/" folder
3. In the terminal, run the command "python app.py" to run the UI Dashboard 
