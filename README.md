Problem Discussion

In this paper, we attempt to develop machine learning models to predict the outbreak of Dengue in the cities of San Juan and Iquitos in South America. Dengue is a mosquito-borne tropical disease caused by the dengue virus. As it is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation. We are given historical data for these two cities aggregated by the number of Dengue cases reported weekly over the period from 1990 to 2008. The data also contains climate data from four different sources. Since this is a time series data, we discuss below our approach to build accurate machine learning models based on time series, and methods to do exploratory data analysis, data visualization and feature selection before applying predictive techniques.

Introduction

Significance

Dengue fever is a mosquito-borne disease that occurs in tropical and sub-tropical parts of the world. In mild cases, symptoms are like the flu: fever, rash, and muscle and joint pain. In severe cases, dengue fever can cause severe bleeding, low blood pressure, and even death. Because it is carried by mosquitoes, the transmission dynamics of dengue are related to climate variables such as temperature and precipitation. Although the relationship to climate is complex, a growing number of scientists argue that climate change is likely to produce distributional shifts that will have significant public health implications worldwide. In recent years’ dengue fever has been spreading. Historically, the disease has been most prevalent in Southeast Asia and the Pacific islands. These days many of the nearly half billion cases per year are occurring in Latin America. A time series model having good degree of accuracy can be utilized in predicting the number off Dengue cases and thereby help authorities and medical professionals to do contingency planning. From the perspective of data scientists analyzing time-series data like this proves to be a worthwhile challenge which helps in developing understanding different dynamics affecting time-series models. 

Related Work
Literature Overview
We now discuss different research papers, journals, conferences and publications where various time series models that were used for forecasting outbreak of epidemic diseases like Malaria, Dengue, Typhoid etc. 

1. SARIMA (Seasonal ARIMA) implementation on
time series to forecast the number of Malaria
incidence, Adhistya Erna Permanasari et. al, ResearchGate; Oct 2013. 
In this paper, importance of choosing an a-priori forecasting method in predicting the number of disease incidence is underlined. This paper analyses and presents the use of Seasonal Autoregressive Integrated Moving Average (SARIMA) method for developing a forecasting model that able to support and provide prediction number of disease incidence in human. The dataset for model development was collected from time series data of Malaria occurrences in United States obtained from a study published by Centers for Disease Control and Prevention (CDC). It resulted SARIMA (0,1,1) (1,1,1)12 as the selected model. The model achieved 21.6% for Mean Absolute Percentage Error (MAPE). It indicated the capability of final model to closely represent and made prediction based on the Malaria historical dataset.

2. Comparison of different Methods for Univariate Time Series Imputation in R; Steffen Moritz et al, / Cologne University of Applied Sciences
This paper provides an overview of univariate time series imputation in general and an in-detail insight into the respective implementations within R packages. Furthermore, we experimentally compare the R functions on different time series using four different ratios of missing data. Our results show that either an interpolation with seasonal kalman filter from the zoo package or a linear interpolation on seasonal loess decomposed data from the forecast package were the most effective methods for dealing with missing data in most of the scenarios assessed in this paper.

3. Comparison of ARIMA and Random Forest time series models for prediction of avian influenza H5N1 outbreaks, BMC Bioinformatics 2014, Michael J Kane, Published: 13 August 2014
In assembling this review, the authors applied ARIMA and Random Forest time series models to incidence data of outbreaks of highly pathogenic avian influenza (H5N1) in Egypt, available through the online EMPRES-I system. The authors found that the Random Forest model outperformed the ARIMA model in predictive ability. Random Forest time series modeling provided enhanced predictive ability over existing time series models for the prediction of infectious disease outbreaks. This result, along with those showing the concordance
between bird and human outbreaks (Rabinowitz et al. 2012), provides a new approach to predicting these dangerous outbreaks in bird populations based on existing, freely available data. The authors uncover time-series structure of outbreak severity for highly pathogenic avian influenza and thus conclude that the Random Forest model is effective for predicting outbreaks of H5N1 in Egypt.

4. Time series regression model for infectious disease and weather, Computers in Biology and Medicine, Chisato Imai, Ben Armstrong et al, Environmental Research, Accepted 28 June 2015
In this paper, the authors discuss and present potential solutions for five issues often arising in such analyses: changes in immune population, strong autocorrelations, a wide range of plausible lag structures and association patterns, seasonality adjustments, and large over dispersion. The potential approaches were illustrated with datasets of cholera cases and rainfall from Bangladesh and influenza and temperature in Tokyo. Though this article focuses on the application of the traditional time series regression to infectious diseases and weather factors, we also briefly introduce alternative approaches, including mathematical modeling, wavelet analysis, and autoregressive integrated moving average (ARIMA) models.
Modifications proposed to standard time series regression practice include using sums of past cases
as proxies for the immune population, and using the logarithm of lagged disease counts to control autocorrelation due to true contagion, both of which are motivated from “susceptible-infectious-recovered” (SIR) models. The complexity of lag structures and association patterns can often be informed by biological mechanisms and explored by using distributed lag non-linear models. For over dispersed models, alternative distribution models such as quasi-Poisson and negative binomial should be considered. Time series regression can be used to investigate dependence of infectious diseases on weather, but may need modifying to allow for features specific to this context.

5. Application of time-series autoregressive integrated moving average model in predicting the epidemic situation of newcastle disease, Jing Li and Chongwei Hu, World Automation Congress, Vol.2, No.3, June 2010
This study fitted an ARIMA model based on the monthly incidence of Newcastle disease collected from the Veterinary bulletin in two areas (A and B) from Jan. 2000 to Dec. 2007. The SPSS software was used to set up time series model which was applied to predict the incidence of Newcastle disease from Jan. to Dec. 2008 in above areas and validated by comparing with the actual incidence. The predicted incidence of ARIMA model was consistent with the actual incidence of Newcastle disease, which indicated the constructed ARIMA model can be applied to predict the incidence of Newcastle disease to supply the reliable reference for this disease in future.
Dataset
Data Mining and Data Cleaning
In this dataset, we are given two cities, San Juan and Iquitos, with test data for each city spanning 5 and 3 years, while the train data for each city spanning 18 and 8 years respectively. The data is given on a weekly basis along with total cases of Dengue during that week as the test label.

As with any other problem dealing with data science, we start by performing an Exploratory Data Analysis (EDA). EDA is an approach/philosophy for data analysis that employs a variety of techniques (mostly graphical) to 
a.	Maximize insight into a data set; 
b.	Uncover underlying structure; 
c.	Extract important variables;
d.	Detect outliers and anomalies;
e.	Test underlying assumptions;
f.	Develop parsimonious models; and
g.	Determine optimal factor settings.1
The training and test data had many missing values especially in the numerical variables. These were imputed by calculating the mean of the respective fields. Since our train labels are the number of Dengue cases per week, and since Dengue outbreak is known to exhibit high seasonality round the year, we try to fir our labels to a time series. Therefore, in time series analysis, the role of predictor variables in training our model is limited. 

However, to forecast, we need to make our time series data stationary. What this means is that we need to remove seasonality from our data to extra the base trend of series. A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time. 

Data Visualization
Data Visualization is an important aspect of exploratory data analysis (EDA). In this case, can visualize our target variable across the time period to view its seasonality in the dengue outbreaks.  Plots are graphed to see that the outbreaks are fairly seasonal for both cities.    

The extreme peaks are due to epidemics. Thereafter we also have a look at the affecting variables to find out the co-relation between variables. The correlation plot of the climate variables gathered from various sources are also plotted. 
 

Proposed Approaches
Application of Predictive Techniques

We tried 3 different models on our time series data, namely Random Forest, Neural Networks and Seasonal ARIMA. We now discuss each of these models, and how we used them.

Predictive Technique #1 Random Forest

A random forest is a multi-way classifier which consists of number of trees, with each tree grown using some form of randomization [2]. The leaf nodes of each decision tree are labelled by estimates of the posterior distribution over the image classes [2].
Each internal node contains a test that best splits the space of data to be classified [2]. An image is classified by sending it down every tree and aggregating the reached leaf distributions [2]. Randomness can be injected at two points during training: in subsampling the training data so that each tree is grown using a different subset; and in selecting the node tests - Formulation / Libraries [2].

As per the given problem statement, we are required to not only classify the 	given test samples but also predict the probabilities of each sample belonging to each species. This probability gives some kind of confidence on the prediction. Given below are details of classifier and calibrator hyperparameter values we tuned after trial and error to achieve optimal model performance:

Libraries(R): 
We used randomForest function from randomForest library to model and predict the number of weekly cases for both cities.
Predictive Technique #2 Seasonal ARIMA

An autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model. Both models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity. 

The AR part of ARIMA indicates that the evolving variable of interest is regressed on its own lagged (i.e., prior) values. The MA part indicates that the regression error is a linear combination of error terms whose values occurred contemporaneously and at various times in the past. The I (for "integrated") indicates that the data values have been replaced with the difference between their values and the previous values (and this differencing process may have been performed more than once). The purpose of each of these features is to make the model fit the data as well as possible. 

Seasonal ARIMA models are usually denoted ARIMA(p,d,q)(P,D,Q)m, where m refers to the number of periods in each season, and the uppercase P,D,Q refer to the autoregressive, differencing, and moving average terms for the seasonal part of the ARIMA model.

Given below are details of Seasonal ARIMA model we tuned after trial and error; and our optimal model performance:

Libraries(R): 
1.	‘Arima’ function was used from the forecast library.
2.	Auto.arima function was used to calibrate the best parameters.
Predictive Technique #3 Neural Network

Artificial neural networks were designed to be modelled after the structure of the brain. They were first devised in 1943 by researchers Warren McCulloch and Walter Pitts [3]. Backpropagation, discovered by Paul Werbos [4], is a way of training artificial neural networks by attempting to minimize the errors. This algorithm allowed scientists to train artificial networks much more quickly. Each artificial neural network consists of many hidden layers. Each hidden layer in the artificial neural network consists of multiple nodes. Each node is linked to other nodes using incoming and outgoing connections. Each of the connections can have a different, adjustable weight. Data is passed through these many hidden layers and the output is eventually interpreted as different results.
 
In this example diagram, there are three input nodes, shown in red. Each input node represents a parameter from the dataset being used. Ideally the data from the dataset would be pre-processed and normalized before being put into the input nodes.
There is only one hidden layer in this example and it is represented by the nodes in blue. This hidden layer has four nodes in it. Some artificial neural networks have more than one hidden layer.

The output layer in this example diagram is shown in green and has two nodes. The connections between all the nodes (represented by black arrows in this diagram) are weighted differently during the training process. We are using the nnetar function from the library forecast to set up the neural network. We used a number of hidden nodes however, the best nnetar result was achieved by using a model with 18 nodes for San Juan and 12 nodes for Iquitos in the hidden layer.

Results and Discussions
Model Performance
We uploaded the predicted number of weekly cases of Dengue in both cities to Driven Data under the team name – ‘TTU_Max_Pradnya’. Our Driven Data MAE (Mean Absolute Error) scores for the three models are tabulated next:

Model	             MAE 
Seasonal ARIMA	  25.7163
Random Forest   	27.2933
Neural Net	      34.8846

Learning and Limitations
We now discuss our learnings in terms of advantages and limitations of each model.

Overall Competition
Learnings
•	We should not forecast at time T by training models with values at time T. 
•	We learned to work with time series and forecast with the help of external regressors.
•	Working with time series and forecasting is more error prone as compared to predictions based on regression and classification.

Limitations
•	Highly co-related environmental factors when used as external regressors for time series model led to decreased forecasting power.
•	We will need to run repeated cross validations and grid search to better estimate the parameters.

Random Forest
Features
•	It’s one of the most accurate learning algorithms available
•	Hyper parameter tuning relatively easy

Limitations
•	Can attempt to fit a complex tree to the data, leading to overfitting.
•	Intuitively easy to understand, but difficult to get an insight as to what the algorithm does.

Seasonal ARIMA
Features
•	The ARIMA is very easy to deal with. We can estimate how many AR, and MA parameters we need by looking at the autocorrelation and partial-autocorrelation functions.
•	It can analyze effects of repeated measure factors.

Limitations
•	There's not really a way to introduce priors into ARIMA models. 
•	ARIMA would not be an appropriate model for time series that are not stationary (not even after differencing).

Neural Network
Features
•	Few assumptions need to be verified for constructing models.
•	ANNs can model complex nonlinear relationships between independent and dependent variables, and so they allow the inclusion of many variables.
Limitations
•	Greater computational burden, proneness to overfitting, and the empirical nature of model development.
•	Tuning the hyper parameters is challenging. Tuning neural networks need high computing power. The key issue lies in the selection of proper size of the hidden layer nodes to attain high efficiency with less computational complexity.

Conclusions
We observed better performance for Arima using the forecast model as compared to auto.arima. Our best performing Arima gave a MAE of 25.7163 along with a rank of 45 on Public leaderboard. Random Forest gave good accuracy and ran efficiently over the data set and outperformed Neural. As future work, we plan to use cross validation and grid search to better estimate parameters for increasing the efficiency of the models. We also plan to study Generalized Additive/Multiplicative Models.
