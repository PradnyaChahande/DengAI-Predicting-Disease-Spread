library(forecast) 
library(fpp) 
library(caret) 
library(neuralnet) 
library(randomForest) 
library(psych)
library(VIM) 
library(mice) 
library(ResourceSelection) 
library(corrplot) 
library(party)
library(magrittr)
library('dplyr')

par(mfrow=c(1,1))

#dengue_features_train_combined is one week shifted data

mydata=read.csv("C:/Users/Pradnya/Documents/TTU_Spring_2017/Predictive_Analysis/Finals/dengue_features_train_combined.csv")
mytestdata=read.csv("C:/Users/Pradnya/Documents/TTU_Spring_2017/Predictive_Analysis/Finals/dengue_features_test.csv")
submissions = read.csv('C:/Users/Pradnya/Documents/TTU_Spring_2017/Predictive_Analysis/Finals/submission_format.csv')

aggr(mydata, prop=T,numbers=T)

#head(mydata[,-25])
#head(mytestdata)

full = bind_rows(mydata[,-25],mytestdata)

features = c('ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm',	'reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm')

f_related =c('reanalysis_min_air_temp_k',
             'reanalysis_specific_humidity_g_per_kg',
               'reanalysis_dew_point_temp_k',
               'station_avg_temp_c',
               'station_min_temp_c')

full[features] %<>% na.locf(fromLast = TRUE)

full$total_cases = 0
full$total_cases[1:1456] = mydata[,25]

sj=full[1:936,] 
iq=full[937:1456,] 

#sj$total_cases = mydata[1:936,25]
#iq$total_cases = mydata[937:1456,25]

sj_test = full[1457:1716,-25]
iq_test = full[1717:1872,-25]

#count(sj_test)

head(iq)
aggr(sj, prop=T,numbers=T)
aggr(iq, prop=T,numbers=T)


mycor=cor(iq[,5:25]) 
corrplot(mycor, method="circle", type="lower") 

mycor=cor(sj[,5:25]) 
corrplot(mycor, method="circle", type="lower") 

kdepairs(sj[,5:25])
kdepairs(iq[,5:25])

sjtot=ts(sj$total_cases, frequency=52, start=c(1990,04,30)) 
iqtot=ts(iq$total_cases, frequency=52, start=c(2000,07,01)) 

#par(mfrow=c(1,2))
plot(sjtot)
plot(iqtot)

acf(sjtot)
pacf(sjtot)

tsdisplay(sjtot)
tsdisplay(diff(sjtot,1))


f=auto.arima(sjtot, xreg =sj[f_related])
f1=auto.arima(iqtot,xreg =iq[f_related])

#----------------------xreg Seasonal Arima-----------------------------------

#score = 34.8846

fit <- Arima(sjtot, order=c(1,1,1), seasonal=c(0,0,0), xreg =sj[f_related] ) 
fit_Arima=forecast(fit, h = 260, xreg = sj_test[f_related])
plot(forecast(fit, h = 260, xreg = sj_test[f_related]))
tsdisplay(residuals(fit))
Box.test(residuals(fit_Arima),fitdf=7, lag=151, type="Ljung")  




fit_iq <- Arima(iqtot, order=c(0,1,2), seasonal=c(0,0,1),xreg =iq[f_related]) 
fit_Arima_iq=forecast(fit_iq, h = 156, xreg =iq_test[f_related])
plot(forecast(fit_iq, h = 156, xreg =iq_test[f_related]))
tsdisplay(residuals(fit_Arima_iq))
Box.test(residuals(fit_Arima_iq), fitdf=8, lag=151, type="Ljung")
fit_Arima_iq$mean


xreg_arima_sj_sol <- data.frame(submissions[1:260,-4], total_cases = round(fit_Arima$mean))

xreg_arima_iq_sol <- data.frame(submissions[261:416,-4], total_cases =round(fit_Arima_iq$mean))

xreg_arima_solution <- bind_rows(xreg_arima_sj_sol,xreg_arima_iq_sol)

write.csv(xreg_arima_solution, file = 'xreg_arima_predicted_Solution.csv', row.names = F)


#----------------------Seasonal Arima-----------------------------------

#score = 25.7163 Rank 45 out of 788 

fit <- Arima(sjtot, order=c(1,1,1), seasonal=c(0,0,0)) 
fit_Arima=forecast(fit, h = 260)
plot(forecast(fit))
tsdisplay(residuals(fit))
Box.test(residuals(fit_Arima),fitdf=2, lag=151, type="Ljung")  

fit_iq <- Arima(iqtot, order=c(0,1,2), seasonal=c(0,0,1)) 
fit_Arima_iq=forecast(fit_iq, h = 156)
plot(forecast(fit_Arima_iq))
tsdisplay(residuals(fit_Arima_iq))
Box.test(residuals(fit_Arima_iq), fitdf=3, lag=151, type="Ljung")

arima_sj_sol <- data.frame(submissions[1:260,-4], total_cases = round(fit_Arima$mean))

arima_iq_sol <- data.frame(submissions[261:416,-4], total_cases =round(fit_Arima_iq$mean))

arima_solution <- bind_rows(arima_sj_sol,arima_iq_sol)

write.csv(arima_solution, file = 'arima_predicted_Solution.csv', row.names = F)

##------------------------------------ NNETAR ---------------------- ----------------------##

#score =34.8846 

fit_nnetar_sj <- nnetar(sjtot,repeats=25, size=12, decay=0.1,linout=TRUE)
plot(forecast(fit_nnetar_sj,h=260))
a=forecast(fit_nnetar_sj,h=260)


fit_nnetar_iq <- nnetar(iqtot,repeats=25, size=18, decay=0.1,linout=TRUE)
plot(forecast(fit_nnetar_iq,h=156))
b=forecast(fit_nnetar_sj,h=156)

nnetar_sj_sol <- data.frame(submissions[1:260,-4], total_cases = round(a$mean))

nnetar_iq_sol <- data.frame(submissions[261:416,-4], total_cases =round(b$mean))

nnetar_solution <- bind_rows(nnetar_sj_sol,nnetar_iq_sol)

write.csv(nnetar_solution, file = 'nnetar_predicted_Solution.csv', row.names = F)


## -------------------------------------------- Random Forest1 start ----------------------------##

##Score = 28.1538

sj_rf_model <- randomForest(total_cases ~
                              ndvi_ne + ndvi_nw + ndvi_se + ndvi_sw + precipitation_amt_mm +	
                              reanalysis_air_temp_k + reanalysis_avg_temp_k + reanalysis_dew_point_temp_k +
                              reanalysis_max_air_temp_k + reanalysis_min_air_temp_k + reanalysis_precip_amt_kg_per_m2 +
                              reanalysis_relative_humidity_percent + reanalysis_sat_precip_amt_mm +
                              reanalysis_specific_humidity_g_per_kg + reanalysis_tdtr_k + station_avg_temp_c +
                              station_diur_temp_rng_c + station_max_temp_c + station_min_temp_c + station_precip_mm
                            , data = sj)

print(sj_rf_model)

sj_rf_prediction<- predict(object=sj_rf_model, sj_test)

iq_rf_model <- randomForest(total_cases ~
                              ndvi_ne + ndvi_nw + ndvi_se + ndvi_sw + precipitation_amt_mm +	
                              reanalysis_air_temp_k + reanalysis_avg_temp_k + reanalysis_dew_point_temp_k +
                              reanalysis_max_air_temp_k + reanalysis_min_air_temp_k + reanalysis_precip_amt_kg_per_m2 +
                              reanalysis_relative_humidity_percent + reanalysis_sat_precip_amt_mm +
                              reanalysis_specific_humidity_g_per_kg + reanalysis_tdtr_k + station_avg_temp_c +
                              station_diur_temp_rng_c + station_max_temp_c + station_min_temp_c + station_precip_mm
                            , data = iq)

print(iq_rf_model)

iq_rf_prediction<- predict(object=iq_rf_model, iq_test)

rf_sj_sol <- data.frame(submissions[1:260,-4], total_cases = round(sj_rf_prediction))
rf_iq_sol <- data.frame(submissions[261:416,-4], total_cases =round(iq_rf_prediction))

rf_solution <- bind_rows(rf_sj_sol,rf_iq_sol)

write.csv(rf_solution, file = 'rf_predicted_Solution.csv', row.names = F)

##---------------------------------------------Random Forest1 End--------------------------------------##

## -------------------------------------------- Random Forest2 start ----------------------------##
## score = 27.2933, used only those variable which have higher correlation to total_cases
sj_rf1_model <- randomForest(total_cases ~
                              reanalysis_specific_humidity_g_per_kg +
                              reanalysis_dew_point_temp_k + 
                              station_avg_temp_c +
                              station_min_temp_c
                            , data = sj)

print(sj_rf1_model)

sj_rf1_prediction<- predict(object=sj_rf1_model, sj_test)

iq_rf1_model <- randomForest(total_cases ~
                              reanalysis_specific_humidity_g_per_kg +
                              reanalysis_dew_point_temp_k + 
                              station_avg_temp_c +
                              station_min_temp_c
                            , data = iq)

print(iq_rf1_model)

iq_rf1_prediction<- predict(object=iq_rf1_model, iq_test)

rf1_sj_sol <- data.frame(submissions[1:260,-4], total_cases = round(sj_rf1_prediction))
rf1_iq_sol <- data.frame(submissions[261:416,-4], total_cases =round(iq_rf1_prediction))

rf1_solution <- bind_rows(rf1_sj_sol,rf1_iq_sol)

write.csv(rf1_solution, file = 'rf1_predicted_Solution1.csv', row.names = F)

##---------------------------------------------Random Forest2 End--------------------------------------##


