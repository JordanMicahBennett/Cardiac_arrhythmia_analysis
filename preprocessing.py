import os
a=os.getcwd()

datafile=open(a+"\\dataset.txt")  
b=datafile.read() 
c=b.split("\n")
newfile=open(a+"\\preprocesseddata.txt","w")

#print(len(c))
del_row=[]
for i in range(0,len(c)):
	e=c[i].split(",")
	if len(e)!=280:
		#print(i)
		#print(len(e))
		del_row.append(i)

#print(del_row)
for i in del_row:
	del(c[i])
#print(len(c))
data=[]


###################            omitting others for ambigiousness 
anomaly_name=["Normal","Isochemic Changes","Old Anterior Myocardic Infraction",
"Old Inferior Myocardic Infraction","Sinus Tachycardy","Sinus Bradycardy",
"Ventricular Premature Contraction","Supraventricular Premature Contraction",
"Left Bundle Branch Block","Right Bundle Branch Block","1. degree AtrioVentricular block",
"2. degree AV block","3. degree AV block","Left ventricule hypertrophy","Atrial Fibrillation or Flutter",
"Others"]
feature_dict={}
row=0
for d in c:
	#row+=1
	e=d.split(",")
	f=int(e[279])
	if len(e)!=280:
		continue
	if anomaly_name[f-1] in feature_dict.keys():
		feature_dict[anomaly_name[f-1]]+=1
		if e[279]!="16":
			data.append(e)
	else:
		feature_dict[anomaly_name[f-1]]=1
		if e[279]!="16":
			data.append(e)
	row+=1
#print(row)
#print(feature_dict)
#print(len(data))

import pandas as pd
import numpy as np
from numpy import array
import random
from sklearn.preprocessing import StandardScaler

list_data=list(data)
#X=arr_data[:,0:278]
#y=arr_data[:,279]
#print(y)
missing={}
for i in range(0,len(list_data)):
	for j in range(0,len(list_data[0])):
		if list_data[i][j]=="?":
			if j in missing.keys():
				missing[j].append(i)
			else:
				missing[j]=[i]
#print(missing)
#print(len(missing[13]))
#print(len(missing[11]))
#print(len(missing[10]))
#print(data)
mean={}
for i in missing.keys():
	if int(i)!=13:
		if i not in mean.keys():
			mean[i]=0
for i in list_data:
	for j in mean.keys():
		if i[j]!="?":
			mean[j]=mean[j]+float(i[j])
for i in mean:
	mean[i]=mean[i]/(858-len(missing[i]))

for i in range(0,len(list_data)):
	for j in range(0,len(list_data[i])):
		if j!=13 and list_data[i][j]=="?":
			list_data[i][j]=mean[j]
#print(mean)
#print(arr_data[:,-267])
#print(X[0][-3])

for i in range(0,len(list_data)):
	del(list_data[i][13])
#print(list_data)
arr_data=np.float_(list_data)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier  
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


kfold=KFold(10,True)
count=np.zeros((1,278))
size_arr=len(arr_data[0])
for train,test in kfold.split(arr_data):
	train_data=arr_data[train]
	test_data=arr_data[test]
	X_train=train_data[:,0:size_arr-2]
	y_train=train_data[:,size_arr-1]
	X_test=test_data[:,0:size_arr-2]
	y_test=test_data[:,size_arr-1]
	#model=RandomForestClassifier(n_estimators =100)
	sel = SelectFromModel(RandomForestClassifier(n_estimators = 40))
	sel.fit(X_train, y_train)
	#model.fit(X_train,y_train)
	#pred=model.predict(X_test)
	#acc=0;
	#for i in range(0,len(pred)):
		#if pred[i]==y_test[i]:
			#acc=acc+1
	#acc=acc/len(pred)
	#print((acc))
	#print(len(y_test))
	feat=sel.get_support()
	for i in range(0,len(feat)):
		if feat[i]==True:
			count[0][i]=count[0][i]+1

tot=0;
for i in count[0]:
	if i!=0:
		tot=tot+1
#print(tot)


#print(list_data)

for i in range(0,len(list_data)):
	for j in range(len(list_data[i])-2,0,-1):
		if count[0][j]==0:
			del(list_data[i][j])
#print(len(list_data[0]))
#print(list_data)

new_data=np.float_(list_data)
#print(new_data)
size_new=len(new_data[0])
scaler = StandardScaler() 

for train,test in kfold.split(new_data):
	train_data=new_data[train]
	test_data=new_data[test]
	#train = StandardScaler().fit_transform(train)
	#for i in range(0,len(train[i])-1):
		#nor_train=StandardScaler().fit_transorm(train[:,i])
		#print(nor_train)
	X_train=train_data[:,0:size_new-2]
	y_train=train_data[:,size_new-1]
	X_test=test_data[:,0:size_new-2]
	y_test=test_data[:,size_new-1]
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)  
	model_rfc=RandomForestClassifier(n_estimators =40)
	model_rfc.fit(X_train,y_train)
	pred_rfc=model_rfc.predict(X_test)
	acc_rfc=0
	for i in range(0,len(pred_rfc)):
		if pred_rfc[i]==y_test[i]:
			acc_rfc=acc_rfc+1
	acc_rfc=acc_rfc/len(pred_rfc)
	print("rfc "+str(acc_rfc))

	cm = confusion_matrix(y_target=y_test,y_predicted=pred_rfc,binary=False)
	fig, ax = plot_confusion_matrix(conf_mat=cm)
	plt.show()

	model_knn = KNeighborsClassifier(n_neighbors=15)  
	model_knn.fit(X_train, y_train) 
	pred_knn=model_knn.predict(X_test)
	acc_knn=0
	for i in range(0,len(pred_knn)):
		if pred_knn[i]==y_test[i]:
			acc_knn=acc_knn+1
	acc_knn=acc_knn/len(pred_knn)
	print("knn "+str(acc_knn)+"\n")
	#print(X_test)


datafile.close()
newfile.close()

