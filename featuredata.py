#####-------------------getting the directory and file to work with 

import os
a=os.getcwd()


#####-------------------getting the text file and creating a new file to store the newly selected features

datafile=open(a+"\\dataset.txt")  
b=datafile.read() 
c=b.split("\n")
newfile=open(a+"\\newdata.txt","w")


#####--------------------Storing the anomaly name

anomaly_name=["Normal","Isochemic Changes","Old Anterior Myocardic Infraction",
"Old Inferior Myocardic Infraction","Sinus Tachycardy","Sinus Bradycardy",
"Ventricular Premature Contraction","Supraventricular Premature Contraction",
"Left Bundle Branch Block","Right Bundle Branch Block","1. degree AtrioVentricular block",
"2. degree AV block","3. degree AV block","Left ventricule hypertrophy","Atrial Fibrillation or Flutter",
"Others"]



#####-------------------finding out the instances where there might be some unusaualities in the data
#####-------------------and marking the instance number in order to rescind it from the data

feature_dict={}
row=0
for d in c:
	row+=1
	e=d.split(",")
	f=int(e[279])
	if len(e)!=280:
		continue
	if anomaly_name[f-1] in feature_dict.keys():
		feature_dict[anomaly_name[f-1]]+=1
	else:
		feature_dict[anomaly_name[f-1]]=1
#print(feature_dict)

#####-------------------The above print command will print the following:
#####-------------------{'Supraventricular Premature Contraction': 3, 
#####-------------------'Sinus Bradycardy': 50,
#####-------------------'Right Bundle Branch Block': 100,
#####-------------------'Normal': 489,
#####-------------------'Ventricular Premature Contraction': 6, 
#####-------------------'Left ventricule hypertrophy': 8, 
#####-------------------'Old Anterior Myocardic Infraction': 30, 
#####-------------------'Others': 44, 
#####-------------------'Isochemic Changes': 88, 
#####-------------------'Old Inferior Myocardic Infraction': 30, 
#####-------------------'Sinus Tachycardy': 26, 
#####-------------------'Left Bundle Branch Block': 18, 
#####-------------------'Atrial Fibrillation or Flutter': 10}





#####-------------------For applying machine learning, there should be sufficient amount of instances
#####-------------------after running the current program, we found out that some of the anomalies have 
#####-------------------been identified only 3 to 6 times which might fail the learning algorithm, as a
#####-------------------result, some of the data is needed to be erased in order to proceed to next level
#####-------------------of data mining. For example: Normal undiseased data have identified to be present
#####-------------------489 times while there is only 3 instances to identify supraventricular premature
#####-------------------contraction. Therefore, some of the anomalies have to be circumvented in order to
#####-------------------increase accuracy of the learning algorithm. The threshold value of detection is 
#####-------------------selected to be 26. That means if an aberration have at least 26 instances in the 
#####-------------------dataset it will proceed to the next level.


j=[]
for i in feature_dict.keys():
	if feature_dict[i]<26:
		j.append(i)
for i in j:
	del feature_dict[i]  ###---deleting feature instances if found number of instances less than the threshold

print(feature_dict)
 

#####-------------------The above print command will print the following:
#####-------------------{'Sinus Bradycardy': 50,
#####-------------------'Right Bundle Branch Block': 100, 
#####-------------------'Normal': 489, 
#####-------------------'Old Anterior Myocardic Infraction': 30, 
#####-------------------'Others': 44, 
#####-------------------'Isochemic Changes': 88, 
#####-------------------'Old Inferior Myocardic Infraction': 30, 
#####-------------------'Sinus Tachycardy': 26}




#####-------------------Finding out and deleting instances with missing values.Also
#####-------------------while looking at the feature values within the dataset it was
#####-------------------observed that some of the feature values did not change too much
#####-------------------or took the same value for a huge amount of time. So all of those 
#####-------------------data have to identified and scraped off the file in order to apply
#####-------------------and fit the data properly in machine learning algorithm. I set a 
#####-------------------threshold of 250, that means if a feature doesn't change its value
#####-------------------in at least 250 recorded instances it will not be allowed to be in
#####-------------------the final dataset


row=0;
existdata=[]
feature_number=[]
feature_number_dict={}
for d in c:
	row+=1
	e=d.split(",")
	f=int(e[279])
	column=0
	flag=0
	#print(e[13])
	for j in e:
		column+=1
		if j!="?" and row!=452:
			if column in feature_number_dict.keys() and float(j)!=0:
				feature_number_dict[column]+=1
			elif float(j)!=0:
				feature_number_dict[column]=1
		elif j=="?" and column!=14:
			flag=1
	if flag==0 and (anomaly_name[f-1] in feature_dict.keys()):
		existdata.append(row)
#print(existdata) #####-------------------uncomment this command to print the existing feature data  

#####-------------------The above print will print the number of instances without any missing values

remaining_feature=[]
nodata=[13]
for i in range(1,280):
	if i==13:
		continue
	if i in feature_number_dict.keys():
		if feature_number_dict[i]<250:
			nodata.append(i)
		else:
			remaining_feature.append(i)
#print(nodata)   #####-------------------uncomment this command to print the truncated feature number
#print(len(nodata))
print(remaining_feature)   #####-------------------uncomment this to print the remaining feature number

#####-------------------The above print command will print the number of features
#####-------------------do not change very much at all and should be stripped off.




#####-------------------The next code snippet will carry out the task of creating a new
#####-------------------text file integrating all the feature values and truncating all
#####-------------------missing values and mundanely repetitive feature values

row=0
for d in c:
	row+=1
	if row in existdata:
		e=d.split(",")
		column=0
		for k in e:
			column+=1
			if column not in nodata:
				if column==280 and int(e[column-1])==10:
					newfile.write(str(7))
				elif column==280 and int(e[column-1])==16:
					newfile.write(str(8))
				elif column==280:
					newfile.write(e[column-1])
				else:
					newfile.write(e[column-1]+",")
		newfile.write("\n")
newfile.close()
datafile.close()

#####-------------------new file named newdata.txt is generated consisting 
#####-------------------the selected raw data that will be further processed for 
#####-------------------application in machine learning algorithm

#####-------------------The remaining features are as follows
#####-------------------age,gender,height,weight,QRS duration
#####-------------------P-R interval,Q-T interval,T interval,
#####-------------------P interval, vector angles of QRS, T, P,
#####-------------------Heart rate, Average width of R wave, S wave
#####-------------------Number of intrinisic deflections,various channel 
#####-------------------ECG values,amplitude of JJ wave, R wave, S wave,
#####-------------------P wave, T wave, QRSA shape value, QRSTA and various 
#####-------------------channel values for ECG monitoring


#####-------------------1:Normal, 2: Isochemic changes, 3:old Anterior Myocardiac 
#####-------------------infraction, 4: old inferior myocardiac infraction, 5: sinus
#####-------------------tachycardy, 6: sinus bradycardy, 7: Right bundle branch
#####-------------------block, 8.others
