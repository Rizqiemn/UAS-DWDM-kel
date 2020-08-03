import pandas as pd
from sklearn import preprocessing

#Tautan https://www.worldometers.info/coronavirus/?
#Data telah disunting ulang agar dapat dioperasikan kedalam naive bayes.
#dengan tambahan rating untuk mengaktifkan klasifikasinya.
data = pd.read_excel("sarscov19.xlsx",sheet_name='Lembar1')


#Missing values
data['TOTAL KASUS']=data['TOTAL KASUS'].fillna(0)
data['TOTAL KEMATIAN']=data['TOTAL KEMATIAN'].fillna(0)
data['DI NYATAKAN SEMBUH']=data['DI NYATAKAN SEMBUH'].fillna(0)

X = data[['TOTAL KASUS','TOTAL KEMATIAN','DI NYATAKAN SEMBUH']]
y=data['RATING']


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)       # training model dengan method fit()
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
akurasi = accuracy_score(ytest, y_model) 
print(akurasi)