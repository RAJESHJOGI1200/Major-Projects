from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from mobileapp.models import Register
from django.contrib import messages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Create your views here.

def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')


Registration = 'register.html'
def register(request):
    if request.method == 'POST':
        Name = request.POST['Name']
        email = request.POST['email']
        password = request.POST['password']
        conpassword = request.POST['conpassword']
        age = request.POST['Age']
        contact = request.POST['contact']

        print(Name, email, password, conpassword, age, contact)
        if password == conpassword:
            user = User(email=email, password=password)
            # user.save()
            return render(request, 'login.html')
        else:
            msg = 'Register failed!!'
            return render(request, Registration,{msg:msg})

    return render(request, Registration)

# Login Page 
def login(request):
    if request.method == 'POST':
        lemail = request.POST['email']
        lpassword = request.POST['password']

        d = User.objects.filter(email=lemail, password=lpassword).exists()
        print(d)
        return redirect(userhome)
    else:
        return render(request, 'login.html')

def userhome(request):
    return render(request,'userhome.html')

def view(request):
    global df,msg
    if request.method=='POST':
        g = int(request.POST['num'])
        df = pd.read_csv('20230329093832Mobile-Addiction-.csv')
        col = df.head(g).to_html()
        return render(request,'view.html',{'table':col})
    return render(request,'view.html')


def module(request):
        df = pd.read_csv('20230329093832Mobile-Addiction-.csv')
        df.drop('Timestamp',inplace=True,axis=1)
        df.drop('Full Name :',inplace=True,axis=1)
        df.rename(columns={'Gender :':'Gender'},inplace=True)
        
        for column in df.columns:
            # Find the most frequent value for the current column
            most_frequent_value = df[column].mode()[0]
            # Replace missing values in the current column with the most frequent value
            df[column].fillna(most_frequent_value, inplace=True)
    
        df['For how long do you use your phone for playing games?'] = df['For how long do you use your phone for playing games?'].replace({'>2 hours': 1, '<2 hours': 0})
        encoder = LabelEncoder()
        for column in  df.columns:
            df[column] = encoder.fit_transform(df[column])

        
        X = df.drop('whether you are addicted to phone?', axis=1)
        y = df['whether you are addicted to phone?'] 

        maybe_instance = df[df['whether you are addicted to phone?'] == 0]
        df_augmented = pd.concat([df, maybe_instance]*5, ignore_index=True)  

        X = df_augmented.drop('whether you are addicted to phone?', axis=1)
        y = df_augmented['whether you are addicted to phone?']   

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        if request.method=='POST':
            model = request.POST['algo']

            if model == "1":
                re = RandomForestClassifier(n_estimators=50, max_depth=5, max_features='sqrt', random_state=42)
                re.fit(X_train_resampled, y_train_resampled)
                re_pred = re.predict(X_test)
                ac = accuracy_score(y_test, re_pred)
                precision = precision_score(y_test, re_pred, average='weighted')
                recall = recall_score(y_test, re_pred, average='weighted')
                f1 = f1_score(y_test, re_pred, average='weighted')
                acc_percent = round(ac*100,2)
                msg = f'Accuracy of RandomForest: {acc_percent}%\n'
                msg += f'Precision: {precision}\n'
                msg += f'Recall: {recall}\n'
                msg += f'F1-score: {f1}\n'
                return render(request, 'module.html', {'msg': msg})
            
            elif model == "2":
                de = DecisionTreeClassifier(max_depth=5, random_state=42)
                de.fit(X_train_resampled, y_train_resampled)
                de_pred = de.predict(X_test)
                ac1 = accuracy_score(y_test, de_pred)
                precision = precision_score(y_test, de_pred, average='weighted')
                recall = recall_score(y_test, de_pred, average='weighted')
                f1 = f1_score(y_test, de_pred, average='weighted')
                acc_percent = round(ac1*100,2)
                msg = f'Accuracy of Decision Tree: {acc_percent}%\n'
                msg += f'Precision: {precision}\n'
                msg += f'Recall: {recall}\n'
                msg += f'F1-score: {f1}\n'
                return render(request, 'module.html', {'msg': msg})
            
            elif model == "3":
                le = LogisticRegression(random_state=42, max_iter=1000)
                le.fit(X_train_resampled, y_train_resampled)
                le_pred = le.predict(X_test)
                ac2 = accuracy_score(y_test, le_pred)
                precision = precision_score(y_test, le_pred, average='weighted')
                recall = recall_score(y_test, le_pred, average='weighted')
                f1 = f1_score(y_test, le_pred, average='weighted')
                acc_percent = round(ac2*100,2)
                msg = f'Accuracy of Logistic Regression: {acc_percent}%\n'
                msg += f'Precision: {precision}\n'
                msg += f'Recall: {recall}\n'
                msg += f'F1-score: {f1}\n'
                return render(request, 'module.html', {'msg': msg})

        return render(request, 'module.html')




def prediction(request):
    df = pd.read_csv('20230329093832Mobile-Addiction-.csv')
    df.drop('Timestamp', inplace=True, axis=1)
    df.drop('Full Name :', inplace=True, axis=1)
    df.rename(columns={'Gender :': 'Gender'}, inplace=True)

    # Drop the "Maybe" class
    df = df[df['whether you are addicted to phone?'] != 'Maybe']

    for column in df.columns:
        # Find the most frequent value for the current column
        most_frequent_value = df[column].mode()[0]
        # Replace missing values in the current column with the most frequent value
        df[column].fillna(most_frequent_value, inplace=True)

    df['For how long do you use your phone for playing games?'] = df['For how long do you use your phone for playing games?'].replace({'>2 hours': 1, '<2 hours': 0})
    encoder = LabelEncoder()
    for column in df.columns:
        df[column] = encoder.fit_transform(df[column])

    X = df.drop('whether you are addicted to phone?', axis=1)
    y = df['whether you are addicted to phone?']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    if request.method == 'POST':
        b = float(request.POST['f2'])
        c = float(request.POST['f3'])
        d = float(request.POST['f4'])
        e = float(request.POST['f5'])
        f = float(request.POST['f6'])
        g = float(request.POST['f7'])
        h = float(request.POST['f8'])
        i = float(request.POST['f9'])
        j = float(request.POST['f10'])
        k = float(request.POST['f11'])
        l = float(request.POST['f12'])
        m = float(request.POST['f13'])
        n = float(request.POST['f14'])
        o = float(request.POST['f15'])
        p = float(request.POST['f16'])
        q = float(request.POST['f17'])
        r = float(request.POST['f18'])
        s = float(request.POST['f19'])

        data = [[b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s]]
        re = RandomForestClassifier()
        re.fit(X_train_resampled, y_train_resampled)
        pred = re.predict(data)
        if pred == 0:
            msg = 'User not Addicted'
        elif pred == 1:
            msg = 'User Addicted'
        elif pred == 2:
            msg = 'User may be Addicted'

        return render(request, 'prediction.html', {'msg': msg})
    return render(request, 'prediction.html')

def graph(request):
    return render(request,'graph.html')

