from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import sys, os, datetime, pandas as pd, joblib


class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi("ui\\main.ui", self)

        self.btnEWSRaw = self.findChild(QPushButton, "btnEWSRaw")
        self.btnERPRaw = self.findChild(QPushButton, "btnERPRaw")
        self.txtEWSRaw = self.findChild(QLineEdit, "txtEWSRaw")
        self.txtERPRaw = self.findChild(QLineEdit, "txtERPRaw")

        self.btnModel = self.findChild(QPushButton, "btnModel")
        self.txtModel = self.findChild(QLineEdit, "txtModel")

        self.btnStartRaw = self.findChild(QPushButton, "btnStartRaw")
        self.btnStartModel = self.findChild(QPushButton, "btnStartModel")

        self.popup = QMessageBox

        self.btnModel.clicked.connect(self.getModel)
        self.btnEWSRaw.clicked.connect(self.getEWSRaw)
        self.btnERPRaw.clicked.connect(self.getERPRaw)
        self.btnStartModel.clicked.connect(self.createModel)
        self.btnStartRaw.clicked.connect(self.predictData)

        self.trainingData = 'training\\trainingData.xlsx'
        self.mergeRaw = ''
        
        self.setWindowTitle('Job Abandonment')
        #self.setWindowIcon(QIcon('icon\\win.ico'))
        self.show()

        #self.modelLeaveReason()

    #--------------------PREDICTION MODEL------------------------

    def getModel(self):
        file_path, _= QFileDialog.getOpenFileName(None, "", QDir.homePath() + "/Desktop")
        self.txtModel.setText(file_path)

    def createModel(self):
        popup = self.popup.question(self, "Job Abandonment", "Are you sure you want to add new data and update the model?", self.popup.Yes | self.popup.No)
        if popup == self.popup.Yes:
            modelPath = self.txtModel.text()
            if os.path.isfile(modelPath):
                if self.headerMatch(modelPath):
                    self.backUpData()
                    self.appendData()
                    self.modelLeaveReason()
                    self.popup.information(self, "Job Abandonment", "New training data successfully added.")
                else:
                    self.popup.information(self, "Job Abandonment", f"File doesn't matched!")

    def headerMatch(self, file_path):
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)
            # Get the headers of the new data
            dfHeader = df.columns.tolist()
            dfTraining = pd.read_excel(self.trainingData)  # replace with your training data path
            # Get the headers of the training data
            dfTrainingHeader = dfTraining.columns.tolist()
            # Compare the headers
            if dfHeader == dfTrainingHeader:
                return True
        except Exception as e:
            self.popup.information(self, "Job Abandonment", f"An error occurred: {e}")


    def backUpData(self):
        today = datetime.date.today()
        backUpFile = f'backup\\training_data_backup_{today}.xlsx'
        dfTrainingData = pd.read_excel(self.trainingData)
        dfTrainingData.to_excel(backUpFile, index=False)

    def appendData(self):
        filepath = self.txtModel.text()
        dfNewData = pd.read_excel(filepath)
        dfTrainingData = pd.read_excel(self.trainingData)  # replace with your training data path

        # Append new data to training data
        dfTrainingData = pd.concat([dfTrainingData, dfNewData])

        columns_to_check = ['EWSID','EmpID','EmpName','Person Type','Original  Tenure Bracket','Job','Location','Line of Business','InterventionStatus','RAG','LeavingReason','AttritionType']

        # Remove duplicates based on the specified columns
        dfTrainingData = dfTrainingData.drop_duplicates(subset=columns_to_check)

        # Write the updated training data back to the file
        dfTrainingData.to_excel(self.trainingData, index=False)  # replace with your training data path


    def modelLeaveReason(self):
        df = pd.read_excel(self.trainingData)

        print(df)

        df.columns = df.columns.astype(str)
        df = df.drop(['EWSID','EmpID','EmpName','Person Type','Job','AttritionType'],axis=1)

        encoder = OneHotEncoder(handle_unknown='ignore')
        catColumns = ['Original  Tenure Bracket','InterventionStatus','Location','Line of Business','RAG']

        for col in catColumns:
            df[col] = df[col].astype(str)

        encodedCol = pd.DataFrame(encoder.fit_transform(df[catColumns]).toarray())
        encodedCol.columns = encodedCol.columns.astype(str)

        df = df.drop(catColumns, axis=1)
        df = df.join(encodedCol)

        x = df.drop('LeavingReason', axis=1)
        y = df['LeavingReason']

        xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier()

        clf.fit(xTrain, yTrain)

        yPred = clf.predict(xTest)

        accuracy = accuracy_score(yTest, yPred)
        print(accuracy)

        importances = clf.feature_importances_

        joblib.dump(clf, 'training\\model.pkl')
        joblib.dump(encoder, 'training\\encoder.pkl')


        #['EWSID','EmpID','EmpName','Person Type','Original  Tenure Bracket','Job','Location','Line of Business','InterventionStatus','RAG','LeavingReason','AttritionType']

    #------------------------RAW DATA---------------------------

    def getEWSRaw(self):
        file_path, _= QFileDialog.getOpenFileName(None, "", QDir.homePath() + "/Desktop")
        self.txtEWSRaw.setText(file_path)

    def getERPRaw(self):
        file_path, _= QFileDialog.getOpenFileName(None, "", QDir.homePath() + "/Desktop")
        self.txtERPRaw.setText(file_path)

    def predictData(self):
        self.mergeData()
        self.predictLeaveReason()

    def mergeData(self):
        EWSData = pd.read_excel(self.txtEWSRaw.text())
        EWSData =EWSData[EWSData['LeavingReason'].isna()]
        EWSData = EWSData[['EWSID','EmpID','EmpName','InterventionStatus','SupervisorName','Client','RAG']]
        
        ERPData = pd.read_excel(self.txtERPRaw.text())
        ERPData = ERPData[['Employee Number','Person Type','Original  Tenure Bracket','Job','Location','Line of Business']]
        ERPData = ERPData.rename(columns={'Employee Number':'EmpID'})

        self.mergeRaw = pd.merge(EWSData,ERPData,on='EmpID',how='left')

    def predictLeaveReason(self):
        
        # Load the model
        clf = joblib.load('training\\model.pkl')

        # Load the encoder
        encoder = joblib.load('training\\encoder.pkl')

        # Load your data from an Excel file
        input_data = self.mergeRaw
        original_data = input_data.copy()  # Keep a copy of the original data
        input_data = input_data.drop(['EWSID','EmpID','EmpName','SupervisorName','Client','Person Type','Job'], axis=1)

        # Convert categorical columns to string type
        categorical_columns = ['Original  Tenure Bracket','InterventionStatus','Location','Line of Business','RAG']
        for col in categorical_columns:
            input_data[col] = input_data[col].astype(str)

        # Apply the encoder to the categorical columns
        encoded_columns = pd.DataFrame(encoder.transform(input_data[categorical_columns]).toarray())

        # Convert the encoded column names to strings
        encoded_columns.columns = encoded_columns.columns.astype(str)

        # Drop the old categorical columns from original df
        input_data = input_data.drop(categorical_columns, axis=1)

        # Join the encoded df with the original df
        input_data = input_data.join(encoded_columns)

        # Predict on the new data
        y_input_pred = clf.predict(input_data)

        # Add the predicted 'Criticality' to the original data
        original_data['LeavingReason'] = y_input_pred

        # Save the DataFrame to an Excel file
        self.output = os.path.dirname(self.txtERPRaw.text()) + "/Output_" + datetime.datetime.now().strftime("%Y_%m_%d") + ".xlsx"
        original_data.to_excel(self.output, index=False)


app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()        



