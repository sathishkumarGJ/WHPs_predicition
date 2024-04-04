#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template,make_response
import pandas as pd
import joblib
import csv
from modlamp.descriptors import GlobalDescriptor
from propy.PyPro import GetProDes 



app= Flask(__name__)

antimicrobial_model = joblib.load('AMP_ALLFRMs_split_03_XG_jlib') 
antiinflammatory_model = joblib.load('AIP_ALLFRMs_split_03_XG_jlib')


#Defining AAs groups
CHTgroups={1:["A",'G','Q','N','P','S','T','V'],2:['W','Y','F'],3:['D','E'],4:['H','R','K'],5:['C','I','L','M']} # i need to change Group to CHT 

def check_seq (sequence):
    AAletter= ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    for k in sequence:
        if k not in AAletter:
            return "invalid"
    return "valid"
def _Str2Num(proteinsequence):
    """
	substitute protein sequences by their corresponding group numbers.
	res = Group No.
	counter = Length of peptide
	proteinsequence = Peptide sequence
    """
    repmat={}
    counter = len(proteinsequence)-2
    
    for i in CHTgroups:
        for j in CHTgroups[i]:
            repmat[j]=i
            
			
    res=proteinsequence
    
    for i in repmat:
        res=res.replace(i,str(repmat[i]))
    
    return res, counter, proteinsequence

def Calculate_CHT(proteinsequence):

    proteinnum, counter, proteinseq=_Str2Num(proteinsequence)     #subst. protein seq. by group numbers obtained from k-means clustering technique for r-CTF (ww)
    """
        Caculate the no of times 111, 112,... occurs
    """
    res={}
    for i in range(1,6):
        for j in range(1,6):
            for k in range(1,6):
                temp=str(i)+str(j)+str(k)
                res[temp]=sum(1 for i in range(len(proteinnum)) if proteinnum.startswith(temp, i)) / counter
                
    
    return res

    
    return res
#defining functions that we will be needing
def other_desc(seq, dict,excl=False):
    add=0
    count=0
    for i in seq.strip():
        add+= dict[i]
        count+=1
    if excl:
        return add
    else:
        return add/count

def hydrophobicity(seq):
    dict= {"A":-0.17, "R":-0.81, "N":-0.42, "D":-1.23, "C":0.24, "Q":-0.58, "E":-2.02, "G":-0.01, "H":-0.96, "I":0.31,
           "L":0.56, "K":-0.99, "M":0.23, "F":1.13, "P":-0.45, "S":-0.13, "T":-0.14, "W":1.85, "Y":0.94, "V": -0.07}
    add=0
    count=0
    for i in seq.strip():
        add+= dict[i]
        count+=1
    return add/count
def get_dict(code):
    return GetProDes(" ").GetAAindex1(name=code)

def Calculate_PCP(seq):
    desc= GlobalDescriptor(seq)
    desc.calculate_all(amide=False)
    gobal_desc=desc.descriptor
    gobal_desc=np.delete(gobal_desc, [1])  # to remove molecular weight from global descriptors
    gobal_desc=gobal_desc.tolist()
    gobal_desc.append(hydrophobicity(seq))
    
    val= {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2,'I': 4.5, 'L': 3.8,'K': -3.9,'M': 1.9,
 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7,'W': -0.9,'Y': -1.3,'V': 4.2}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 0.0,'R': 2.45,'N': 0.0, 'D': 0.0, 'C': 0.0, 'E': 1.27, 'Q': 1.25, 'G': 0.0, 'H': 1.45, 'I': 0.0, 'L': 0.0, 'K': 3.67,
 'M': 0.0, 'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 6.93, 'Y': 5.06, 'V': 0.0}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 0.67, 'R': -2.1, 'N': -0.6, 'D': -1.2, 'C': 0.38, 'E': -0.76, 'Q': -0.22, 'G': 0.0, 'H': 0.64, 'I': 1.9, 'L': 1.9, 'K': -0.57,
 'M': 2.4, 'F': 2.3, 'P': 1.2, 'S': 0.01, 'T': 0.52, 'W': 2.6, 'Y': 1.6, 'V': 1.5}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 0.52,'R': 0.68, 'N': 0.76, 'D': 0.76, 'C': 0.62, 'E': 0.68, 'Q': 0.68, 'G': 0.0, 'H': 0.7, 'I': 1.02, 'L': 0.98, 'K': 0.68, 'M': 0.78,
 'F': 0.7,'P': 0.36, 'S': 0.53, 'T': 0.5, 'W': 0.7, 'Y': 0.7, 'V': 0.76}
    gobal_desc.append(other_desc(seq,val))
    
    val={'A': 31.0, 'R': 124.0, 'N': 56.0, 'D': 54.0, 'C': 55.0, 'E': 83.0, 'Q': 85.0, 'G': 3.0, 'H': 96.0, 'I': 111.0, 'L': 111.0, 'K': 119.0, 'M': 105.0,
 'F': 132.0, 'P': 32.5, 'S': 32.0, 'T': 61.0, 'W': 170.0, 'Y': 136.0, 'V': 84.0}
    gobal_desc.append(other_desc(seq,val,True))
    
    val= {'A': 27.5, 'R': 105.0, 'N': 58.7, 'D': 40.0, 'C': 44.6, 'E': 62.0, 'Q': 80.7, 'G': 0.0, 'H': 79.0, 'I': 93.5, 'L': 93.5, 'K': 100.0,
 'M': 94.1,'F': 115.5, 'P': 41.9, 'S': 29.3, 'T': 51.3, 'W': 145.5, 'Y': 117.3, 'V': 71.5}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5, 'E': 12.3, 'Q': 10.5, 'G': 9.0, 'H': 10.4, 'I': 5.2, 'L': 4.9, 'K': 11.3,
 'M': 5.7, 'F': 5.2, 'P': 8.0, 'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 0.0, 'R': 4.0, 'N': 2.0, 'D': 1.0, 'C': 0.0, 'E': 1.0, 'Q': 2.0, 'G': 0.0, 'H': 1.0, 'I': 0.0, 'L': 0.0, 'K': 2.0, 'M': 0.0,
 'F': 0.0, 'P': 0.0, 'S': 1.0, 'T': 1.0, 'W': 1.0, 'Y': 1.0, 'V': 0.0}
    gobal_desc.append(other_desc(seq,val, True))
    
    val= {'A': 1.29, 'R': 0.83, 'N': 0.77, 'D': 1.0, 'C': 0.94, 'E': 1.54, 'Q': 1.1, 'G': 0.72, 'H': 1.29, 'I': 0.94, 'L': 1.23, 'K': 1.23,
 'M': 1.23, 'F': 1.23, 'P': 0.7, 'S': 0.78, 'T': 0.87, 'W': 1.06, 'Y': 0.63, 'V': 0.97}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 0.96, 'R': 0.67, 'N': 0.72, 'D': 0.9, 'C': 1.13, 'E': 0.33, 'Q': 1.18, 'G': 0.9, 'H': 0.87, 'I': 1.54, 'L': 1.26, 'K': 0.81,
 'M': 1.29, 'F': 1.37, 'P': 0.75, 'S': 0.77, 'T': 1.23, 'W': 1.13, 'Y': 1.07, 'V': 1.41}
    gobal_desc.append(other_desc(seq,val))
    
    val= {'A': 0.72, 'R': 1.33, 'N': 1.38, 'D': 1.04, 'C': 1.01, 'E': 0.75, 'Q': 0.81, 'G': 1.35, 'H': 0.76, 'I': 0.8, 'L': 0.63, 'K': 0.84,
 'M': 0.62, 'F': 0.58, 'P': 1.43, 'S': 1.34, 'T': 1.03, 'W': 0.87, 'Y': 1.35, 'V': 0.83}
    gobal_desc.append(other_desc(seq,val))
    
    #scaling the new features with the scaler made using training data
    #scl= joblib.load('PCD_min_max_scaled_final_jlib') 
    #gobal_desc=scl.transform([gobal_desc])
    return gobal_desc

def antimicrobial_features(sequence):
     #Amino Acid Symbols
    AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
 
    final_features = []
    for i in AALetter:
        counter=0
        for j in sequence:
            if j==i:
                counter = counter+1    
        final_features.append(counter / len(sequence))

    physio = Calculate_PCP(sequence)
#scaling the new features with the scaler made using training data
    scl= joblib.load('AMP01_PCD_min_max_scaled_final_jlib')
    physio=scl.transform([physio])
    physio= physio.tolist()
    CHT = Calculate_CHT(sequence)
    final_features= physio[0] + final_features + list(CHT.values()) 
    return final_features

def antiinflammatory_features(sequence):
     #Amino Acid Symbols
    AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
 
    final_features = []
    for i in AALetter:
        counter=0
        for j in sequence:
            if j==i:
                counter = counter+1    
        final_features.append(counter / len(sequence))

    physio = Calculate_PCP(sequence)
#scaling the new features with the scaler made using training data
    scl= joblib.load('AIP_PCD_min_max_scaled_final_jlib')
    physio=scl.transform([physio])
    physio= physio.tolist()
    CHT = Calculate_CHT(sequence)
    final_features= physio[0] + final_features + list(CHT.values()) 
    return final_features

def model_predict_antimicrobial(sequence):
    final_features= antimicrobial_features(sequence)
    return antimicrobial_model.predict([final_features])

def model_predict_antiinflammatory(sequence):
    final_features= antiinflammatory_features(sequence)
    return antiinflammatory_model.predict([final_features])
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sequence = request.form.get("sequence")
    f= request.files["csvfile"]
    
    if f.filename=='':  
        if check_seq(sequence)=="invalid":
            return render_template('index.html', sequence_text = "Result for sequence: "+ sequence,error_text = "Heads Up!! Enter valid sequence. Please return and reload the page")
       
        
        if model_predict_antimicrobial(sequence)==[1]:
            
            if model_predict_antiinflammatory(sequence)==[1]:
                antiinflammatory= ""
            else: 
                antiinflammatory= "not "
                
        else:
            
            return render_template('index.html', sequence_text = "Result for sequence: "+ sequence,prediction_text='It is Not-Wound Healing peptide' )
        
        return render_template('index.html', sequence_text="Result for sequence: " + sequence,prediction_text='It is wound healing peptide')

    else:
        df=[]
        csvfile= pd.read_csv(request.files.get('csvfile'), header= None)
        for i in range(len(csvfile)):
            new=[]
            new.append(csvfile.iloc[i,0])
            
            if check_seq(csvfile.iloc[i,0])=="invalid": 
                new.append("Invalid Sequence")
            elif model_predict_antimicrobial(csvfile.iloc[i,0])==0:
                new.append("Not-Wound Healing Peptide")
            else:
                new.append("Anti-Antimicrobial")
                if model_predict_antiinflammatory(csvfile.iloc[i,0])==1:
                    new.append("Anti-inflammatory")
                else: 
                    new.append("Not-Wound Healing Peptide")
                    
                    
            df.append(new)
        newdf = pd.DataFrame(df, columns = ['Sequence', 'Anti-microbial', 'Anti-inflammatory'])
        resp = make_response(newdf.to_csv(index= False))
        resp.headers["Content-Disposition"] = "attachment; filename=predicted.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
                          


if __name__ == "__main__":
    app.run(debug=False)


# In[ ]:




