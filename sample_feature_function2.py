#from Skills import processing_resumes
import nltk
import os
import threading
import time
import pycrfsuite
from sklearn.model_selection import train_test_split
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
# import pandas as pd
# import numpy as np
# from collections import Counter
# from nltk.tag.stanford import StanfordNERTagger
import re
from skills_stopword import Skill_stop
stopwords=Skill_stop()
import zipfile
import os
import docx2txt
from zipfile import BadZipFile
my_text=[]
import glob
from itertools import *
def feature_functions(resume_path):
    final_skills=[]
    count=0
    #for resume_path in list_resume:
    if resume_path.endswith('.docx'):
        names=resume_path.split("\\")[4]
        count+=1
        my_text = docx2txt.process(resume_path)
        extracted_doctext = my_text
        my_text = extracted_doctext
        nstr = re.sub(r'[?|$|.|!|\t|:|<|>|"|,/|(|)|.|,|]', r'', my_text)
        text = [word for word in nstr.split() if word not in stopwords]
        #filtered_text = (' '.join(text))
        #txt = filtered_text.replace("-", "-")
        # remove
        remove_number = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', my_text)

        # Email_id EXTRACTION
        email = re.findall('\S+@\S+', my_text)
        email = ','.join(email)
        email_ = email
        email = email.split(',')
        email = list(set(email))
        email = ','.join(email)

        # Phone_number EXTRACTION
        remove_year_format = re.sub('\d{4} - \d{4}', '', my_text)
        remove_year_format = re.sub('\d{2}-\d{2}-\d{4} - \d{2}-\d{2}-\d{4}', '', remove_year_format)
        remove_year_format = re.sub('\d{4} -\d{4}', '', remove_year_format)
        remove_year_format = re.sub('\d{4}- \d{4}', '', remove_year_format)
        remove_year_format = re.sub('\d{4} - \d{2} - \d{2} - \d{4} - \d{2} - \d{2}', ',', remove_year_format)
        splited_text_phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', remove_year_format)
        splited_text_phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', remove_year_format)
        splited_text_phone = ' '.join(splited_text_phone)

        # Experience EXTRACTION
        #splited_year = re.findall('\d{4} - \d{4}', filtered_text)
        splited_year = re.findall('\d{4} - \d{4}', my_text)
        splited_year = ' '.join(splited_year)

        #NAME=extracted_doctext.split()[0]+' '+extracted_doctext.split()[1]
        NAME1 = extracted_doctext.split()[0]
        NAME2=  extracted_doctext.split()[1]
        # #SKILLS
        text = [word for word in extracted_doctext.split( ) if word not in stopwords]
        text = ' '.join(text)
        remove_number_ = ' '.join(remove_number)
        data = text.replace(NAME1, '')
        data = data.replace(NAME2, '')
        data = data.replace(email_, '')
        data = data.replace(splited_text_phone, '')
        data = data.replace(splited_year, '')
        data = data.replace(remove_number_, '')
        data=nltk.word_tokenize(data)
        data=[word for word in data if word not in stopwords]
        pos_tagged = nltk.pos_tag(data)
        group_tag = []
        for key_tag,value_tag in pos_tagged:
            if value_tag == 'NNP':
                group_tag.append(key_tag.lower())
        text = [word for word in group_tag if word not in stopwords]
        text = set(text)
        a = []
        skill_labels=[]
        for s in text:
            #value=(s,'skill')
            #a.append(value)
            a.append(s)
        txt_tokenized=nltk.word_tokenize(my_text)
        resume_converted=[]
        for i in txt_tokenized:
            if i.lower() in a:
                resume_converted.append(i)
            else:
                resume_converted.append('0')
        print(resume_converted)
        return resume_converted,txt_tokenized,names

if __name__ == "__main__":
    X_train = []
    Y_train = []
    start_time=time.time()
    check_list=[]
    def converted_resume_1():
        list_resume = glob.glob("D:\\vibhav\\resume2\\USM_RECRUITMENT\\input_resume\\6000resumes (2)\\6000resumes\\6000resumes\\resumes_used_fro training\\New folder (2)\\*.docx")
        count=0
        actual_count=0
        for resume_path in list_resume:
            actual_count+=1
            try:
                #result,txt_stopword_removed,names=feature_functions(resume_path)
                result,txt_tokenized,names=feature_functions(resume_path)
                #check_list.append(names)
                count+=1
                Y_train.append(result)
                X_train.append(txt_tokenized)
            except (KeyError,BadZipFile,TypeError):
                continue
    converted_resume_1()
    print(X_train)
    print(Y_train)
    trainer= pycrfsuite.Trainer(verbose=True)
    for xseq,yseq in zip(X_train,Y_train):
		trainer.append(xseq,yseq)
    trainer.train('skill_trained_crf_trained_changed_5214.model')
