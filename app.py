#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_sim():
    data = pd.read_csv('naukri_recommend.csv')
    data.reset_index(inplace=True)
    data['Search']=data['Search'].str.lower()
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['Clean_data'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix,count_matrix)
    return data,sim

def rcmd(value):
    value = value.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the job_roles is in our database or not
    if value not in data['Search'].unique():
        return('This job_roles is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the job_role in the dataframe
        i = data.loc[data['Search']==value].index[0]

        # fetching the row containing similarity scores of the job
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- job_roles scores
        # not taking the first index since it is the same job_roles
        lst = lst[1:11]

        # making an empty list that will containg all 10 job_roles recommendations
        title_indices=[i[0] for i in lst]
    
        role=data['ROLES'].iloc[title_indices]
        comp=data['Companies'].iloc[title_indices]
        loc=data['Location'].iloc[title_indices]
        sal=data['Salary'].iloc[title_indices]
        skills=data['Skills'].iloc[title_indices]
        job_titles=data['Title_new'].iloc[title_indices]
        # input values into dataframe
        rec_df=pd.DataFrame({'Given Job_roles':role,'Company':comp,'location':loc,'salary':sal,'Skills':skills,'Title':job_titles})
        rec_df.reset_index(inplace=True)
        rec_df.drop("index",axis=1,inplace=True)
        
        return rec_df
    
    
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    Job_Roles = request.args.get('Job_Roles')
    r = rcmd(Job_Roles)
    #movie = movie.upper()
    if type(r)== type('string'):
        return render_template('recommend.html',Job_Roles=Job_Roles,r=r,t='s')
    else:
        return render_template('recommend.html',Job_Roles=Job_Roles,r=r,t=[r.to_html(classes='data',header="true")])
        #return render_template('jobs.html',Job_Roles=Job_Roles,r=r,t='text_file')
        



if __name__ == '__main__':
    app.run()


# In[ ]:





# In[ ]:




