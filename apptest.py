import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from fonctions import *
#import variables



		

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1,3,1])
col1.write("")
col2.title('GroundTruth')
col3.write("")

st.sidebar.title('Questions Selector')


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	correl=pickle.load( open( "correlations.p", "rb" ) )
	questions=pd.read_csv('questions.csv',index_col=0)
	#questions.drop('Idquest',axis=0,inplace=True)
	#questions.drop([i for i in questions if 'Unnamed' in i],axis=1,inplace=True)
	#questions=questions.T
	#questions.columns=['parent', 'type', 'Treatment', 'Other','question']
	codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
	continues=pickle.load( open( "cont_feat.p", "rb" ) )
	cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
	dummy_cols=pickle.load( open( "dummy.p", "rb" ) )
	
	
	return data,correl,questions,codes,continues,cat_cols,dummy_cols


#Récup des data avec les variables nouvelles
data,correl,questions,codes,continues,cat_cols,dummy_cols=load_data()
#st.write(continues)
######################faudra surement aussi récupérer d'autres trucs sur les types de données des int_cat et int_cat_desc############################
#st.write('categorical:',cat_cols)
#st.write(correl)
st.write(questions)
st.write(codes)

def main():
	L=[]
	graphs=pd.read_csv('graphs.csv',index_col=0,sep='\t')
	#st.write(graphs)
	q1 = st.sidebar.selectbox('Main question:', [None]+[i for i in correl][:])
	if q1 != None:
		df=selectdf(data,correl,q1,cat_cols)
		
		# st.write(df)
		
		q2_list=correl[q1]	
		# TRAITEMENT PARTICULIER DES DONNÉES DE CAT_COLS
		
		quests1=[i for i in df.columns if q1 in i] if q1 in cat_cols else [q1]
		
		if q1 in cat_cols:	

			if q1=='usage':
				fig =px.box(df,y=[i for i in df.columns if 'usage' in i],points='all')

			else:
				cats=[' '.join(i.split(' ')[1:]) for i in quests1]
				fig = px.bar(x=cats, y=df[quests1].applymap(lambda x:1 if x==1 else 0).sum(), labels={'y':'People'})	#.applymap(lambda x:1 if x=='Yes' else 0).sum()
			col1, col2 = st.columns([1,3])
			col1.write('Donnée multiple')
			col1.write(q1)
			col1.write(quests1)
			col2.plotly_chart(fig)
			
		elif q1 in ['latitude','longitude']:
			st.map(df[['latitude','longitude']],zoom=10)
		
		else:
			st.write(questions.loc[q1]['question'])
			fig=px.histogram(df, x=q1,color_discrete_sequence=['green'])
			st.plotly_chart(fig)

		st.write(correl[q1])
#Visualisation des 6 paramètres les plus importants pour la prédiction
			
		if st.sidebar.checkbox('Do you want to generate graphs with other potential correlated questions?'):	
			
			for q2 in q2_list:
				st.write(correl[q1][q2])
				quests2=[i for i in df.columns if q2 in i] if q2 in cat_cols else [q2]
				if q2 in cat_cols:
					st.subheader(q2+': '+', '.join(quests2))			
				else:
					st.subheader(q2+': '+questions.loc[q2]['question'])
				quest=quests1+quests2
				
				# On regarde maintenant si les deux sont des données catégorielles
					
				if q1 in cat_cols and q2 in cat_cols:
					
					df3=selectdf2(q1,q2,df,cat_cols)
							
					col1, col3 = st.columns([1,1])
					df3['ones']=np.ones(len(df3))
					col1.plotly_chart(sunb(q1,q2,q1,q2,df3),use_container_width=True)
					col3.plotly_chart(sunb(q2,q1,q2,q1,df3),use_container_width=True)
					
						
				# On regarde maintenant si une des deux est catégorielle	
					
				elif q1 in cat_cols or q2 in cat_cols:
					
					if q1 in cat_cols:
						cat,autre=q1,q2
					else:
						cat,autre=q2,q1
						
					df3=selectdf2(q1,q2,df,cat_cols)
					#st.write(df3)											
					if autre in dummy_cols:						
						col1, col3 = st.columns([1,1])
						df3['ones']=np.ones(len(df3))
						col1.plotly_chart(sunb(q1,q2,q1,q2,df3),use_container_width=True)
						col3.plotly_chart(sunb(q2,q1,q2,q1,df3),use_container_width=True)
					
						
					elif autre in continues:
						if autre in ['latitude','longitude']:
							df3['county']=data['county']
							col1, col2, col3 = st.columns([1, 1, 1])
							col1.plotly_chart(scattermap(df3,cat,'Wau'))
							col2.plotly_chart(scattermap(df3,cat,'Bentiu'))
							col3.plotly_chart(scattermap(df3,cat,'Malakal'))
							if cat != 'county':
								col1,col3 = st.columns([1, 1])
								col1.plotly_chart(count2('county', cat, df3), use_container_width=True)
								col3.plotly_chart(pourcent2('county', cat, df3), use_container_width=True)
						else:						
							st.plotly_chart(box(autre,cat,autre,cat,df3))
						
					else:
						col1, col3 = st.columns([4,4])
						df3['ones']=np.ones(len(df3))
						col1.plotly_chart(sunb(q1,q2,q1,q2,df3),use_container_width=True)
						col3.plotly_chart(sunb(q2,q1,q2,q1,df3),use_container_width=True)
					
				elif q1 in continues:
					if q1 in ['latitude','longitude']:
						if q2 in ['latitude','longitude']:
							st.title ('Both coordinates')
						else:
							df['county']=data['county']
							col1, col2, col3 = st.columns([1, 1, 1])
							col1.plotly_chart(scattermap(df,q2,'Wau'))
							col2.plotly_chart(scattermap(df,q2,'Bentiu'))
							col3.plotly_chart(scattermap(df,q2,'Malakal'))
							if q2 in continues:
								st.plotly_chart(box('county', q2, q1,q2,df), use_container_width=True)
							elif q2!= 'county':
								col1, col3 = st.columns([1, 1])
								col1.plotly_chart(count2('county', q2, df), use_container_width=True)
								col3.plotly_chart(pourcent2('county', q2, df), use_container_width=True)
						#else:
						#	df['county']=data['county']
						#	st.plotly_chart(scattermap(df,q2,'Wau'))
						#	st.plotly_chart(scattermap(df,q2,'Bentiu'))
						#	st.plotly_chart(scattermap(df,q2,'Malakal'))
						
					else:					
						if q2 in continues:
							st.plotly_chart(scatter(q1,q2,q1,q2,df),use_container_width=True)
						else:
							st.plotly_chart(box(q1,q2,q1,q2,df),use_container_width=True)
											
				elif q2 in ['latitude','longitude']:
					df['county']=data['county']
					col1, col2, col3 = st.columns([1, 1, 1])
					col1.plotly_chart(scattermap(df,q1,'Wau'))
					col2.plotly_chart(scattermap(df,q1,'Bentiu'))
					col3.plotly_chart(scattermap(df,q1,'Malakal'))
					col1, col3 = st.columns([1, 1])
					if q1 != 'county':
						col1.plotly_chart(count2('county', q1, df), use_container_width=True)
						col3.plotly_chart(pourcent2('county', q1, df), use_container_width=True)
				
				elif q2 in continues:
					st.plotly_chart(box(q2,q1,q2,q1,df),use_container_width=True)
					
				elif q1 in dummy_cols:
					if q2 in dummy_cols:
						col1, col3 = st.columns([4,4])
						col1.plotly_chart(sunb(q1,q2,q1,q2,df),use_container_width=True)
						col3.plotly_chart(sunb(q2,q1,q2,q1,df),use_container_width=True)
						
					else:
						if len(df[q1].unique())>4*len(df[q1].unique()):
							col1, col3 = st.columns([5,5])
							col1.plotly_chart(count(q1,q2,q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent(q1,q2,q1,q2,df),use_container_width=True)
							col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
							col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
							
						else:
							#st.write(df)
							col1, col3 = st.columns([5,5])
							col1.plotly_chart(count(q2,q1,q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent(q2,q1,q2,q1,df),use_container_width=True)
							col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
							col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
							
				else:
					st.write('here')
					col1, col3 = st.columns([1,1])
					col1.plotly_chart(count(q1,q2,q1,q2,df),use_container_width=True)
					col3.plotly_chart(pourcent(q1,q2,q1,q2,df),use_container_width=True)
					col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
					col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
					col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
					col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
					
			
			
			
			
		
		else:
			st.write('')
			st.write('')
			st.write('')
			col1, col2, col3 = st.columns([1,3,1])
			col2.text('Select something on the left side')
		
		
		


if __name__== '__main__':
    main()
    
