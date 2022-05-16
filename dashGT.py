import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pickle
import re
from collections import Counter
from PIL import Image
from streamlit_option_menu import option_menu
from dashboard_fonctions import *

st.set_page_config(layout="wide")


@st.cache
def load_data():
	continues = pickle.load(open("cont_feat.p", "rb"))
	data = pd.read_csv('viz.csv', sep='\t')
	data['Gender']=data['Gender'].apply(lambda x:'Male' if x[0]=='M' else 'Female')
	data.drop([i for i in data if 'Unnamed' in i], axis=1, inplace=True)
	correl = pd.read_csv('graphs.csv')
	questions = pd.read_csv('questions.csv',index_col=0)
	#questions.drop([0], axis=0, inplace=True)
	#questions.columns=['code','parent','type','treatment','other','question']
	codes = pd.read_csv('codes.csv', sep='\t')
	return data, correl, questions, codes


data, correl, questions, codes = load_data()


#st.write(questions)

def main():
	#st.write(codes)
	st.sidebar.title("")

	title1, title3 = st.columns([9,2])

	with st.sidebar:
		topic = option_menu(None, ['Machine learning results', 'Correlations'],
							icons=["cpu", 'bar-chart'],
							menu_icon="app-indicator", default_index=0,
							)


	# ______________________________________ SHAP __________________________________#

	if topic == 'Machine learning results':
		
		#st.write(questions)
		
		
		title1.title('Machine learning results on predictive model trained on question:')

		st.title('')
		st.markdown("""---""")
		st.subheader('Note:')
		st.write('A machine learning model has been run on the above mentionned question.'
				 'The objective of this is to identify, specificaly for these question, which are the the aspects of the'
				 ' project that influenced the most the responses to these question.'
				 'The figures below shows which parameters have a greater impact in the prediction '
				 'of the model than a normal random feature (following a statistic normal law)')
		st.write('')
		st.write('HOW TO READ THE GRAPH:')
		st.write('Each line of the graph represents one feature of the survey that is important to predict '
				 'the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. '
				 'A red point represent a high value of the specific feature and a blue point a '
				 'low value (a purple one an intermediate value).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a '
				 'better note for the question while on the left side, this specific caracter of the person '
				 'reduced the final result of the prediction.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and '
				 'the interpretation of the graphs is written below.')
		
		st.write("For more information check out on: [link](http://axiomdashboard.com/shap.html)")
		
		st.markdown("""---""")
		
		
		
		for i in ['Resilient','FuturePlan','Optimism']:
			
			st.subheader(questions.loc[i]['question'])
			
			temp = Image.open(i+'.png')
			image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
			image.paste(temp, (0, 0), temp)
			st.image(image, use_column_width = True)

		

		
	# ______________________________________ CORRELATIONS __________________________________#

	elif topic == 'Correlations':
		sub_topic = st.sidebar.radio('Select the topic you want to look at:',['Geographic','Other'])

		title1.title('Main correlations uncovered from the database related to '+sub_topic)

		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )

		soustableau_correl = correl[correl['categories'].apply(lambda x: sub_topic in x)]
		
		#st.write(soustableau_correl)
		
		st.markdown("""---""")
		k = 0
		for absc in soustableau_correl['variable_x'].unique():
			
			#st.write(absc)
			quest = soustableau_correl[soustableau_correl['variable_x'] == absc]
			#st.write(quest)
			for i in range(len(quest)):
				#st.write(quest.iloc[i])
				if quest.iloc[i]['filter']==quest.iloc[i]['filter']:
					if quest.iloc[i]['filter'] != 'toilet':
						df=data[data[quest.iloc[i]['filter']]=='Yes'].copy()
					else:
						df = data[data[quest.iloc[i]['filter']] != 'Bush'].copy()
				else:
					df=data.copy()
				#st.write(df.shape)
				
				df=select_data(quest.iloc[i],df,cat_cols)
				show_data(quest.iloc[i],df,codes)
				st.markdown("""---""")
				

	
################################################################################################

if __name__ == '__main__':
	
	main()
	
