import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout='wide',page_title='Basic Layout')
st.title('Layout and State App')


# Sidebar
with st.sidebar:
    st.header('Settings')
    theme = st.selectbox('Choose theme',['Light','Dark','System'])

# Tabs
tab1,tab2,tab3 = st.tabs(['Analytics','Data','ML Prediction'])

with tab1:
    st.subheader('Analytics Dashboard')
    col1,col2,col3 = st.columns(3)
    col1.metric('Revenue','$12000','+12%')
    col2.metric('User Satisfaction','1,234','-12%')
    col3.metric('Rating','4.4/5','0.2%')


with tab2:
    st.subheader('Config')
    st.checkbox('Enable Advanced Mode')

with tab3:
    st.subheader('Predict')
    st.checkbox('AI Mode')

# Expanders
st.divider()
with st.expander('See explanation'):
    st.write('This content is hidden by default.')
# Sessions State
st.divider()
st.subheader('State Management')
st.write('Variables reset store in state session')

if 'count' not in st.session_state:
    st.session_state.count=0

col_a,col_b = st.columns(2)
with col_a:
    if st.button('Incerement Counter'):
        st.session_state.count+=1
with col_b:
    st.write(f"Current count: **{st.session_state.count}**")

st.divider()
st.header('DataFrame')
df = pd.DataFrame({
    'Name':['Qirun','Ryan','Rentai','Sung'],
    'Age':[67,76,69,96],
    'City': ['Malaysia','India/African','Japan','Korea']
})

col1,col2 = st.columns(2)
with col1:
    st.write('Interactive DataFrame')
    st.dataframe(df)

with col2:
    st.write('Static Dataframe')
    st.table(df)

# Interactive Widgets

st.divider()
st.header('Interactive widgets')

name = st.text_input('Enter your baddie name', 'bumi')
if st.button('say ni hao'):
    st.success(f"hello,{name}!!")

age = st.slider('Select your age:', 0,100,25)
st.write(f"How are you old: {age}")

# Simple Charts
st.divider()
st.header('Simple Chart')
chart_data = pd.DataFrame((np.random.rand(20,3)), columns=['A','B','D'])
st.line_chart(chart_data)