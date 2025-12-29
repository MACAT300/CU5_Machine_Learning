import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title='Diabetes ML Config', layout='wide')

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data
def load_data():
    URL = 'https://raw.githubusercontent.com/SubaashNair/dataset/refs/heads/main/datasets/medical/diabetes_clean.csv'
    df = pd.read_csv(URL)
    return df

df = load_data()

# -------------------------------
# Model training (DO NOT TOUCH)
# -------------------------------
@st.cache_resource
def train_model(data):
    x = data.drop('diabetes', axis=1)
    y = data['diabetes']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    acc = accuracy_score(y_test, model.predict(x_test))
    return model, acc

model, accuracy = train_model(df)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image(
    "https://cdn.who.int/media/images/default-source/infographics/who-emblem.png?sfvrsn=877bb56a_2",
    width=250
)
st.sidebar.title('Diabetes App')
page = st.sidebar.radio('Navigate', ['Dashboard', 'ML'])
st.sidebar.info(f"Model Accuracy: {accuracy:.2f}")
st.sidebar.markdown('---')
st.sidebar.caption('Built with nasi lemak and teh tarik')

# ===============================
# DASHBOARD PAGE
# ===============================
if page == 'Dashboard':
    st.title('Health Analytics Dashboard')
    st.markdown('Explore the diabetes dataset')

    # ---------------------------
    # Metrics
    # ---------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric('Total Records', len(df))

    with c2:
        st.metric('Avg Glucose', f"{df['glucose'].mean():.1f} mg/dL")

    with c3:
        st.metric('Avg BMI', f"{df['bmi'].mean():.1f}")

    with c4:
        count = len(df[df['diabetes'] == 1])
        st.metric(
            'Diabetes Cases',
            count,
            delta=f"{(count / len(df)) * 100:.2f}%"
        )

    st.divider()

    # ---------------------------
    # Tabs
    # ---------------------------
    tab1, tab2, tab3 = st.tabs(['Overview', 'Deep Dive', 'Data'])

    # ===== TAB 1 =====
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader('Glucose vs BMI Status')
            fig_scatter = px.scatter(
                df,
                x='glucose',
                y='bmi',
                color='diabetes',
                size='age',
                hover_data=['insulin', 'diastolic'],
                title='Glucose vs BMI',
                color_continuous_scale='pinkyl'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.subheader('Outcome Distribution')
            fig_hist = px.histogram(
                df,
                x='age',
                color='diabetes',
                title='Age Distribution by Diabetes Status'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ===== TAB 2 =====
    with tab2:
        c_a, c_b = st.columns(2)

        with c_a:
            st.subheader('Age Distribution')
            fig_age = px.histogram(
                df,
                x='age',
                color='diabetes',
                title='Age Distribution by Diabetes Status'
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with c_b:
            st.subheader('Insulin Level Box Plot')
            fig_box = px.box(
                df,
                x='diabetes',
                y='insulin',
                color='diabetes',
                title='Insulin Levels Box Plot'
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ===== TAB 3 =====
    with tab3:
        st.subheader('Raw Data')
        st.dataframe(df.sample(50), use_container_width=True)

# Prediction page
elif page == "ML":
    st.title('Risk Predictor')
    st.markdown('### ML Diabetes Risk Predictor')

    with st.expander('Patient vital entry form', expanded=True):
        with st.form('prediction_form'):

            c1, c2, c3 = st.columns(3)
            pregnancies = c1.number_input('Pregnancies', 0, 20, 0)
            glucose = c2.number_input('Glucose', 0, 200, 100)
            diastolic = c3.number_input('Diastolic BP', 0, 140, 72)

            c4, c5, c6 = st.columns(3)
            triceps = c4.number_input('Triceps Skin Thickness', 0, 100, 20)
            insulin = c5.number_input('Insulin', 0, 300, 80)
            bmi = c6.number_input('BMI', 0.0, 60.0, 25.0)

            c7, c8 = st.columns(2)
            dpf = c7.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
            age = c8.number_input('Age', 0, 120, 30)

            submit_btn = st.form_submit_button('Analyze Risk', type='primary')

    if submit_btn:
        input_data = [[pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age]]
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        st.divider()

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            if prediction == 1:
                st.error('High Risk Detected')
                st.metric('Risk Probability', f"{prob:.1%}")
            else:
                st.success('Low Risk Detected')
                st.metric('Risk Probability', f"{prob:.1%}")

        with col_r2:
            st.write('### Prediction Confidence')
            st.progress(prob)
            if prob > 0.5:
                st.warning('Please consult a specialist for further testing')
            else:
                st.balloons()
                st.info("Great! Keep maintaining your lifestyle")
