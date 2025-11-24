# =========================================================
# HEALTHCARE RL ULTIMATE DASHBOARD (FIXED FOR STREAMLIT CLOUD)
# =========================================================
# sklearn REMOVED (Streamlit Cloud doesn't support it by default)
# Disease Prediction = PURE PYTHON MODEL (No sklearn required)
# Everything else remains same
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from reportlab.pdfgen import canvas
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Healthcare RL Dashboard",
                   layout="wide",
                   page_icon="🧠")

# ---------------------------------------------------------
# SIMPLE LOGIN SYSTEM
# ---------------------------------------------------------
def login_page():
    st.title("🔐 Login to Healthcare RL Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    btn = st.button("Login")

    if btn:
        if username == "admin" and password == "1592005":
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()


# ---------------------------------------------------------
# DASHBOARD HEADER
# ---------------------------------------------------------
st.title("🧠 Healthcare Reinforcement Learning Dashboard – Ultimate Edition")
st.caption("Advanced + RL + Predictions + Chatbot + PDF + Alerts + Simulation – All in one web app")


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("Dashboard Settings")

num_patients = st.sidebar.slider("Synthetic Patients", 10, 200, 50)
lr = st.sidebar.slider("RL Learning Rate", 0.01, 1.0, 0.3)
gamma = st.sidebar.slider("Discount Factor", 0.5, 0.99, 0.9)
episodes = st.sidebar.slider("Training Episodes", 10, 200, 50)

train_rl = st.sidebar.button("Train RL Model")
simulate = st.sidebar.button("Run Simulation")


# =========================================================
# SYNTHETIC DATA GENERATION
# =========================================================
def generate_data(n):
    return pd.DataFrame({
        "Patient_ID": range(1, n + 1),
        "HR": np.random.randint(60, 150, n),
        "SBP": np.random.randint(90, 180, n),
        "RR": np.random.randint(10, 30, n),
        "Temp": np.round(np.random.normal(37, 0.6, n), 1),
        "Risk": np.random.choice(["Low", "Medium", "High"], size=n, p=[0.5, 0.3, 0.2]),
    })

df = generate_data(num_patients)


# =========================================================
# SIMPLE RL MODEL – Q LEARNING
# =========================================================
states = ["Low", "Medium", "High"]
actions = ["No Action", "IV Fluids", "Vasopressor", "Call Doctor"]

Q = pd.DataFrame(0.0, index=states, columns=actions)


def reward(state, action):
    if state == "High":
        return 12 if action in ["Vasopressor", "Call Doctor"] else -6
    if state == "Medium":
        return 8 if action == "IV Fluids" else -4
    if state == "Low":
        return 5 if action == "No Action" else -2
    return 0


def train_q_learning():
    for ep in range(episodes):
        s = random.choice(states)
        a = random.choice(actions)
        r = reward(s, a)
        Q.loc[s, a] += lr * (r + gamma * Q.loc[s].max() - Q.loc[s, a])


if train_rl:
    train_q_learning()
    st.sidebar.success("🎉 Q-Learning Model Trained!")


# =========================================================
# ALERT SYSTEM
# =========================================================
def show_alerts(hr, sbp, temp, rr):
    if hr > 120:
        st.warning("⚠️ High Heart Rate Detected (HR > 120)")
    if sbp > 160:
        st.error("🚨 High Blood Pressure (SBP > 160)")
    if temp > 38.5:
        st.warning("🌡️ Fever Detected (Temp > 38.5°C)")
    if rr > 24:
        st.warning("😮 High Respiratory Rate (RR > 24)")


# =========================================================
# DISEASE PREDICTION MODEL (NO SKLEARN)
# =========================================================
def predict_disease(hr, sbp, temp, rr):
    """
    Simple weighted rule-based model.
    No external libraries needed.
    """
    score = 0
    score += max(0, hr - 110) * 0.015
    score += max(0, sbp - 150) * 0.012
    score += max(0, temp - 38) * 0.20
    score += max(0, rr - 22) * 0.03

    prob = min(1, score)

    pred = 1 if prob > 0.45 else 0
    return pred, prob


# =========================================================
# DOCTOR CHATBOT
# =========================================================
def doctor_chatbot(user_input):
    user_input = user_input.lower()

    if "fever" in user_input or "temperature" in user_input:
        return "High temperature detected. Stay hydrated and monitor fever."

    if "bp" in user_input or "blood pressure" in user_input:
        return "For high BP, reduce salt intake and rest. Seek care if SBP > 160."

    if "heart" in user_input or "hr" in user_input:
        return "High heart rate can be due to stress, dehydration, or infection."

    if "recommend" in user_input or "treatment" in user_input:
        return "Use RL Engine tab to see AI-recommended treatment."

    return "I am a medical assistant bot. Please describe symptoms."


# =========================================================
# PDF REPORT GENERATOR
# =========================================================
def generate_pdf(patient):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 18)

    c.drawString(50, 800, "Healthcare Patient Report")

    c.setFont("Helvetica", 12)
    y = 760
    for key, value in patient.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Vitals Explorer",
    "RL Engine",
    "Patient Lookup",
    "Prediction Model",
    "Doctor Chatbot"
])


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    st.subheader("📊 Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average HR", f"{df.HR.mean():.1f}")
    col2.metric("Average SBP", f"{df.SBP.mean():.1f}")
    col3.metric("Average Temp", f"{df.Temp.mean():.1f}")
    col4.metric("Average RR", f"{df.RR.mean():.1f}")

    show_alerts(df.HR.mean(), df.SBP.mean(), df.Temp.mean(), df.RR.mean())

    st.write("### Risk Distribution")
    st.plotly_chart(px.bar(df["Risk"].value_counts(),
                           title="Risk Level Distribution"),
                    use_container_width=True)


# =========================================================
# TAB 2 — VITALS EXPLORER
# =========================================================
with tab2:
    st.subheader("📉 Vitals Explorer")
    st.plotly_chart(px.histogram(df, x="HR"), use_container_width=True)
    st.plotly_chart(px.line(df.sort_values("Patient_ID"), x="Patient_ID", y="SBP"), use_container_width=True)
    st.plotly_chart(px.box(df, y="Temp"), use_container_width=True)


# =========================================================
# TAB 3 — RL ENGINE
# =========================================================
with tab3:
    st.subheader("🤖 RL Treatment Recommendation Engine")

    if train_rl:
        st.dataframe(Q.style.highlight_max(axis=1))

        st.plotly_chart(px.imshow(Q, text_auto=True, title="Q-Table Heatmap"),
                        use_container_width=True)
    else:
        st.info("Train the RL model from the sidebar.")


# =========================================================
# TAB 4 — PATIENT LOOKUP
# =========================================================
with tab4:
    st.subheader("🔍 Lookup Patient")

    pid = st.number_input("Enter Patient ID", 1, num_patients)

    patient = df[df["Patient_ID"] == pid]

    if not patient.empty:
        p = patient.iloc[0]

        st.metric("HR", p.HR)
        st.metric("SBP", p.SBP)
        st.metric("Temp", p.Temp)
        st.metric("RR", p.RR)
        st.metric("Risk", p.Risk)

        show_alerts(p.HR, p.SBP, p.Temp, p.RR)

        if train_rl:
            st.info(f"💡 RL Suggests: **{Q.idxmax(axis=1)[p.Risk]}**")

        pdf_buffer = generate_pdf({
            "Patient ID": p.Patient_ID,
            "HR": p.HR,
            "SBP": p.SBP,
            "Temp": p.Temp,
            "RR": p.RR,
            "Risk": p.Risk
        })

        st.download_button("📄 Download PDF", data=pdf_buffer,
                           file_name=f"patient_{pid}_report.pdf")


# =========================================================
# TAB 5 — DISEASE PREDICTION (Fixed)
# =========================================================
with tab5:
    st.subheader("🧬 Disease Prediction (No sklearn)")

    hr_i = st.slider("HR", 40, 170, 90)
    sbp_i = st.slider("SBP", 80, 200, 120)
    temp_i = st.slider("Temperature", 35.0, 41.0, 37.0)
    rr_i = st.slider("RR", 8, 35, 16)

    pred, prob = predict_disease(hr_i, sbp_i, temp_i, rr_i)

    if pred == 1:
        st.error(f"🔴 High Risk — Probability: {prob:.2f}")
    else:
        st.success(f"🟢 Low Risk — Probability: {prob:.2f}")


# =========================================================
# TAB 6 — DOCTOR CHATBOT
# =========================================================
with tab6:
    st.subheader("🩺 Doctor Chatbot")

    user_msg = st.text_input("Ask your question:")

    if user_msg:
        st.info(doctor_chatbot(user_msg))


# =========================================================
# DOWNLOAD RAW DATA
# =========================================================
st.sidebar.download_button("Download Dataset (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="patient_dataset.csv")

if train_rl:
    st.sidebar.download_button("Download Q-Table (CSV)",
                               Q.to_csv().encode("utf-8"),
                               file_name="q_table.csv")


# =========================================================
# LIVE SIMULATION
# =========================================================
st.markdown("---")
st.subheader("🚨 Live ICU Simulation")

if simulate:
    placeholder = st.empty()
    for _ in range(20):
        hr = random.randint(60, 160)
        sbp = random.randint(90, 180)
        temp = round(np.random.normal(37, 0.7), 1)
        rr = random.randint(10, 35)

        with placeholder.container():
            st.metric("HR", hr)
            st.metric("SBP", sbp)
            st.metric("Temp", temp)
            st.metric("RR", rr)

            show_alerts(hr, sbp, temp, rr)

        time.sleep(1)


# =========================================================
# LOGOUT
# =========================================================
logout = st.sidebar.button("Logout")

if logout:
    st.session_state["logged_in"] = False
    st.rerun()

st.success("Loaded Healthcare RL Dashboard Successfully!")

 # streamlit run healthcare_rl_dashboard_ULTIMATE.py
# admin 1234
#Why is my heart rate high?
#How to control blood pressure?
#I have a fever, what should I take?