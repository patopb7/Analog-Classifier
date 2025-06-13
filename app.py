import streamlit as st
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from PIL import Image
from Function_prediction import run_prediction
import pandas as pd
import altair as alt

# Simple model simulation
def model(x):
    return jnp.exp(x) / jnp.sum(jnp.exp(x))  # softmax for demonstration

# Load data
iris = load_iris()
X = iris['data']
y = iris['target'].reshape(-1, 1)
feature_names = iris['feature_names']
target_names = iris['target_names']

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# App title
st.title("Iris Flower Classifier 🌸")

# Sample selection (0–149)
sample_index = st.slider("Select a sample", 0, 149, 0)

# Selected sample data
x_sample = jnp.array(X[sample_index])
y_true = y_onehot[sample_index]
y_pred = model(x_sample)

# True label
st.subheader(f"🎯 Sample {sample_index}:")
true_class = target_names[np.argmax(y_true)]

# Layout with two columns
col1, col2 = st.columns([4, 1])  # col1 = 1/4 width

image_options = [
    "Images/iris_virginica.png",
    "Images/iris_setosa.png",
    "Images/iris_versicolor.png"
]
image_path = image_options[sample_index % 3]

with col1:
    # st.image("Images/Dimensions.png", caption="Iris flower", use_container_width=True)
    st.success(f"{true_class}")
    
with col2:
    st.image(image_path, caption="", use_container_width=True)




# Layout with two columns
col1, col2 = st.columns([2, 1])  # col1 = 1/4 width

with col1:
    st.subheader("📐 Sample features:")
    st.write(f"**Sepal length:** {x_sample[0]:.2f} cm")
    st.write(f"**Sepal width:** {x_sample[1]:.2f} cm")
    st.write(f"**Petal length:** {x_sample[2]:.2f} cm")
    st.write(f"**Petal width:** {x_sample[3]:.2f} cm")
    #st.image(image_path, caption="Iris flower", use_container_width=True)

with col2:
    st.image("Images/Dimensions.png", caption="Iris flower", use_container_width=True)


st.title("⚙️ Analog Classifier")

st.markdown(
    """
    Each input feature is applied as a physical force in a structural system. The resulting displacements are measured at three specific locations. The predicted class corresponds to the point with the **maximum displacement**.
    """
)

rho = np.load("rho_opt0.npy")
nelx, nely = 80, 30

title_figure = st.empty()
plot_area = st.empty()  # crea una zona que puede ser actualizada


fig, ax = plt.subplots(figsize=(6, 3))
ax.imshow(-np.flipud(rho.reshape((nelx, nely)).T), cmap='gray')
#ax.set_title(status)


ax.plot(30, 0, marker='s', color='red', markersize=8)
plt.plot(40, 0, marker='s', color='red', markersize=8)
plt.plot(50, 0, marker='s', color='red', markersize=8)

plt.arrow(20, 29.5, 0, 3, head_width=1, head_length=1, fc='red', ec='red')
plt.arrow(30, 29.5, 0, 3, head_width=1, head_length=1, fc='red', ec='red')
plt.arrow(50, 29.5, 0, 3, head_width=1, head_length=1, fc='red', ec='red')
plt.arrow(60, 29.5, 0, 3, head_width=1, head_length=1, fc='red', ec='red')

ax.text(12, 35, r"Sepal$_{length}$", fontsize=10, fontname="Times New Roman")
ax.text(23.5, 35, r"Sepal$_{width}$", fontsize=10, fontname="Times New Roman")
ax.text(43, 35, r"Petal$_{length}$", fontsize=10, fontname="Times New Roman")
ax.text(55, 35, r"Petal$_{width}$", fontsize=10, fontname="Times New Roman")

ax.text(29, -3, r"d$_1$", fontsize=10, fontname="Times New Roman")
ax.text(39, -3, r"d$_2$", fontsize=10, fontname="Times New Roman")
ax.text(49, -3, r"d$_3$", fontsize=10, fontname="Times New Roman")

plt.ylim(35, -2)

ax.axis('off')
#title_figure.write('sadas')
plot_area.pyplot(fig, clear_figure=True)
    


# --- Run button ---
if st.button("Run prediction"):
    # Model prediction
    st.subheader("🤖 Model prediction:")

    status_message = st.empty()
    status_message.info("Running Prediction...")
    u_pred = run_prediction(rho, sample_index)
    U_prediction = jnp.abs(u_pred)
    
    labels = ['Setosa', 'Versicolor', 'Virginica']
    
    predicted_class = labels[np.argmax(U_prediction)]
    st.info(f"{predicted_class}")

    
    # # Configuración
    # labels = ['Setosa', 'Versicolor', 'Virginica']
    # colors = ['red', 'green', 'blue']
    
    # # Crear figura
    # fig, ax = plt.subplots()
    # ax.bar(labels, [-val for val in -jnp.abs(u_pred)], color=colors)
    
    # # Etiquetas
    # ax.set_title("Class distribution")
    # ax.set_ylabel("Probability")
    # ax.grid(True, axis='y')
    
    # # Mostrar en Streamlit
    # st.pyplot(fig)
    
    
    

    # Probability distribution chart
    st.subheader("📊 Displacement Response")

    
    
    # Tus datos
    #[0.2, 0.5, 0.3]
    labels = ['d1: Setosa', 'd2: Versicolor', 'd3: Virginica']
    colors = ['red', 'green', 'blue']
    
    df = pd.DataFrame({
        "Class": labels,
        "Displacement": U_prediction,
        "Color": colors
    })
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Class", axis=alt.Axis(labelAngle=45)),
        y="Displacement",
        color=alt.Color("Color", scale=None)  # usa colores definidos
    )
    
    st.altair_chart(chart, use_container_width=True)
    status_message.empty()  # elimina el mensaje


else:
    st.markdown("\n👉 Use the sidebar to select your sample and press the button to classify it.")






