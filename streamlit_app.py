import streamlit as st

# Configuración básica de la aplicación
st.set_page_config(page_title="Análisis Financiero", layout="wide")

# Título de la aplicación
st.title("Herramientas de Análisis Financiero")

# Menú principal
menu = st.sidebar.selectbox(
    "Seleccione una categoría",
    ("Acciones", "Gestión de Carteras")
)

# Submenú en la categoría "Acciones"
if menu == "Acciones":
    submenu_acciones = st.sidebar.selectbox(
        "Seleccione un análisis para acciones",
        ("Análisis Técnico", "Análisis Fundamental", "Riesgo")
    )
    
    if submenu_acciones == "Análisis Técnico":
        st.subheader("Análisis Técnico")
        # Agrega aquí tu código para el análisis técnico
        st.write("Aquí puedes agregar el código para el análisis técnico.")
        
    elif submenu_acciones == "Análisis Fundamental":
        st.subheader("Análisis Fundamental")
        # Agrega aquí tu código para el análisis fundamental
        st.write("Aquí puedes agregar el código para el análisis fundamental.")
        
    elif submenu_acciones == "Riesgo":
        st.subheader("Análisis de Riesgo")
        # Agrega aquí tu código para el análisis de riesgo
        st.write("Aquí puedes agregar el código para el análisis de riesgo.")

# Submenú en la categoría "Gestión de Carteras"
elif menu == "Gestión de Carteras":
    submenu_carteras = st.sidebar.selectbox(
        "Seleccione una herramienta para gestión de carteras",
        ("Análisis de Carteras", "Optimización de Carteras")
    )
    
    if submenu_carteras == "Análisis de Carteras":
        st.subheader("Análisis de Carteras")
        # Agrega aquí tu código para el análisis de carteras
        st.write("Aquí puedes agregar el código para el análisis de carteras.")
        
    elif submenu_carteras == "Optimización de Carteras":
        st.subheader("Optimización de Carteras")
        # Agrega aquí tu código para la optimización de carteras
        st.write("Aquí puedes agregar el código para la optimización de carteras.")
