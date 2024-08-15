import streamlit as st

def main():
    # Título de la aplicación
    st.title("Herramientas de Análisis Financiero")
    
    # Menú desplegable
    menu = st.sidebar.selectbox(
        "Selecciona una opción",
        ["Análisis de Acciones", "Gestión de Carteras"]
    )
    
    if menu == "Análisis de Acciones":
        # Submenú para Análisis de Acciones
        submenu = st.sidebar.selectbox(
            "Selecciona un tipo de análisis",
            ["Análisis Técnico", "Análisis Fundamental", "Análisis de Riesgo"]
        )
        
        if submenu == "Análisis Técnico":
            st.subheader("Análisis Técnico")
            st.write("Aquí va el contenido para el análisis técnico de acciones.")
        
        elif submenu == "Análisis Fundamental":
            st.subheader("Análisis Fundamental")
            st.write("Aquí va el contenido para el análisis fundamental de acciones.")
        
        elif submenu == "Análisis de Riesgo":
            st.subheader("Análisis de Riesgo")
            st.write("Aquí va el contenido para el análisis de riesgo de acciones.")
    
    elif menu == "Gestión de Carteras":
        # Submenú para Gestión de Carteras
        submenu = st.sidebar.selectbox(
            "Selecciona un tipo de gestión de cartera",
            ["Optimización de Carteras", "Análisis de Cartera"]
        )
        
        if submenu == "Optimización de Carteras":
            st.subheader("Optimización de Carteras")
            st.write("Aquí va el contenido para la optimización de carteras.")
        
        elif submenu == "Análisis de Cartera":
            st.subheader("Análisis de Cartera")
            st.write("Aquí va el contenido para el análisis de carteras.")

if __name__ == "__main__":
    main()
