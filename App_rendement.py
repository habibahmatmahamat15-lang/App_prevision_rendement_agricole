import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
df= pd.read_csv("rendement_cleaned.csv")

# Sidebar - Navigation
with st.sidebar:
    st.image("senegal.jpg", width=100)
    st.title("Navigation")
    page = st.radio(
        "Sélectionnez une section",
        ["Accueil", "Prévision", "Visualisations", "À propos"]
    )
    st.markdown("---")
    st.info("**Cultures supportées:**\n- Niébé\n- Maïs\n- Pastèque\n- Arachide\n- Mil")
    st.markdown("---")
    st.caption("Version 1.0 - 2026")
## Titre d'application 
st.title("Système de Prévision Agricole")
if page == "Accueil":
    col1, col2 = st.columns([2, 1])
        
    with col1:
        st.markdown("## Bienvenue sur le système de prévision agricole")
        st.write("""
            Ce système permet de prendre de meilleures 
            décisions agricoles concernant vos cultures de maïs et céréales locales.
            """)
            
        st.markdown("### Que pouvez-vous faire ?")
            
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("""
                **🌱 Prédire les rendements agricoles**
                - Estimer le rendement des cultures (t/ha)
                - Basé sur la zone, la culture, la pluviométrie, la quantité d'engrais
                - Aide à la prise de décision avant la saison
                """)
        with col_b:
            st.markdown("""
                **📊 Analyser les données agricoles**
                - Comparer les rendements par zone
                - Identifier les cultures les plus performantes
                - Visualiser l’impact des précipitations
                """)
        with col_c:
            st.markdown("""
                **📈 Suivre et exploiter les résultats**
                - Évaluer la performance du modèle
                - Comparer valeurs réelles et prédites
                - Appui à la planification agricole
            """)
            
                
    with col2:
        st.markdown("### Régions couvertes")
        regions = ["Fatick", "Foundiougne", "Kaolack", "Diourbel", "Birkelane","Nioro"]
        for region in regions:
            st.success(f"✓ {region}")
            
        st.markdown("### Besoin d'aide ?")
        st.info("Consultez la section **À propos** pour plus d'informations.")

elif page == "Prévision":
    st.header("Prévision du rendement agricole")
    st.write("Utilisez vos données pour prédire le rendement des cultures.")

    X = df.drop("yield", axis= 1)
    y = df["yield"]
        
        # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)
        
        # Prédictions
    y_pred = model.predict(X_test)
    
    ## Les valeurs entrées par l'utilisateur
    def input_value(culture_type, zone, rainfall, fertilizer_quantity):
        data= np.array([
            culture_type, zone, rainfall, fertilizer_quantity
        ])
        prediction_data = model.predict(data.reshape(1, -1))
        return prediction_data
    
    col_a, col_b = st.columns(2)

    with col_a:
        culture_type = st.number_input(
            "Code culture (0=Arachide, 1=Maïs, 2=Mil, 3=Niébé, 4=Pastèque)",
            min_value=0,
            max_value=4,
            step=1
        )
        # Sélection zone
        zone = st.number_input(
            "Code zone (0=Birkelane, 1=Diourbel, 2=Fatick, 3=Foundiougne, 4=Kaolack, 5=Nioro)",
            min_value=0,
            max_value=5,
            step=1
        )
    with col_b:
        rainfall= st.number_input(
            "rainfall (mm)", min_value=309.2, max_value=897.6, step=50.0
        )
        fertilizer_quantity= st.number_input(
            "fertilizer_quantity (kg)", min_value=20, max_value=150, step=10)

    if st.button("🔮 prediction"):
        prediction= input_value(culture_type, zone, rainfall, fertilizer_quantity)
        st.success(f"💰 Rendement estimé : **{prediction[0]:,.2f} t/ha**")


    st.subheader("📊 Performance du modèle")
    r2 = r2_score(y_test, y_pred)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    col3, col4 = st.columns(2)
    col3.metric("R² Score", f"{r2:.3f}")
    col4.metric("MSE (t/ha)", f"{mse:,.0f}")
                
    st.subheader("Résultats de la prévision")
    
    cola , colb = st.columns(2)
    with cola:
        st.write("Graphique Réel vs Prédit")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.scatter(y_test, y_pred)
        ax1.plot(
            [y_test.min(), y_test.max()],
            linestyle="--"
        )
        ax1.set_xlabel("Réel")
        ax1.set_ylabel("Prédit")
        ax1.set_title("Réel vs Prédit")
        st.pyplot(fig1)
    with colb:
        # Affichage des prédictions
        st.write("Grille des valeurs réelles vs prédites")
        results = pd.DataFrame({
            "Valeurs réelles": y_test,
            "Prédictions": y_pred
            })
        st.dataframe(results)
        
elif page == "Visualisations":
    st.header("Visualisations des données agricoles")
    st.write("Explorez les tendances et les relations dans vos données agricoles.")
    tab1, tab2 = st.tabs(["Tendances Régionales", "Analyse Climatique"])
    with tab1:
        # Préparer les données pour la visualisation
        df_viz = df.groupby(['zone', 'culture_type'])['yield'].mean().reset_index()
        df_viz.columns = ['Zone', 'Culture', 'Rendement']
        
        # Graphique en barres
        fig = px.bar(
            df_viz,
            x='Zone',
            y='Rendement',
            color='Culture',
            barmode='group',
            title='Rendements Moyens par Zone et Culture (t/ha)',
            color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Répartition du rendement par zone
        st.subheader("📊 Répartition du rendement par zone")
        zone_rendement = df.groupby('zone')['yield'].sum().reset_index()
        zone_rendement.columns = ['Zone', 'Rendement Total']
        
        fig_pie = px.pie(
            zone_rendement,
            values='Rendement Total',
            names='Zone',
            title='Distribution du rendement total par zone',
            color_discrete_sequence=['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#9C27B0', '#00BCD4']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.info("Ces données sont basées sur les prévisions historiques du système.")
    with tab2:

        st.subheader("🌦️ Analyse des précipitations et du rendement")
        fig2 = px.scatter(
            df,
            x='rainfall',
            y='yield',
            color='zone',
            title='Relation entre les précipitations et le rendement',
            labels={'rainfall': 'Précipitations (mm)', 'yield': 'Rendement (t/ha)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("Comprendre l'impact des conditions climatiques sur le rendement agricole.")
elif page == "À propos":
    st.header("À propos du Système de Prévision Agricole")
    tab1, tab2,tab3 = st.tabs(["Informations", "Équipe de Développement", "Contact"])
    with tab1:
        st.write("""
                Ce système a été développé pour aider les agriculteurs à optimiser leurs pratiques agricoles 
                en fournissant des prévisions précises du rendement des cultures basées sur des données réelles.
                
            **Fonctionnalités principales :**
            - Prévisions de rendement basées sur des modèles de machine learning.
            - Visualisations interactives pour analyser les tendances agricoles.
            - Recommandations personnalisées en fonction des conditions locales.
            
            **Technologies utilisées :**
            - Streamlit pour l'interface utilisateur.
            - Pandas et NumPy pour la manipulation des données.
            - Scikit-learn pour le développement des modèles de machine learning.
            - Plotly pour les visualisations interactives.
            
            Pour toute question ou assistance, veuillez contacter l'équipe de support.
        """)
    with tab2:
        st.subheader("Équipe de Développement")
        st.markdown("""
    ### 👤 Auteur du projet

    **Ahmat Mahamat Abdel-Aziz HABIB**  
    🎓 Data Scientist  
    💻 Développeur  
    📊 Analyste de données  
    """)
    with tab3:
        st.subheader("Contact")
        st.markdown("""
    **Pour toute question ou assistance, veuillez nous contacter à :**

    - 📧 **Email** : habibahmatmahamat15@gmail.com  
    - 📞 **Téléphone** : +221 78 752 75 78
    """)


# Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.caption("Système de Prévision Agricole")

with col_f2:
    st.caption("Fait pour le Sénégal")

with col_f3:
    st.caption("L'IA au service de l'agriculture")

