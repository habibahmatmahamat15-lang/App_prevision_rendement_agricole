import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import io

if "historique" not in st.session_state:
    st.session_state["historique"] = []


# Load dataset
df= pd.read_csv("rendement_cleaned.csv")

# =============================
# Préparation des données et entraînement du modèle
# =============================
X = df.drop("yield", axis=1)
y = df["yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =============================
# Fonction de prédiction
# =============================
def input_value(culture_type, zone, rainfall, fertilizer_quantity):
    data = np.array([
        culture_type,
        zone,
        rainfall,
        fertilizer_quantity
    ])
    prediction_data = model.predict(data.reshape(1, -1))
    return prediction_data

# Sidebar - Navigation
with st.sidebar:
    st.title("🌐 Navigation")
    page = st.radio(
        "Sélectionnez une section",
        ["Accueil", "Prévision", "Visualisations", "Historique", "Rapport", "À propos"]
    )
    st.markdown("Types de cultures")
    cultures = ["Niébé", "Maïs", "Pastèque", "Arachide", "Mil"]
    for culture in cultures:
        st.success(f"✓ {culture}")
    st.markdown("---")
    st.caption("Version 1.0 - 2026")
## Titre d'application 
st.title("Système de Prévision Agricole")
if page == "Accueil":
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
        
            
    st.markdown("### Besoin d'aide ?")
    st.info("Consultez la section **À propos** pour plus d'informations.")

elif page == "Prévision":
    st.header("Prévision du rendement agricole")
    st.write("Utilisez vos données pour prédire le rendement des cultures.")

    # =============================
    # MAPPINGS
    # =============================
    culture_mapping = {
        "Arachide": 0,
        "Maïs": 1,
        "Mil": 2,
        "Niébé": 3,
        "Pastèque": 4
    }

    zone_mapping = {
        "Birkelane": 0,
        "Diourbel": 1,
        "Fatick": 2,
        "Foundiougne": 3,
        "Kaolack": 4,
        "Nioro": 5
    }

    # =============================
    # Interface utilisateur
    # =============================
    col_a, col_b = st.columns(2)

    with col_a:
        culture_label = st.selectbox(
            "Type de culture",
            options=list(culture_mapping.keys())
        )

        zone_label = st.selectbox(
            "Zone agricole",
            options=list(zone_mapping.keys())
        )

    with col_b:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=309.2,
            max_value=897.6,
            step=50.0
        )

        fertilizer_quantity = st.number_input(
            "Fertilizer quantity (kg)",
            min_value=20,
            max_value=150,
            step=10
        )

    # =============================
    # Mapping AVANT prédiction
    # =============================
    culture_type = culture_mapping[culture_label]
    zone = zone_mapping[zone_label]

    # =============================
    # Bouton prédiction
    # =============================
    if st.button("🔮 Prédire le rendement"):
        prediction = input_value(
            culture_type,
            zone,
            rainfall,
            fertilizer_quantity
        )

        st.success(
            f"🌾 Rendement estimé : **{prediction[0]:,.2f} t/ha**"
        )
        # Enregistrer dans l'historique
        if "historique" not in st.session_state:
            st.session_state.historique = []

        st.session_state.historique.append({
            "Culture": culture_label,
            "Zone": zone_label,
            "Précipitations (mm)": rainfall,
            "Quantité de fertilisant (kg)": fertilizer_quantity,
            "Rendement estimé (t/ha)": prediction[0]
        })

                
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
        
        # Boutons de téléchargement pour les résultats du modèle
        col_csv_res, col_excel_res = st.columns(2)
        with col_csv_res:
            csv_results = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger résultats (CSV)",
                csv_results,
                "resultats_modele.csv",
                "text/csv"
            )
        with col_excel_res:
            buffer = io.BytesIO()
            results.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            excel_results = buffer.getvalue()
            st.download_button(
                "📊 Télécharger résultats (Excel)",
                excel_results,
                "resultats_modele.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
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
        
        # Afficher les données des rendements moyens
        st.subheader("📋 Données des rendements moyens")
        st.dataframe(df_viz, use_container_width=True)
        
        # Boutons de téléchargement pour les rendements moyens
        col_csv1, col_excel1 = st.columns(2)
        with col_csv1:
            csv_viz = df_viz.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger en CSV",
                csv_viz,
                "rendements_moyens.csv",
                "text/csv"
            )
        with col_excel1:
            buffer = io.BytesIO()
            df_viz.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            excel_viz = buffer.getvalue()
            st.download_button(
                "📊 Télécharger en Excel",
                excel_viz,
                "rendements_moyens.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
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
        
        # Afficher les données de répartition par zone
        st.dataframe(zone_rendement, use_container_width=True)
        
        # Boutons de téléchargement pour la répartition par zone
        col_csv2, col_excel2 = st.columns(2)
        with col_csv2:
            csv_zone = zone_rendement.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger répartition (CSV)",
                csv_zone,
                "repartition_zone.csv",
                "text/csv"
            )
        with col_excel2:
            buffer = io.BytesIO()
            zone_rendement.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            excel_zone = buffer.getvalue()
            st.download_button(
                "📊 Télécharger répartition (Excel)",
                excel_zone,
                "repartition_zone.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
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
        
        # Afficher les données complètes
        st.subheader("📋 Données complètes")
        st.dataframe(df, use_container_width=True)
        
        # Boutons de téléchargement pour les données complètes
        col_csv3, col_excel3 = st.columns(2)
        with col_csv3:
            csv_full = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Télécharger données complètes (CSV)",
                csv_full,
                "donnees_completes.csv",
                "text/csv"
            )
        with col_excel3:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            excel_full = buffer.getvalue()
            st.download_button(
                "📊 Télécharger données complètes (Excel)",
                excel_full,
                "donnees_completes.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.info("Comprendre l'impact des conditions climatiques sur le rendement agricole.")

elif page == "Historique":
    st.header("📜 Historique des prévisions")
    if not st.session_state["historique"]:
        st.info("Aucune prédiction enregistrée pour le moment.")
    else:
        historique_df = pd.DataFrame(st.session_state["historique"])
        st.dataframe(historique_df, use_container_width=True)

        # Boutons de téléchargement
        col_csv_hist, col_excel_hist = st.columns(2)
        with col_csv_hist:
            csv = historique_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger en CSV",
                csv,
                "historique_previsions.csv",
                "text/csv"
            )
        with col_excel_hist:
            buffer = io.BytesIO()
            historique_df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            excel = buffer.getvalue()
            st.download_button(
                "📊 Télécharger en Excel",
                excel,
                "historique_previsions.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
elif page == "Rapport":
    st.header("📊 Rapport d’analyse du modèle")

    st.subheader("🔍 Description du modèle")
    st.markdown("""
    - **Modèle utilisé** : Régression linéaire
    - **Variables explicatives** :
        - Type de culture
        - Zone agricole
        - Pluviométrie
        - Quantité d'engrais
    - **Variable cible** : Rendement agricole (t/ha)
    """)

    st.subheader("📊 Performance du modèle")
    r2 = r2_score(y_test, y_pred)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    col3, col4 = st.columns(2)
    col3.metric("R² Score", f"{r2:.3f}")
    col4.metric("MSE (t/ha)", f"{mse:,.0f}")
    
    st.info("Ces métriques indiquent la performance du modèle de prédiction.")
    st.markdown("""
    - Un R² proche de 1 indique une bonne capacité explicative du modèle.
    - Un MSE faible suggère des erreurs de prédiction réduites.
    """)
    st.subheader("🧠 Interprétation")
    st.markdown("""
    - Le modèle explique une part significative de la variation du rendement.
    - La pluviométrie et la quantité d'engrais ont un impact important.
    - Les performances peuvent être améliorées avec plus de données terrain.
    """)

    st.subheader("✅ Recommandations agricoles")
    st.markdown("""
    - Adapter les apports d'engrais selon la culture.
    - Privilégier les périodes à pluviométrie régulière.
    - Collecter davantage de données locales pour améliorer la précision.
    """)


elif page == "À propos":
    st.header("À propos du Système de Prévision Agricole")
    tab1, tab2 = st.tabs(["Informations", "Équipe de Développement"])
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
        st.divider()
        st.subheader("Contact")
        st.markdown("""
        **Pour toute question ou assistance, veuillez nous contacter :**
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











