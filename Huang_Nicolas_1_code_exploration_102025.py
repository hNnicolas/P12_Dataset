# ----------------------------------------------------------
# Projet ZenAssist – Classification des réclamations clients
# ----------------------------------------------------------

# ----------------------------------------------------------
# 0. Installer les packages nécessaires
# ----------------------------------------------------------
# pip install mistralai scikit-learn pandas matplotlib python-dotenv

# ----------------------------------------------------------
# 1. Import des bibliothèques
# ----------------------------------------------------------
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from dotenv import load_dotenv

# Config pandas et matplotlib
pd.set_option('display.max_columns', None)
plt.style.use('seaborn')

# ----------------------------------------------------------
# 2. Charger la clé API Mistral depuis .env.local
# ----------------------------------------------------------
load_dotenv('.env.local') 
API_KEY = os.getenv('MISTRAL_API_KEY')
if API_KEY is None:
    raise ValueError("La clé Mistral API n'est pas définie dans le .env.local !")

client = MistralClient(api_key=API_KEY)

# ----------------------------------------------------------
# 3. Chargement des données (simulation)
# ----------------------------------------------------------
data = {
    "reclamation": [
        "Mon produit est arrivé cassé.",
        "Impossible de me connecter à mon compte.",
        "Le paiement a été refusé alors que ma carte est valide.",
        "La livraison a pris 3 semaines au lieu de 3 jours.",
        "Mon mot de passe ne fonctionne plus.",
        "Le colis ne correspond pas à ma commande.",
        "Je veux annuler mon abonnement.",
        "Le produit ne démarre pas après l’installation.",
        "J’ai été débité deux fois pour la même commande.",
        "La taille du vêtement n’est pas conforme au guide."
    ] * 200,
    "etiquette": [
        "Produit", "Compte", "Paiement", "Livraison", "Compte",
        "Produit", "Abonnement", "Produit", "Paiement", "Produit"
    ] * 200
}

df = pd.DataFrame(data)

# ----------------------------------------------------------
# 3a. Suppression des données invalides
# ----------------------------------------------------------
df = df.dropna(subset=['reclamation', 'etiquette'])

# ----------------------------------------------------------
# 3b. Graphe de répartition des étiquettes
# ----------------------------------------------------------
df['etiquette'].value_counts().plot(kind='bar', figsize=(8,5), color='skyblue')
plt.title("Répartition des étiquettes dans le dataset")
plt.ylabel("Nombre de réclamations")
plt.show()

# ----------------------------------------------------------
# 3c. Split train/test
# ----------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['etiquette'])
print(f"Taille dataset : {len(df)}, Train : {len(train_df)}, Test : {len(test_df)}")

# ----------------------------------------------------------
# 4. Définition des prompts (3 variantes à tester)
# ----------------------------------------------------------
def build_prompt_v1(text):
    return f"""
Tu es un classificateur de réclamations clients.
Classe la réclamation suivante dans UNE SEULE catégorie parmi :
- Produit, Compte, Paiement, Livraison, Abonnement

Réclamation : "{text}"
Réponds uniquement avec la catégorie.
"""

def build_prompt_v2(text):
    return f"""
Analyse la réclamation suivante et donne uniquement l’étiquette correspondante.
Choix possibles : Produit, Compte, Paiement, Livraison, Abonnement.

Réclamation : {text}
Réponse attendue : une seule étiquette.
"""

def build_prompt_v3(text):
    return f"""
Catégorise la réclamation suivante dans le domaine concerné.
Options : Produit / Compte / Paiement / Livraison / Abonnement.

Réclamation : {text}
Réponds uniquement par le nom exact de l’option.
"""

prompts = {
    'Prompt v1': build_prompt_v1,
    'Prompt v2': build_prompt_v2,
    'Prompt v3': build_prompt_v3,
}

# ----------------------------------------------------------
# 5. Fonction d’appel LLM
# ----------------------------------------------------------
def classify_with_llm(text, build_prompt, model="open-mixtral-8x7b", temperature=0.0):
    """Classe une réclamation en utilisant un prompt spécifique et le modèle Mistral."""
    response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=build_prompt(text))],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# ----------------------------------------------------------
# 6. Test rapide sur 20 exemples (jeu d’entraînement)
# ----------------------------------------------------------
sample_df = train_df.sample(20, random_state=42)

for prompt_name, builder in prompts.items():
    print(f"\n=== {prompt_name} (20 exemples) ===")
    preds = [classify_with_llm(txt, builder) for txt in sample_df['reclamation']]
    print(classification_report(sample_df['etiquette'], preds))

# ----------------------------------------------------------
# 7. Évaluation sur ~1000 exemples du jeu de test
# ----------------------------------------------------------
subset_df = test_df.sample(min(1000, len(test_df)), random_state=42)
results = {}

for prompt_name, builder in prompts.items():
    print(f"\n=== {prompt_name} (1000 exemples) ===")
    start_time = time.time()
    preds = [classify_with_llm(txt, builder) for txt in subset_df['reclamation']]
    end_time = time.time()

    acc = accuracy_score(subset_df['etiquette'], preds)
    f1 = f1_score(subset_df['etiquette'], preds, average='weighted')
    avg_time = (end_time - start_time)/len(subset_df)

    print(f"Accuracy: {acc:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Temps moyen de réponse: {avg_time:.2f} sec")
    print(classification_report(subset_df['etiquette'], preds))

    results[prompt_name] = {'accuracy': acc, 'f1': f1, 'time': avg_time}

    cm = confusion_matrix(subset_df['etiquette'], preds, labels=subset_df['etiquette'].unique())
    disp = ConfusionMatrixDisplay(cm, display_labels=subset_df['etiquette'].unique())
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Matrice de confusion - {prompt_name}")
    plt.show()

# ----------------------------------------------------------
# 8. Implémentation ML traditionnelle (TF-IDF + modèles ML)
# ----------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='french', max_features=5000)
X_train_ml = vectorizer.fit_transform(train_df['reclamation'])
X_test_ml = vectorizer.transform(test_df['reclamation'])

label_encoder = LabelEncoder()
y_train_ml = label_encoder.fit_transform(train_df['etiquette'])
y_test_ml = label_encoder.transform(test_df['etiquette'])

ml_models = {
    'MultinomialNB': MultinomialNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'LinearSVC': LinearSVC()
}
ml_results = {}

for name, model in ml_models.items():
    print(f"\n=== {name} ===")
    start_time = time.time()
    model.fit(X_train_ml, y_train_ml)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test_ml)
    test_time = time.time() - start_time

    acc = accuracy_score(y_test_ml, y_pred)
    f1 = f1_score(y_test_ml, y_pred, average='weighted')

    print(f"Accuracy: {acc:.2f}, F1-score: {f1:.2f}")
    print(f"Temps entraînement: {train_time:.2f}s, Prédiction: {test_time:.2f}s")
    print(classification_report(y_test_ml, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test_ml, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Matrice de confusion - {name}")
    plt.show()

    ml_results[name] = {'accuracy': acc, 'f1': f1, 'train_time': train_time, 'pred_time': test_time}

# ----------------------------------------------------------
# 9. Comparaison LLM vs ML
# ----------------------------------------------------------
llm_metrics_df = pd.DataFrame(results).T[['accuracy','f1']].rename(columns={'accuracy':'Accuracy LLM','f1':'F1 LLM'})
ml_metrics_df = pd.DataFrame(ml_results).T[['accuracy','f1']].rename(columns={'accuracy':'Accuracy ML','f1':'F1 ML'})

comparison_df = pd.concat([llm_metrics_df, ml_metrics_df], axis=1, join='inner')
print("\n=== Comparaison LLM vs ML ===")
print(comparison_df)

comparison_df.plot(kind='bar', figsize=(10,6))
plt.title("Performance comparée : LLM vs ML")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()
