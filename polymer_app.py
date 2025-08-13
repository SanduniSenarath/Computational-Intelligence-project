import streamlit as st
import pickle
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

st.set_page_config(page_title="PolyProp Predictor", layout="centered")

# ===== Load models =====
try:
    with open("trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    if not isinstance(models, dict):
        st.error("Loaded model is not in correct dict format")
        models = {}
except Exception as e:
    st.error(f"Error loading models: {e}")
    models = {}

# ===== Detect feature size =====
feature_size = 2064  # fallback
if models:
    first_model = next(iter(models.values()))
    if hasattr(first_model, "n_features_in_"):
        feature_size = first_model.n_features_in_

# ===== Featurization =====
def smiles_to_vec(smi, rad=2, bits=feature_size):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, rad, nBits=bits)
    arr = np.zeros(bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ===== Title =====
st.markdown('<h1 style="color:#0073e6; text-align:center;">PolyProp Predictor</h1>', unsafe_allow_html=True)

# ===== About Section =====
with st.expander("About PolyProp Predictor", expanded=True):
    st.markdown("""
PolyProp Predictor predicts key physical properties of polymers based on their SMILES codes.

- Developed with **Streamlit & RDKit**  
- Models trained on polymer data  
- Instant predictions with a clean UI  

Designed for researchers and polymer enthusiasts.
""")

# ===== Prediction Section =====
st.markdown('<h2 style="color:#0073e6;">Predict Polymer Properties</h2>', unsafe_allow_html=True)
st.markdown("Enter a polymer's **SMILES string** below:")

# Input
smi = st.text_input("Polymer SMILES", value="CCO").strip()
props = list(models.keys()) if models else []

show_all = st.checkbox("Show all properties")
selected_prop = None if show_all else st.selectbox("Select property", props)

# Prediction
if st.button("Predict"):
    if not smi:
        st.error("Please enter a SMILES string.")
    else:
        feat = smiles_to_vec(smi)
        if feat is None:
            st.error("Invalid SMILES string.")
        else:
            if show_all:
                st.markdown("### Predictions:")
                for prop in props:
                    val = models[prop].predict(feat.reshape(1, -1))[0]
                    st.success(f"{prop}: {val:.4f}")
            else:
                if selected_prop not in models:
                    st.error(f"No model available for {selected_prop}.")
                else:
                    val = models[selected_prop].predict(feat.reshape(1, -1))[0]
                    st.success(f"{selected_prop}: {val:.4f}")
