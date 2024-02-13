import streamlit as st
from streamlit_ketcher import st_ketcher
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

# Import the predict function from predict.py
from predict import predict

# Function to preprocess molecule for prediction
def prepare_molecule(smiles_input):
    molecule = Chem.MolFromSmiles(smiles_input)
    df = pd.DataFrame({'SMILES': [smiles_input]})
    return molecule, df

# Main Streamlit code
selected = st.sidebar.selectbox("Select an Option", ["Single Molecule Prediction"])

if selected == "Single Molecule Prediction":
    st.subheader("Input or Draw a Molecule")
    agree_draw_smiles = st.checkbox("Draw chemical structure")

    if agree_draw_smiles:
        # Drawing functionality
        smile_code = st_ketcher()
        smiles_input = smile_code  # Use the drawn molecule's SMILES
        st.markdown(f"SMILES: {smiles_input}")
    else:
        # Text input for SMILES
        smiles_input = st.text_input("Input SMILES", key="text")

    # Process the SMILES input or drawn molecule
    if smiles_input:
        try:
            molecule, df = prepare_molecule(smiles_input)
            if molecule:
                # Show the molecule's structure
                img = Draw.MolToImage(molecule)
                st.image(img, caption='Chemical structure', use_column_width=True)

                # Make prediction
                prediction = predict(smiles_input)
                st.write(prediction)

        except Exception as e:
            st.error(f"Error: {str(e)}")
