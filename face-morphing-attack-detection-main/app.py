
import os
import sys
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from backbones import mixnet_s

MODEL_PATH = os.path.join(APP_DIR, "best_model.pth")

st.set_page_config(
    page_title="Face Morphing Attack Detection",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = mixnet_s(
        embedding_size=128,
        width_scale=1.0,
        gdw_size=1024,
        shuffle=False
    )
    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

st.title("Face Morphing Attack Detection")
st.write(
    "Upload a face image to classify it as a Morph Attack "
    "or Bona Fide using the trained MixNet model."
)

st.warning(
    "Research demonstration only. Predictions may be incorrect "
    "and should not be used as the sole basis for identity verification."
)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing image..."):
            model = load_model()
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                probabilities = F.softmax(model(tensor), dim=1)[0]

            predicted_class = int(probabilities.argmax().item())
            confidence = float(probabilities[predicted_class].item())

        if predicted_class == 0:
            st.error("Prediction: Morph Attack")
        else:
            st.success("Prediction: Bona Fide")

        st.metric("Model confidence score", f"{confidence * 100:.2f}%")
        st.caption(
            "This softmax score is not a calibrated probability "
            "or a guarantee of correctness."
        )
