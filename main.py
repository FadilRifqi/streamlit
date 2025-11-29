import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm
import cv2
import torch.serialization

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# ====== Load model (modify to match your checkpoint) ======
from model import FaceModel  # your FaceModel class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    ckpt = torch.load("best_model_efficientnet_b0.pth", map_location=device, weights_only=False)
    model = FaceModel(
        ckpt["backbone"],
        ckpt["embedding_size"],
        len(ckpt["label_classes"]),
        pretrained=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, ckpt["label_classes"]

model, classes = load_model()

# ===== MTCNN untuk face detection =====
mtcnn = MTCNN(image_size=224, margin=0, device=device)

# ===== Preprocessing (same as validation) =====
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

st.title("🧑 Face Recognition App (EfficientNet + ArcFace)")
st.write("Upload foto wajah untuk mengenali identitas.")

uploaded_file = st.file_uploader("Upload file JPG/PNG", type=["jpg","jpeg","png"])

def predict(img):
    # Step 1: Face detection
    face = mtcnn(img)

    if face is None:
        return None, "Face not detected"

    # MTCNN output can be:
    # (1, 3, 224, 224)  OR  (3, 224, 224)
    face = face.cpu()

    if face.dim() == 3:
        # shape: (3,224,224) → jadikan batch 1
        face = face.unsqueeze(0)

    # Now face = (1,3,224,224)
    face_np = face.squeeze(0).permute(1,2,0).numpy()  # (224,224,3)
    face_np = (face_np * 255).astype("uint8")
    face_pil = Image.fromarray(face_np)

    # Preprocess
    inp = val_transform(face_pil).unsqueeze(0).to(device)

    # Embedding
    with torch.no_grad():
        emb = model(inp)   # returns embedding
    emb = emb.cpu().numpy()

    # ArcFace similarity to class centers
    weight = model.arcface.weight.data.cpu().numpy()
    weight_norm = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    emb_norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    sims = emb_norm.dot(weight_norm.T)
    idx = sims.argmax()
    conf = sims[0, idx]

    return classes[idx], float(conf)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    with st.spinner("Detecting..."):
        label, conf = predict(img)

    if label is None:
        st.error("Wajah tidak terdeteksi.")
    else:
        st.success(f"Identitas: **{label}** (Confidence: {conf:.4f})")
