import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from pathlib import Path
import ssl
import datetime

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Get current year for footer
current_year = datetime.datetime.now().year

# Set page config
st.set_page_config(
    page_title="Pneumonia Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for language if not exists
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Translation dictionary
translations = {
    'en': {
        'title': 'Pneumonia Classifier',
        'subtitle': 'AI detection of pneumonia from chest X-rays',
        'threshold_title': 'Classification Threshold',
        'threshold_info': 'Adjust the classification threshold to balance sensitivity (lower values detect more potential cases) and specificity (higher values reduce false positives).',
        'upload_title': 'Upload Chest X-ray',
        'upload_text': 'Drag and drop your X-ray image',
        'upload_format': 'Supported formats: PNG, JPG, JPEG',
        'uploaded_image_caption': 'Uploaded Chest X-ray',
        'result_header': 'Analysis Result',
        'result_normal': 'NORMAL',
        'result_pneumonia': 'PNEUMONIA DETECTED',
        'confidence_prefix': 'Confidence',
        'normal': 'Normal',
        'pneumonia': 'Pneumonia',
        'threshold_message': 'Classification threshold',
        'additional_info_normal': 'The analysis did not identify characteristic patterns of pneumonia in this image. This result is based solely on the AI algorithm and does not replace clinical evaluation. Additional information such as patient history, symptoms, and complementary exams are essential for correct diagnosis.',
        'additional_info_pneumonia': 'The analysis detected patterns that may indicate pneumonia in this image. This result is based solely on the AI algorithm and serves as an aid to medical interpretation. A complete clinical evaluation is necessary, considering symptoms, patient history, and complementary exams for diagnostic confirmation.',
        'error_processing': 'Error processing image',
        'no_image_title': 'Sample Result Preview',
        'no_image_description': 'Upload a chest X-ray image to get an analysis',
        'developed_by': 'Developed by',
        'about_title': 'About this Model',
        'architecture_label': 'Architecture',
        'accuracy_label': 'Accuracy',
        'validation_set': 'on validation set',
        'disclaimer_title': 'Important Notice',
        'disclaimer_content': 'This tool is for educational and research purposes only. It should not be used for medical diagnosis or clinical decision-making. The results presented do not replace the evaluation of qualified healthcare professionals. Always consult a physician for proper diagnosis and treatment.',
        'how_to_use': 'How to use',
        'image_uploaded': 'Image uploaded successfully!',
        'adjust_threshold': 'Adjust the threshold slider on the left to change the classification sensitivity.',
        'sample_preview': 'Sample Result Preview'
    },
    'pt': {
        'title': 'Classificador de Pneumonia',
        'subtitle': 'Detecção de pneumonia em raios-X do tórax usando IA',
        'threshold_title': 'Limiar de Classificação',
        'threshold_info': 'Ajuste o limiar de classificação para equilibrar sensibilidade (valores mais baixos detectam mais casos potenciais) e especificidade (valores mais altos reduzem falsos positivos).',
        'upload_title': 'Carregar Raio-X do Tórax',
        'upload_text': 'Arraste e solte sua imagem de raio-X',
        'upload_format': 'Formatos suportados: PNG, JPG, JPEG',
        'uploaded_image_caption': 'Raio-X do Tórax Carregado',
        'result_header': 'Resultado da Análise',
        'result_normal': 'NORMAL',
        'result_pneumonia': 'PNEUMONIA DETECTADA',
        'confidence_prefix': 'Confiança',
        'normal': 'Normal',
        'pneumonia': 'Pneumonia',
        'threshold_message': 'Limiar de classificação',
        'additional_info_normal': 'A análise não identificou padrões característicos de pneumonia nesta imagem. Este resultado baseia-se apenas no algoritmo de IA e não substitui a avaliação clínica. Informações adicionais como histórico do paciente, sintomas e exames complementares são essenciais para o diagnóstico correto.',
        'additional_info_pneumonia': 'A análise detectou padrões que podem indicar pneumonia nesta imagem. Este resultado é baseado apenas no algoritmo de IA e serve como auxílio à interpretação médica. Uma avaliação clínica completa é necessária, considerando sintomas, histórico do paciente e exames complementares para confirmação diagnóstica.',
        'error_processing': 'Erro ao processar imagem',
        'no_image_title': 'Exemplo de Visualização de Resultado',
        'no_image_description': 'Carregue uma imagem de raio-X do tórax para obter uma análise',
        'developed_by': 'Desenvolvido por',
        'about_title': 'Sobre este Modelo',
        'architecture_label': 'Arquitetura',
        'accuracy_label': 'Precisão',
        'validation_set': 'no conjunto de validação',
        'disclaimer_title': 'Aviso Importante',
        'disclaimer_content': 'Esta ferramenta é apenas para fins educacionais e de pesquisa. Não deve ser utilizada para diagnóstico médico ou tomada de decisão clínica. Os resultados apresentados não substituem a avaliação de profissionais de saúde qualificados. Sempre consulte um médico para diagnóstico e tratamento adequados.',
        'how_to_use': 'Como usar',
        'image_uploaded': 'Imagem carregada com sucesso!',
        'adjust_threshold': 'Ajuste o controle deslizante de limiar à esquerda para alterar a sensibilidade da classificação.',
        'sample_preview': 'Exemplo de Visualização de Resultado'
    }
}

# Custom CSS for improved aesthetics - dark minimalist theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
        color: #fafafa;
        padding: 0.5rem;
    }
    
    .main-wrapper {
        display: flex;
        flex-direction: column;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #4f8bf9;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 0;
        transition: all 0.2s ease;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background-color: #3b77db;
        transform: translateY(-2px);
    }
    
    .header-container {
        background-color: #262730;
        color: white;
        padding: 0.9rem 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .title-area {
        display: flex;
        align-items: center;
        margin-bottom: 0.25rem;
    }
    
    .title-icon {
        margin-right: 1rem;
    }
    
    .title-text {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .subtitle-text {
        font-size: 0.85rem;
        font-weight: 300;
        opacity: 0.8;
    }
    
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #fafafa;
        display: flex;
        align-items: center;
    }
    
    .section-title-icon {
        margin-right: 0.5rem;
        color: #4f8bf9;
    }
    
    .stProgress > div > div {
        background-image: none;
        border-radius: 10px;
        height: 6px;
    }
    
    .normal-progress .stProgress > div > div {
        background-color: #28a745 !important;
    }
    
    .pneumonia-progress .stProgress > div > div {
        background-color: #dc3545 !important;
    }
    
    .threshold-control {
        background-color: #262730;
        padding: 0.9rem;
        border-radius: 10px;
        margin-bottom: 0.9rem;
        border-left: 3px solid #4f8bf9;
    }
    
    .threshold-title {
        display: flex;
        align-items: center;
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    
    .threshold-icon {
        margin-right: 0.5rem;
    }
    
    .threshold-info {
        background-color: #1e1e2f;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }
    
    .threshold-info p {
        margin: 0.4rem 0;
    }
    
    .result-container {
        background-color: #262730;
        padding: 1.1rem;
        border-radius: 10px;
        margin-top: 0;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .result-header {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #fafafa;
    }
    
    .result-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
    }
    
    .result-normal {
        color: #28a745;
    }
    
    .result-pneumonia {
        color: #dc3545;
    }
    
    .confidence-meter {
        margin: 0.9rem 0;
        background: #1e1e2f;
        padding: 0.9rem;
        border-radius: 8px;
    }
    
    .label-normal {
        color: #28a745;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .label-pneumonia {
        color: #dc3545;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .about-section {
        margin-top: 0.9rem;
    }
    
    .about-title {
        display: flex;
        align-items: center;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: #fafafa;
    }
    
    .about-icon {
        margin-right: 0.5rem;
        color: #4f8bf9;
    }
    
    .model-info {
        margin-bottom: 0.9rem;
    }
    
    .info-label {
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    
    .info-detail {
        background-color: #1e1e2f;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
    }
    
    .disclaimer {
        background-color: #3b2e46;
        padding: 0.75rem;
        border-radius: 6px;
        border-left: 3px solid #dc3545;
        margin-top: 0.6rem;
        font-size: 0.8rem;
    }
    
    .disclaimer-title {
        font-weight: 600;
        color: #ff88a1;
        margin-bottom: 0.4rem;
    }
    
    .footer {
        text-align: center;
        padding: 0.75rem;
        color: #6c757d;
        font-size: 0.8rem;
        margin-top: 1rem;
        border-top: 1px solid #2d303e;
    }
    
    /* Custom upload box */
    .upload-box {
        border: 2px dashed rgba(150, 150, 150, 0.4);
        border-radius: 10px;
        padding: 1.25rem 1rem;
        text-align: center;
        margin-bottom: 0.9rem;
        transition: all 0.2s ease;
        background-color: #1e1e2f;
    }
    
    .upload-box:hover {
        border-color: #4f8bf9;
    }
    
    .upload-text {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        color: #fafafa;
    }
    
    .upload-subtext {
        font-size: 0.8rem;
        color: rgba(200, 200, 200, 0.7);
        margin-bottom: 0.6rem;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #dc3545 !important;
    }
    
    div.stSlider > div > div > div > div {
        background-color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner customization */
    div.stSpinner > div {
        border-top-color: #4f8bf9 !important;
    }
    
    /* Info box styling */
    .stAlert {
        background-color: #1e1e2f !important;
        border: none !important;
        color: #fafafa !important;
        padding: 0.6rem 0.75rem !important;
        font-size: 0.85rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Column gap reduction */
    div.row-widget.stRadio > div {
        flex-direction: row;
        gap: 0.25rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        padding: 0.4rem 0.6rem !important;
    }
    
    .streamlit-expanderContent {
        padding: 0.4rem 0.8rem !important;
    }
    
    /* Block elements - maximize horizontal space */
    .block-container {
        max-width: 95% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    
    /* Elements in columns */
    .stImage img {
        max-width: 100%;
        height: auto;
    }
    
    /* Success box styling */
    .element-container div[data-testid="stSuccess"] {
        padding: 0.4rem 0.6rem !important;
        margin-bottom: 0.6rem !important;
    }
    </style>
""", unsafe_allow_html=True)

class PneumoniaClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # Use MobileNetV3 Small as the base model
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        # Replace the last layer with our own classifier
        in_features = self.model.classifier[0].in_features
        
        # Use a simple classifier with proper initialization
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.Hardswish(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes)
        )
        
        # Initialize the last layer with appropriate weights
        torch.nn.init.xavier_uniform_(self.model.classifier[0].weight)
        torch.nn.init.zeros_(self.model.classifier[0].bias)
        torch.nn.init.xavier_uniform_(self.model.classifier[3].weight)
        torch.nn.init.zeros_(self.model.classifier[3].bias)
    
    def forward(self, x):
        # Add input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")
        
        # Ensure proper input shape (batch_size, 3, 224, 224)
        if x.shape[1:] != torch.Size([3, 224, 224]):
            raise ValueError(f"Expected input shape (B, 3, 224, 224), got {x.shape}")
        
        # Forward pass
        return self.model(x)

@st.cache_resource
def load_model():
    # Initialize model architecture
    model = PneumoniaClassifier()
    
    try:
        # Check if model file exists
        model_path = Path('mobilenet_model.pth')
        if not model_path.exists():
            print(f"Model file not found at {model_path.absolute()}")
            st.error(f"Model file not found at {model_path.absolute()}")
            return None
        
        # Load the saved weights and make sure they match the model structure
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if state_dict keys match model keys
        model_keys = set(model.state_dict().keys())
        load_keys = set(state_dict.keys())
        
        # Print keys for debugging
        print("Model expected keys:", model_keys)
        print("Loaded state dict keys:", load_keys)
        
        # Find missing or unexpected keys
        missing_keys = model_keys - load_keys
        unexpected_keys = load_keys - model_keys
        
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
        
        # Handle case where keys don't match
        if missing_keys or unexpected_keys:
            print("Keys don't match, attempting to load compatible parts...")
            # Create new state dict with only matching keys
            compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            # Load what we can
            model.load_state_dict(compatible_state_dict, strict=False)
        else:
            # Normal loading when keys match
            model.load_state_dict(state_dict)
        
        print("Loaded model successfully")
        
        # Set model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        import traceback
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Error loading model: {str(e)}")
        
        # Create a dummy model for testing
        print("Creating a dummy model for testing...")
        dummy_model = PneumoniaClassifier()
        dummy_model.eval()
        return dummy_model

def preprocess_image(_image):
    """Preprocess the image for model input."""
    # Ensure image is RGB (convert if grayscale)
    if _image.mode != 'RGB':
        _image = _image.convert('RGB')
        
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(_image).unsqueeze(0)

def predict_image(image, threshold=0.5):
    """
    Process and predict an image using the loaded model
    
    Args:
        image: PIL Image object
        threshold: Classification threshold (float between 0 and 1)
        
    Returns:
        prediction_class: String "NORMAL" or "PNEUMONIA"
        confidence: Percentage confidence in the prediction
        pneumonia_probability: Raw probability of pneumonia
    """
    try:
        # Load model
        model = load_model()
        
        if model is None:
            st.error("Model could not be loaded.")
            return "NORMAL", 50.0, 0.5
        
        # Create a copy of the image to avoid modifying the original
        img_copy = image.copy()
        
        # Preprocess image
        processed_image = preprocess_image(img_copy)
        
        # Make prediction
        with torch.no_grad():
            # Reset model to evaluation mode to be sure
            model.eval()
            
            # Force clear cache and ensure fresh computation
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Ensure tensor is on CPU and make a fresh copy
            processed_image = processed_image.clone().detach()
            
            # Generate random ID for this prediction to ensure it's unique
            prediction_id = f"pred_{np.random.randint(10000)}"
            print(f"Processing image {prediction_id}")
            
            # Forward pass
            outputs = model(processed_image)
            
            # Print outputs for debugging
            print(f"Image {prediction_id} - Raw model outputs: {outputs}")
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print(f"Image {prediction_id} - Probabilities: {probabilities}")
            
            # Extract probability for pneumonia (class 1)
            pneumonia_probability = probabilities[0, 1].item()
            normal_probability = probabilities[0, 0].item()
            
            print(f"Image {prediction_id} - Normal: {normal_probability:.4f}, Pneumonia: {pneumonia_probability:.4f}")
        
        # Apply threshold for classification
        prediction_class = "PNEUMONIA" if pneumonia_probability >= threshold else "NORMAL"
        
        # Calculate confidence percentage (always show confidence for the predicted class)
        if prediction_class == "PNEUMONIA":
            confidence = pneumonia_probability * 100
        else:
            confidence = (1 - pneumonia_probability) * 100
        
        print(f"Image {prediction_id} - Final prediction: {prediction_class}, Confidence: {confidence:.2f}%")
        
        return prediction_class, confidence, pneumonia_probability
    
    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Error during prediction: {str(e)}")
        return "ERROR", 0.0, 0.5

def ensure_model_freshness():
    """
    Function to ensure the model is loaded freshly and cache is cleared.
    This helps prevent the model from reusing cached results.
    """
    # Clear cache keys that might be affecting prediction
    for key in list(st.session_state.keys()):
        if 'load_model' in key or 'preprocess_image' in key or 'predict_image' in key:
            del st.session_state[key]
    
    # Force reload of model if needed
    if 'model' in st.session_state:
        del st.session_state['model']
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Ensure fresh model state
    ensure_model_freshness()
    
    # Language selection at the top right
    if 'language' not in st.session_state:
        st.session_state.language = 'en'  # Default language
    
    # Language selector UI with proper buttons instead of JavaScript
    col_lang1, col_lang2 = st.columns([5, 2])
    with col_lang2:
        lang_cols = st.columns(2)
        with lang_cols[0]:
            if st.button("English", use_container_width=True, 
                         type="primary" if st.session_state.language == 'en' else "secondary"):
                st.session_state.language = 'en'
                st.rerun()
        with lang_cols[1]:
            if st.button("Português", use_container_width=True,
                         type="primary" if st.session_state.language == 'pt' else "secondary"):
                st.session_state.language = 'pt'
                st.rerun()
    
    # Get translations for current language
    t = translations[st.session_state.language]
    
    # Header with reduced height
    st.markdown(
        f"""
        <div class="header-container">
            <div style="display: flex; align-items: center;">
                <div>
                    <div class="title-text">{t['title']}</div>
                    <div class="subtitle-text">{t['subtitle']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Create three column layout for main content with better proportions
    col1, col2, col3 = st.columns([0.8, 1, 1.4])
    
    # First column: Threshold settings and model info
    with col1:
        # Classification threshold control
        st.markdown(
            f"""
            <div class="threshold-control">
                <div class="threshold-title">
                    <span class="threshold-icon"></span>
                    {t['threshold_title']}
                </div>
            """,
            unsafe_allow_html=True
        )
        
        # Threshold slider
        threshold = st.slider(
            label=" ", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,
            step=0.01,
            format="%.2f"
        )
        
        # Threshold explanation
        st.markdown(
            f"""
            <div class="threshold-info">
                <p>{t['threshold_info']}</p>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # About model section in first column - using native Streamlit components
        with st.expander(f"{t['about_title']}"):
            st.markdown(f"**{t['architecture_label']}**")
            st.code("MobileNetV3 Small", language="")
            
            st.markdown(f"**{t['accuracy_label']}**")
            st.code(f"~93% {t['validation_set']}", language="")
            
            st.warning(f"**{t['disclaimer_title']}**\n\n{t['disclaimer_content']}")
    
    # Second column: File upload
    with col2:
        # Upload title
        st.markdown(
            f"""
            <div class="section-title">
                <span class="section-title-icon"></span>
                {t['upload_title']}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Custom file uploader with design
        st.markdown(
            f"""
            <div class="upload-box">
                <div class="upload-text">{t['upload_text']}</div>
                <div class="upload-subtext">{t['upload_format']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Actual file uploader (hidden but functional)
        uploaded_file = st.file_uploader(
            label=" ",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        # Add a tutorial section below the uploader when no file is uploaded
        if uploaded_file is None:
            with st.expander(t['how_to_use'], expanded=False):
                if st.session_state.language == 'en':
                    st.markdown("""
                    **Getting Started:**
                    1. Upload a chest X-ray image (JPEG, PNG)
                    2. The model will classify it as Normal or Pneumonia
                    3. Review the confidence scores and probability bars
                    
                    **Tips:**
                    - Use the threshold slider to adjust sensitivity
                    - Higher threshold = fewer false positives
                    - Lower threshold = fewer false negatives
                    """)
                else:
                    st.markdown("""
                    **Começando:**
                    1. Carregue uma imagem de raio-X do tórax (JPEG, PNG)
                    2. O modelo classificará como Normal ou Pneumonia
                    3. Analise os níveis de confiança e barras de probabilidade
                    
                    **Dicas:**
                    - Use o controle deslizante de limiar para ajustar a sensibilidade
                    - Limiar mais alto = menos falsos positivos
                    - Limiar mais baixo = menos falsos negativos
                    """)
        else:
            # Show additional tips when an image is uploaded
            st.success(t['image_uploaded'])
            st.info(t['adjust_threshold'])
    
    # Third column: Results
    with col3:
        # Only show if file is uploaded
        if uploaded_file is not None:
            try:
                # Load and process the image
                image = Image.open(uploaded_file)
                
                # Display the image
                st.image(image, caption=t['uploaded_image_caption'], width=None)
                
                # Add a button to trigger prediction (forces refresh)
                if st.button("Analyze Image", key="analyze_btn", type="primary"):
                    # Reset any previous results
                    if 'prediction' in st.session_state:
                        del st.session_state['prediction']
                        del st.session_state['confidence']
                        del st.session_state['pneumonia_prob']
                    
                    # Explicitly create a new model for this prediction
                    with st.spinner('Analyzing image...'):
                        # Make prediction with the selected model
                        prediction, confidence, pneumonia_prob = predict_image(image, threshold)
                        
                        # Store results in session state
                        st.session_state['prediction'] = prediction
                        st.session_state['confidence'] = confidence
                        st.session_state['pneumonia_prob'] = pneumonia_prob
                        
                        # Force rerun to display results
                        st.rerun()
                
                # Check if we have prediction results to display
                if 'prediction' in st.session_state:
                    prediction = st.session_state['prediction']
                    confidence = st.session_state['confidence']
                    pneumonia_prob = st.session_state['pneumonia_prob']
                    
                    # Determine UI elements based on prediction
                    if prediction == "NORMAL":
                        icon = ""
                        color_class = "result-normal"
                    else:
                        icon = ""
                        color_class = "result-pneumonia"
                    
                    # Display results using native Streamlit components where possible
                    
                    # Create a styled container for the results
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    
                    # Result header and prediction
                    st.markdown(f'<div class="result-header">{t["result_header"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-value {color_class}">{t[f"result_{prediction.lower()}"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'{t["confidence_prefix"]}: <strong>{confidence:.2f}%</strong>', unsafe_allow_html=True)
                    
                    # Start confidence meter
                    st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
                    
                    # Normal probability
                    normal_prob = 1.0 - pneumonia_prob
                    st.markdown(f'<div class="label-normal">{t["normal"]}: {normal_prob*100:.2f}%</div>', unsafe_allow_html=True)
                    st.progress(normal_prob)
                    
                    # Pneumonia probability
                    st.markdown(f'<div class="label-pneumonia" style="margin-top: 10px">{t["pneumonia"]}: {pneumonia_prob*100:.2f}%</div>', unsafe_allow_html=True)
                    st.progress(pneumonia_prob)
                    
                    # End confidence meter
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Threshold information
                    st.markdown(f'{t["threshold_message"]}: <strong>{threshold:.2f}</strong>', unsafe_allow_html=True)
                    
                    # Additional information based on prediction
                    st.markdown(f'<div style="margin-top: 15px">{t[f"additional_info_{prediction.lower()}"]}</div>', unsafe_allow_html=True)
                    
                    # End container
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Show instruction to click analyze button
                    st.info("Click 'Analyze Image' to process the X-ray")
            except Exception as e:
                st.error(f"{t['error_processing']}: {str(e)}")
        else:
            # Display sample X-ray instructions header
            st.markdown(
                f"""
                <div style="background-color: #262730; padding: 1.25rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.75rem; color: #fafafa; display: flex; align-items: center; justify-content: center;">
                        {t['sample_preview']}
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #1e1e2f; border-radius: 8px; padding: 1.5rem 1rem; margin-bottom: 0.75rem;">
                        <div style="font-size: 1.2rem; font-weight: 600; color: #fafafa; margin-bottom: 0.5rem;">{t['result_header']}</div>
                        <div style="font-size: 0.9rem; color: #a0a0a0; max-width: 80%; text-align: center; margin-bottom: 1.2rem;">
                            {t['no_image_description']}
                        </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create sample result preview columns
            sample_cols = st.columns(2)
            
            # Normal column
            with sample_cols[0]:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="font-size: 0.85rem; color: #28a745; font-weight: 600;">{t['normal']}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.progress(0.7)  # 70% normal
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 0.8rem; color: #a0a0a0;">70%</div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Pneumonia column
            with sample_cols[1]:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="font-size: 0.85rem; color: #dc3545; font-weight: 600;">{t['pneumonia']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(0.3)  # 30% pneumonia
                st.markdown(
                    """
                    <div style="text-align: center; font-size: 0.8rem; color: #a0a0a0;">30%</div>
                    """,
                    unsafe_allow_html=True
                )
            
            # End container
            st.markdown(
                f"""
                    </div>
                    <div style="font-size: 0.8rem; color: #a0a0a0;">
                        {t['no_image_description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Footer with github link
    st.markdown(
        f"""
        <div class="footer">
            {t['developed_by']} <a href="https://github.com/alexandre-amaral" target="_blank">Alexandre Amaral</a>   - {current_year} 
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 