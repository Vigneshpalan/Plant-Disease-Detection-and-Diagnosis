import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer


processor = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


model2 = GPT2LMHeadModel.from_pretrained("best_gpt2_plant_disease_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model2.eval()
model = torch.load('model.pth')

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    processed_image = processor(image).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(processed_image)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    labels = model.config.id2label
    predicted_class_label = labels[predicted_class_id]
    
    return predicted_class_label

def generate_text(prompt, max_new_tokens=34, num_beams=5, temperature=0.7, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

    with torch.no_grad():
        output = model2.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    first_entry = generated_text.split('"')[0]
    cleaned_text = generated_text.replace("Treatment:", "").strip()
    return cleaned_text

def handle_prediction(label):
    healthy_labels = [
        'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___healthy', 'Grape___healthy', 'Peach___healthy', 
        'Pepper,_bell___healthy', 'Potato___healthy', 'Raspberry___healthy', 
        'Soybean___healthy', 'Strawberry___healthy', 'Tomato___healthy'
    ]
    
    if label in healthy_labels:
        return "Healthy: This plant is thriving with no visible signs of disease."
    else:
        return generate_text(f"About the disease {label} and its treatment---->>")

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🌱 Plant Disease Prediction and Description")

uploaded_file = st.file_uploader("📂 Choose an image...", type="jpg")

if uploaded_file is not None:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
 
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 Uploaded Image', use_column_width=True)

    predicted_label = predict_image(uploaded_file)
  
  
    st.markdown(f'<div class="message-box bot-message"><strong>🧠 Predicted Disease:</strong> {predicted_label}</div>', unsafe_allow_html=True)
    
   
    result_text = handle_prediction(predicted_label)
   
    st.markdown(f'<div class="generated-text"><strong>📝 Result:</strong><br>{result_text}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("Please upload an image to start.")
