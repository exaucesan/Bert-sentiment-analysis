import torch
import gradio as gr
from transformers import AutoTokenizer
from modeling_bert import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google-bert/bert-base-cased"
max_length = 128

id_label = {
    0: "sexual_violence",
    1: "Physical_violence",
    2: "emotional_violence",
    3: "economic_violence",
    4: "Harmful_Traditional_practice",
}

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = Model(model_name=model_name, num_classes=5)
model.load_state_dict(torch.load("violence_model.pt", map_location=device))
model.to(device)
model.eval()


def predict(text):
    ids = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = ids["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    return {id_label[i]: float(probs[i]) for i in range(len(id_label))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Ecris un texte a classer..."),
    outputs=gr.Label(num_top_classes=5),
    title="Classification du type de violence (BERT)",
    description="Modele BERT entraine pour identifier le type de violence decrit dans un texte.",
)

if __name__ == "__main__":
    demo.launch()
