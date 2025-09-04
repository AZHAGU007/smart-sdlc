import gradio as gr
from transformers import pipeline

# Load Hugging Face pipeline
pipe = pipeline("fill-mask", model="web3se/SmartBERT-v3")

def predict(text):
    try:
        results = pipe(text)
        # Format results for UI
        formatted = [f"{r['sequence']} (score: {r['score']:.4f})" for r in results]
        return "\n".join(formatted)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter text with [MASK]", placeholder="The capital of France is [MASK]."),
    outputs=gr.Textbox(label="Predictions"),
    title="SmartBERT-v3 Masked Language Model",
    description="Enter a sentence containing [MASK] and SmartBERT will predict possible replacements."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
