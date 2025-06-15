import gradio as gr
from transformers import pipeline

# Load emotion model
emotion_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

def detect_emotion(image):
    results = emotion_pipe(image)
    if results:
        top = results[0]
        return f"{top['label']} ({100*top['score']:.1f}%)"
    return "No face detected"

gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="High Accuracy Emotion Detector",
    description="Powered by dima806/facial_emotions_image_detection (~91% accurate)"
).launch(server_name="0.0.0.0", server_port=8080)