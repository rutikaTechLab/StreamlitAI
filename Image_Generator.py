import customtkinter as ctk
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from tkinter import filedialog


# Load Stable Diffusion Model
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model... (First time may take time)")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)
print("Model Loaded Successfully")

generated_image = None



# Generate Image Function
def generate_image():
    global generated_image
    
    prompt = prompt_entry.get()
    
    if prompt == "":
        status_label.configure(text="Please enter prompt")
        return

    status_label.configure(text="Generating Image...")
    app.update()

    image = pipe(prompt).images[0]
    generated_image = image

    # Resize for UI preview
    preview = image.resize((400, 400))
    preview_ctk = ctk.CTkImage(light_image=preview, size=(400, 400))

    image_label.configure(image=preview_ctk)
    image_label.image = preview_ctk

    status_label.configure(text="Image Generated ✅")



# Save Image Function
def save_image():
    global generated_image

    if generated_image is None:
        status_label.configure(text="No image to save")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")]
    )

    if file_path:
        generated_image.save(file_path)
        status_label.configure(text="Image Saved ✅")



# UI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("AI Image Generator")
app.geometry("500x700")

title_label = ctk.CTkLabel(app, text="Hugging Face Image Generator", font=("Arial", 20))
title_label.pack(pady=10)

prompt_entry = ctk.CTkEntry(app, width=400, placeholder_text="Enter your prompt...")
prompt_entry.pack(pady=10)

generate_btn = ctk.CTkButton(app, text="Generate Image", command=generate_image)
generate_btn.pack(pady=10)

image_label = ctk.CTkLabel(app, text="")
image_label.pack(pady=20)

save_btn = ctk.CTkButton(app, text="Save Image", command=save_image)
save_btn.pack(pady=10)

status_label = ctk.CTkLabel(app, text="")
status_label.pack(pady=10)

app.mainloop()
