import torch
from torchvision import models
from torch import nn
import numpy as np
from torchsummary import summary
from torchvision import transforms


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

file_name = None
ans = None

def record_pic_location():
    global file_name
    file_name = filedialog.askopenfilename()
    if file_name:
        with Image.open(file_name) as img:
            width, height = img.size
        print(f"[inf] file = {file_name}, width = {width}, height = {height}")
        return  width, height
    else:
        print("[ERROR]")
        return None, None, None

def btn_click():
    global  pic_width, pic_height
    pic_width, pic_height = record_pic_location()

    if pic_height and pic_width:
        pic_width = 300/pic_width
        pic_height = int(pic_height * pic_width)
        pic_width = 300

        img = Image.open(file_name)
        img.thumbnail((pic_width, pic_height))
        photo = ImageTk.PhotoImage(img)

        label_image.config(image=photo)
        label_image.image = photo

        start()
        print(ans)
        text_label.config(text=ans)
        

def start():
    global file_name
    if not file_name:
        print("[ERROR] No file selected")
        return
    
    model = models.vgg16(pretrained=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))   

    filename = file_name
    input_image = Image.open(filename).convert('RGB')

    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device) # 增加一维(笔数)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_batch)

    # 转成机率
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    # 显示最大机率的类别名称
    with open("imagenet.categories", "r") as f:
        # 取第一栏
        categories = [s.strip().split(',')[0] for s in f.readlines()]

    global ans
    ans = categories[torch.argmax(probabilities).item()]
    


# Create the main application window
root = tk.Tk()
root.title("models.vgg16")

# Create a button that will open the file dialog
open_file_button = tk.Button(root, text="照片路徑", command=btn_click)
open_file_button.pack(pady=20)

label_image = tk.Label(root)
label_image.pack()

text_label = tk.Label(root, text="")
text_label.pack()




# Start the Tkinter event loop
root.mainloop()