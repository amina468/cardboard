import cv2
from PIL import Image, ImageDraw, ImageFont
import io
from torchvision import transforms
import streamlit as st
from detect import model
import gdown

# a file
url = "https://drive.google.com/file/d/10zGK2zRDomXtCeQ3etXziedkLxjRsTwk/view?usp=sharing"
output = "weights.pt"
gdown.download(url, output, fuzzy=True)


def draw_predections(image, boxes, labels, scores, thr):
    # Convert PIL image to ImageDraw object
    if not isinstance(image, Image.Image):
        image = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    width = 4
    nboxes = 0
    
    for boxe, label, score in zip(boxes, labels, scores):
        score = score.item()
        # print(f'{score}, {thr}, {score>thr}')
        if score > thr:
            nboxes += 1
            # print(f'Got in with: {score}, {thr}, {score>thr}')
            # Extract prediction details
            bbox = boxe.cpu().detach().numpy()
            # score = prediction['scores'][0].cpu().detach().numpy()
            category_id = label.cpu().detach().numpy()
            category_name = f'Object {category_id} '

            color = "#008080"
            
            # Define the font properties for text overlay
            font = ImageFont.load_default()
            
            # Draw the bounding box rectangle
            draw.rectangle(tuple(bbox), outline=color, width=width)
            
            # Prepare the text overlay
            text = f"{category_name} ({score:.2f})"
            
            # Calculate the position for text overlay
            # text_width, text_height = draw.textsize(text, font=font) # font = font
            x = bbox[0]
            y = bbox[1]
            
            l, t, r, b = draw.textbbox((x, y), text, font=font)
            h = b - t + width
            # Draw the text overlay
            draw.rectangle((l, t - h, r, b - h), fill=color)
            draw.text((x, y - h), text, fill="white", font=font) # font = font
    
    return image, nboxes

def get_model():
    my_model = model.Model('yolo', 2, 'cpus')
    my_model.load(r'weights.pt')
    return my_model


def main():
    # Get model
    my_model = get_model()

    # Get image
    st.title("COUNT BOXES")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    thr = st.slider('Detection threshold', 0.1, 0.9, 0.5, step=0.1)

    # Run predection on image
    if uploaded_file is not None:
        image = io.BytesIO(uploaded_file.read())
        image = Image.open(image)
        w, h = image.size
        image = image.resize((w * 2, h * 2))
        
        # Perform object detection
        pred = my_model.predict(image)
        
        # Process detection results
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']

        # Display the image with bounding boxes
        image, nboxes = draw_predections(image, boxes, labels, scores, thr)

        st.image(image, channels="BGR")
        if nboxes < 1:
            st.header(f"**Your image doesn't contain any boxes.**")
        elif nboxes < 2:
            st.header(f'**Your image contains :red[{nboxes}] boxe.**')
        else:
            st.header(f'**Your image contains :red[{nboxes}] boxe(s).**')


if __name__ == '__main__':
    main()
