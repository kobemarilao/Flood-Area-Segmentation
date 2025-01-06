
import streamlit as st
import streamlit_antd_components as sac
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import segmentation_models_pytorch as smp
import cv2 
import collections.abc as container_abcs

@st.cache_resource
def load_model():
    model = torch.load("D:/Downloads/ads/road.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache_resource 
def load_model2():
    model2 = smp.Unet(
        encoder_name="resnet50",  # 
        encoder_weights="imagenet",  
        in_channels=3,  
        classes=1  
    )
    model2.load_state_dict(torch.load("D:/Downloads/ads/flood.pth", map_location=torch.device('cpu')))
    model2.to('cpu')
    model2.eval()
    return model2

model = load_model()
model2 = load_model2()

select_classes = ['background', 'road']

def generate_prediction(image_path, model2):
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    input_tensor = transform(image).unsqueeze(0).to('cpu')  # Add batch dimension and send to CPU

    # Perform inference
    with torch.no_grad():
        pred = model2(input_tensor)
        pred = torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dims, move to CPU
        pred = np.where(pred < 0.5, 0, 1).astype(np.int16)  # Thresholding for binary mask as per the model

    # Create a color map for the binary mask
    color_mask = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.uint8)
    color_mask[pred == 1] = [255, 255, 0, 255]  # Red for the foreground (change to the color you want)
    color_mask[pred == 0] = [0, 0, 0, 0]    # Black for the background

    # Convert the colored prediction to an image
    pred_img = Image.fromarray(color_mask)  # Convert color mask to image
    return pred_img


# Function to generate heatmap #for model 1
def generate_heatmap(image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)  # Get model output
        if output.ndim == 4 and output.shape[1] > 1:
            # If model outputs multiple channels, take the mean across channels
            road_channel_idx = select_classes.index('road')
            heatmap = output.squeeze(0)[road_channel_idx].numpy()
        else:
            # Single-channel output
            heatmap = output.squeeze().numpy()

    # Normalize heatmap for visualization
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Convert heatmap to an RGBA image with transparency
    color_map = plt.cm.cool 
    rgba_image = color_map(heatmap)  
    rgba_image = (rgba_image * 255).astype(np.uint8) 

    alpha_threshold = 15  
    rgba_image[:, :, 3] = (heatmap * 255 > alpha_threshold).astype(np.uint8) * 255

    return Image.fromarray(rgba_image, 'RGBA')
    #return heatmap


# Add a menu to the sidebar
with st.sidebar:
    st.sidebar.image("https://scontent.fcgy2-4.fna.fbcdn.net/v/t1.15752-9/462581375_2549919758534783_8988233902342983983_n.png?_nc_cat=105&ccb=1-7&_nc_sid=9f807c&_nc_eui2=AeHS4MVac3x8yuqbSo75JP8TfeBEpLtCg5N94ESku0KDk_sbDinvn5cesW9NiWNqz9-7UtlfPOjvw3qadmT-gMWJ&_nc_ohc=VYv1zyn8wugQ7kNvgEn4sOL&_nc_zt=23&_nc_ht=scontent.fcgy2-4.fna&oh=03_Q7cD1QGdB_oTcKjiXgSLFNxzZGtyy2gXQqisIYIFTt_6wwi2yw&oe=67806700", use_column_width=True) 

    menu_selection = sac.menu(
        [
            sac.MenuItem('Home', icon='house-fill'),
            sac.MenuItem(type='divider'),
            sac.MenuItem('About Us', icon='box-fill'),
        ],
        format_func='upper',
        variant='left-bar',
        open_all=True,
    )

# Display content based on the selected menu item
if menu_selection == 'Home':

    # Create two columns for uploading the pre-flooded and post-flooded images
    hom1, hom2 = st.columns(2)

    # Column 1: Pre-Flooded Image Upload
    with hom1:
        st.title('Upload Pre-Flooded Photo.')
        uploaded_file_pre = st.file_uploader(
            label=" ",  # Using a space as a label to hide it
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=False,
            key="pre_flooded"  # Unique key 
        )
        if uploaded_file_pre is not None:
            image_pre = Image.open(uploaded_file_pre)
            st.image(image_pre, caption="Pre-Flooded Image")

            # Generate and display heatmap
            st.title("Pre-Flooded Roads")
            heatmap = generate_heatmap(image_pre, model)
            st.image(heatmap, caption="Mapped Roads", use_column_width=True)
            

    # Column 2: Post-Flooded Image Upload
    with hom2:
        st.title('Upload Post-Flooded Photo.')
        uploaded_file_post = st.file_uploader(
            label=" ",  # Using a space as a label to hide it
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=False,
            key="post_flooded"  # Unique key 
        )
        if uploaded_file_post is not None:
            image_post = Image.open(uploaded_file_post)
            st.image(image_post, caption="Post-Flooded Image")

            # Generate predicted flood mask
            st.title("Identified Flood Areas")
            pred_img = generate_prediction(uploaded_file_post, model2)
            st.image(pred_img, caption="Identified Flooded Areas", use_column_width=True)
            
    # Merging both heatmap, predicted image, and post-flooded image
    if uploaded_file_pre is not None and uploaded_file_post is not None:
        st.title("Merged Results")

        # Generate heatmap for pre-flooded image
        heatmap = generate_heatmap(image_pre, model)
        
        # Generate predicted flood mask for post-flooded image
        pred_img = generate_prediction(uploaded_file_post, model2)
        
        # Resize both heatmap and pred_img to 512x512
        heatmap_resized = heatmap.resize((512, 512))
        pred_img_resized = pred_img.resize((512, 512))
        
        # Convert both images to NumPy arrays
        heatmap_array = np.array(heatmap_resized)
        pred_img_array = np.array(pred_img_resized)
        
        # Resize the post-flooded image to 512x512
        post_flooded_resized = image_post.resize((512, 512))
        
        # Convert the post-flooded image to a NumPy array
        post_flooded_array = np.array(post_flooded_resized)
        
        # Ensure all images (heatmap, pred_img, and post-flooded) are RGBA
        if heatmap_array.shape[2] != 4:
            heatmap_array = cv2.cvtColor(heatmap_array, cv2.COLOR_RGB2BGRA)
        if pred_img_array.shape[2] != 4:
            pred_img_array = cv2.cvtColor(pred_img_array, cv2.COLOR_RGB2BGRA)
        if post_flooded_array.shape[2] != 4:
            post_flooded_array = cv2.cvtColor(post_flooded_array, cv2.COLOR_RGB2BGRA)
        
        # Apply cv2.addWeighted to overlay the post-flooded image with heatmap and predicted mask
        overlay_img = cv2.addWeighted(post_flooded_array, 1, heatmap_array, 1, 0)
        final_image = cv2.addWeighted(overlay_img, 1, pred_img_array, 1, 0)

        # Convert the final image back to PIL for display in Streamlit
        final_image_pil = Image.fromarray(final_image)

        # Display the final merged result
        st.image(final_image_pil, caption="Final Merged Result", use_column_width=True)


elif menu_selection == 'About Us':
    st.title('About Us')
    st.markdown("""
    **Welcome to the Flood Zones Identifier**
    
    Your essential tool for navigating through flood-prone areas in the Philippines. Our platform is designed to provide real-time information about areas affected by flooding, helping residents, commuters, and emergency services make informed decisions.

    **Our Mission**
    We aim to enhance public safety and minimize disruptions caused by natural disasters like flooding. By delivering precise and user-friendly tools, we empower individuals and communities to stay safe while planning their daily routes.

    **What We Do**
    The Flood Zones Identifier offers:

    - **Flood Mapping**: Visual representation of flooded zones in your area, such as Bajada, Davao City.
    - **Street Listings**: Highlighting affected roads for better situational awareness.
    - **Route Optimization**: Providing alternative routes to help you navigate safely to your destination.

    Whether you're traveling from Downtown Davao to Catalunan Pequeño or any other location, our system ensures you are well-informed and prepared. Stay safe and plan ahead with the **Flood Zones Identifier**.
    """)

    st.markdown("""
        <style>
            .stCaption {
                text-align: center;
                display: block;
                width: 100%;
                margin-top: 3px;
            }
        </style>
        """, unsafe_allow_html=True)

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # In each column, display an image and its name
    with col1:
        st.image("https://scontent.fcgy2-1.fna.fbcdn.net/v/t1.15752-9/470051793_1106852564161664_4812549164318226772_n.png?_nc_cat=106&ccb=1-7&_nc_sid=9f807c&_nc_eui2=AeHOFjzUgi19WE2VoYtrCnzO_PM9kjUhvZX88z2SNSG9lQMz_7neWAAIKpqS_c-61oIM94_X5vX9i4Y4qupaKjfr&_nc_ohc=aOeHFJ4FIrwQ7kNvgF4zA3v&_nc_zt=23&_nc_ht=scontent.fcgy2-1.fna&oh=03_Q7cD1QFVBWLFH-ZsZP4iv-k8aVe1a4vPUW5MB8LkvfVUFZTl5A&oe=678071E5")  # Replace with your image path
        st.markdown('<p style="text-align: center; margin-bottom: 0;">Yza J. Prochina</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; margin-top: 0;">yjprochina00114@usep.edu.ph</p>', unsafe_allow_html=True)

    with col2:
        st.image("https://scontent.fcgy2-3.fna.fbcdn.net/v/t1.15752-9/462578127_2056613758134848_4439941361628263564_n.png?_nc_cat=111&ccb=1-7&_nc_sid=9f807c&_nc_eui2=AeGpqEMEK0dx9PuDg9XeEISsQh3SAF2oxPVCHdIAXajE9c43yD33tM84d6EQ-Wwc4xhGAj8PemS8bC3pcZMR1tf4&_nc_ohc=soF8WApwhrwQ7kNvgERp3NM&_nc_zt=23&_nc_ht=scontent.fcgy2-3.fna&oh=03_Q7cD1QFR_QiJZNH97JE4YsQ3V8NCdIJevOx3AlzEZYVW-6gCHA&oe=678092DB")  # Replace with your image path
        st.markdown('<p style="text-align: center; margin-bottom: 0;">Sharill Mives B. Castillo</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; margin-top: 0;">smbcastillo00092@usep.edu.ph</p>', unsafe_allow_html=True)

    with col3:
        st.image("https://scontent.fcgy2-2.fna.fbcdn.net/v/t1.15752-9/462566969_2048769672260527_3106865969120221634_n.png?_nc_cat=103&ccb=1-7&_nc_sid=9f807c&_nc_eui2=AeGDeKrANqSWjG9ZYH5tKW0vai9tgUnM2P9qL22BSczY_1gHbQYgDrgYPogIb1Gwa6m3Ky0RvqAiQZ0pqiUYgW9-&_nc_ohc=sRdJgTrC3EoQ7kNvgEZAcLq&_nc_zt=23&_nc_ht=scontent.fcgy2-2.fna&oh=03_Q7cD1QEexa_7keBK-HlHNEMHf4M7WX2CCevJBWgphfkgytXexw&oe=67808BB5")  # Replace with your image path
        st.markdown('<p style="text-align: center; margin-bottom: 0;">Christian Kobe A. Marilao</p>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; margin-top: 0;">ckamarilao@usep.edu.ph</p>', unsafe_allow_html=True)


    st.write("\n")
    st.markdown("""
    The dashboard was developed by a dedicated team of 3, namely, Sharil Mives Castillo, Christian Kobe Marilao, and Yza Prochina. 
    We are 3rd year Bachelor of Science in Computer Science Major in Data Science (BSCS-DS) students at University of Southeastern Philippines. 
    Combining each other's expertise, we designed and implemented a dashboard to address the specific challenges of flood management and navigation safety all over the Philippines. 
    This dashboard is for the developer’s Learning Evidence in their Applied Data Science subject/course.
""")
