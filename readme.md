# Handwritten Word Recognition Model  
![Handwritten word recognition](https://github.com/user-attachments/assets/8592c044-be31-4057-bc9e-9cbc47c987e2)

## Project Overview  
This project automates the transcription of handwritten text into digital form, leveraging deep learning techniques. It uses the IAM Dataset to train a handwriting recognition model, facilitating seamless data digitization.  

## Dataset Description  
- **Source**: IAM Handwriting Database.  
- **Contents**:  
  - Images of handwritten text lines and paragraphs.  
  - Corresponding text labels for training.  

## Implementation  
1. **Model Training**:  
   - Notebook: `AI_Project.ipynb` contains all training steps and configurations.  
   - Model File: `test.keras` (pretrained model).  
2. **Dashboard**:  
   - `dashboard.py` hosts a Streamlit-based web application for live handwriting recognition.  
   - To run:  
     - Prepare data using the first cell of `AI_Project.ipynb`.  
     - Use Streamlit commands to launch the app and interact with the recognition model.  
3. **Results**:  
   - Achieved 92% accuracy on the test dataset, demonstrating robust recognition capabilities.  

## How to Run  
1. Clone the repository.  
2. Install the required dependencies from `requirements.txt`.  
3. Run `dashboard.py` alongside `test.keras` to launch the web application.  

---
