# Warning Sign Analysis System

A web-based system for analyzing warning signs in images using computer vision and deep learning techniques.

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open the web interface in your browser:

http://localhost:5000

3. Upload an image containing warning signs:
   - Click the upload button or drag and drop an image
   - Wait for the analysis to complete
   - View the results showing:
     - Detected warning signs
     - Saliency scores
     - Placement evaluation
     - Visual heatmap

## Project Structure
    Final_Project/
    ├── app.py # Flask application
    ├── sign_analyzer.py # Warning sign analysis logic
    ├── templates/ # HTML templates
    │ ├── index.html # Main interface
    │ └── result.html # Result display
    ├── ASNet.h5 # ASNet model
    └── requirements.txt # Dependencies
    
