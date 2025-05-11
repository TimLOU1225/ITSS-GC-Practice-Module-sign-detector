from flask import Flask, render_template, request, jsonify
import os
from sign_analyzer import WarningSignAnalyzer
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize analyzer
analyzer = WarningSignAnalyzer()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def encode_image(image):
    """Encode image to base64 string"""
    if image is None:
        return None
    
    try:
        # Ensure image is in RGB format
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert BGR to RGB if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None
        
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(filepath)
        
        try:
            # Analyze image
            result, saliency_map = analyzer.analyze_image(filepath)
            
            if result:
                # Visualize results
                result_img = analyzer.visualize_results(filepath, result, saliency_map)
                # Encode result image
                result_base64 = encode_image(result_img)
                
                if result_base64 is None:
                    return jsonify({'error': 'Failed to encode result image'}), 500
                
                # 将 NumPy 的布尔类型转换为 Python 的布尔类型
                is_reasonable = bool(result['is_reasonable'])
                
                return jsonify({
                    'success': True,
                    'result': {
                        'class_name': result['class_name'],
                        'saliency_score': float(result['saliency_score']),
                        'is_reasonable': is_reasonable,  # 使用转换后的布尔值
                        'image': result_base64
                    }
                })
            else:
                # If no sign detected, return original image
                original_img = cv2.imread(filepath)
                if original_img is None:
                    return jsonify({'error': 'Failed to read uploaded image'}), 500
                    
                result_base64 = encode_image(original_img)
                if result_base64 is None:
                    return jsonify({'error': 'Failed to encode original image'}), 500
                
                return jsonify({
                    'success': True,
                    'result': {
                        'message': 'No warning sign detected',
                        'image': result_base64
                    }
                })
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing file: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 