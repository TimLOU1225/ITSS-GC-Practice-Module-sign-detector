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

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化分析器
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
        print("收到分析请求")
        print(f"请求文件: {request.files}")
        print(f"请求表单: {request.form}")
        
        if 'file' not in request.files:
            print("错误：没有找到文件")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        print(f"文件名: {file.filename}")
        
        if file.filename == '':
            print("错误：文件名为空")
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"错误：不支持的文件类型: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image'}), 400
            
        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        print(f"保存文件到: {filepath}")
        file.save(filepath)
        
        try:
            # 创建分析器实例
            analyzer = WarningSignAnalyzer()
            
            # 分析图片
            print("开始分析图片")
            result_tuple = analyzer.analyze_image(filepath)
            if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                print(f"错误：analyze_image 返回值格式不正确: {result_tuple}")
                raise ValueError("analyze_image did not return a (result, saliency_map) tuple")
                
            result, saliency_map = result_tuple
            
            if result is None:
                print("错误：分析结果为 None")
                raise ValueError("Analysis result is None")
                
            # 可视化结果
            print("生成可视化结果")
            result_img = analyzer.visualize_results(filepath, result, saliency_map)
            if result_img is None:
                print("错误：无法生成可视化结果")
                raise ValueError("Failed to generate visualization")
                
            # 编码结果图片
            print("编码结果图片")
            result_base64 = encode_image(result_img)
            if result_base64 is None:
                print("错误：无法编码结果图片")
                raise ValueError("Failed to encode result image")
            
            # 将 NumPy 的布尔类型转换为 Python 的布尔类型
            is_reasonable = bool(result['is_reasonable'])
            
            print("分析完成，返回结果")
            return jsonify({
                'success': True,
                'result': {
                    'class_name': result['class_name'],
                    'saliency_score': float(result['saliency_score']),
                    'is_reasonable': is_reasonable,
                    'image': result_base64
                }
            })
            
        except Exception as e:
            print(f"分析过程中出错: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"已删除临时文件: {filepath}")
            except Exception as e:
                print(f"清理文件时出错: {str(e)}")
                
    except Exception as e:
        print(f"请求处理过程中出错: {str(e)}")
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 