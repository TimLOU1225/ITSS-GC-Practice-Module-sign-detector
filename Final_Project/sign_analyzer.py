import cv2
import numpy as np
from ultralytics import YOLO
from SOD import get_saliency_map
import os
from sign_detector import SignDetector

class WarningSignAnalyzer:
    def __init__(self):
        self.yolo_model = YOLO('yolov8l.pt')
        self.sign_detector = SignDetector()
        self.asnet_model_path = os.path.join('ASNet.h5')
        
    def analyze_image(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {
                'class_name': 'Error: cannot read image',
                'position': None,
                'saliency_score': None,
                'is_reasonable': None,
                'confidence': None,
                'source': 'none'
            }, np.zeros((256, 256), dtype=np.float32)
        
        # 获取显著性图
        try:
            saliency_map_result = get_saliency_map(image_path, self.asnet_model_path)
        except Exception as e:
            return {
                'class_name': f'Error: saliency map failed ({str(e)})',
                'position': None,
                'saliency_score': None,
                'is_reasonable': None,
                'confidence': None,
                'source': 'none'
            }, np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        if saliency_map_result is None or saliency_map_result[0] is None:
            return {
                'class_name': 'Error: saliency map is None',
                'position': None,
                'saliency_score': None,
                'is_reasonable': None,
                'confidence': None,
                'source': 'none'
            }, np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        saliency_map, _ = saliency_map_result
        
        # 将显著性图二值化，降低阈值到0.10
        saliency_binary = (saliency_map > 0.10).astype(np.uint8) * 255
        
        # 找到显著区域的轮廓
        contours, _ = cv2.findContours(saliency_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 合并所有显著区域为一个矩形框
        if contours:
            boxes = [cv2.boundingRect(cnt) for cnt in contours]
            x1 = min(box[0] for box in boxes)
            y1 = min(box[1] for box in boxes)
            x2 = max(box[0] + box[2] for box in boxes)
            y2 = max(box[1] + box[3] for box in boxes)
            height, width = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            # 在显著区域内进行YOLO检测
            roi = image[y1:y2, x1:x2]
            yolo_results = self.yolo_model(roi, conf=0.25)
            # 处理YOLO检测结果
            for result in yolo_results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id]
                    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                    x1_abs = x1 + rx1
                    y1_abs = y1 + ry1
                    x2_abs = x1 + rx2
                    y2_abs = y1 + ry2
                    sign_region = saliency_map[y1_abs:y2_abs, x1_abs:x2_abs]
                    if sign_region.size > 0:
                        avg_saliency = float(np.mean(sign_region))
                        return {
                            'class_name': class_name,
                            'position': (x1_abs, y1_abs, x2_abs, y2_abs),
                            'saliency_score': avg_saliency,
                            'is_reasonable': avg_saliency > 0.10,
                            'confidence': float(box.conf[0]),
                            'source': 'yolo'
                        }, saliency_map
        # 如果没有检测到目标，也返回可视化图片，position为原图全图
        height, width = image.shape[:2]
        return {
            'class_name': 'No warning sign detected',
            'position': (0, 0, width, height),
            'saliency_score': 0.0,
            'is_reasonable': False,
            'confidence': 0.0,
            'source': 'none'
        }, saliency_map

    def visualize_results(self, image_path, result, saliency_map=None):
        # 读取原始图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图片")
        
        # 确保显著性图与原始图像尺寸相同
        if saliency_map is not None:
            saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))
            saliency_map = 1 - saliency_map  # 反转，使红色代表显著性高
        
        # 创建三栏画布
        height, width = img.shape[:2]
        canvas = np.zeros((height, width * 3, 3), dtype=np.uint8)
        canvas[:, :width] = img  # 原图
        canvas[:, width:width*2] = cv2.applyColorMap(
            (saliency_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )  # 只显示热力图
        overlay = cv2.addWeighted(img, 0.7, cv2.applyColorMap(
            (saliency_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        ), 0.3, 0)
        canvas[:, width*2:] = overlay  # 混合图
        
        if result:
            x1, y1, x2, y2 = result['position']
            # 在两个图像上都绘制边界框
            color = (0, 255, 0) if result['is_reasonable'] else (0, 0, 255)
            
            # 在原始图像上绘制
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = f"{result['class_name']} ({result['saliency_score']:.2f})"
            cv2.putText(canvas, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 在显著性图上绘制
            cv2.rectangle(canvas, (x1 + width, y1), (x2 + width, y2), color, 2)
            cv2.putText(canvas, text, (x1 + width, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return canvas

def process_directory(input_dir, output_dir):
    # 创建分析器实例
    analyzer = WarningSignAnalyzer()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"处理图片: {image_file}")
        
        # 分析图片
        result, saliency_map = analyzer.analyze_image(image_path)
        
        if result:
            print(f"检测到警告标识: {result['class_name']}")
            print(f"显著性得分: {result['saliency_score']:.2f}")
            print(f"位置是否合理: {'是' if result['is_reasonable'] else '否'}")
            
            # 可视化结果
            result_img = analyzer.visualize_results(image_path, result, saliency_map)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"result_{image_file}")
            cv2.imwrite(output_path, result_img)
            print(f"结果已保存到: {output_path}")
        else:
            print("未检测到警告标识")

def main():
    # 设置输入输出路径
    input_dir = "test_images/Sign/train/images"
    output_dir = "results"
    
    # 处理目录中的所有图片
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
