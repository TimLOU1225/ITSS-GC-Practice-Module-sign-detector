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
            raise ValueError("无法读取图片")
            
        # 获取显著性图
        saliency_map, _ = get_saliency_map(image_path, self.asnet_model_path)
        
        # 使用 YOLO 检测
        yolo_results = self.yolo_model(image, conf=0.25)
        
        # 使用颜色和形状检测
        color_results = self.sign_detector.detect_signs(image)
        
        # 合并结果
        all_detections = []
        
        # 处理 YOLO 结果
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]  # 获取类别名称
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 确保区域有效
                if x1 < x2 and y1 < y2:
                    sign_region = saliency_map[y1:y2, x1:x2]
                    if sign_region.size > 0:  # 确保区域不为空
                        avg_saliency = np.mean(sign_region)
                        
                        all_detections.append({
                            'class_name': class_name,  # 添加类别名称
                            'position': (x1, y1, x2, y2),
                            'saliency_score': avg_saliency,
                            'is_reasonable': avg_saliency > 0.3,
                            'confidence': float(box.conf[0]),
                            'source': 'yolo'
                        })
        
        # 处理颜色检测结果
        for detection in color_results:
            x1, y1, x2, y2 = detection['position']
            
            # 确保区域有效
            if x1 < x2 and y1 < y2:
                sign_region = saliency_map[y1:y2, x1:x2]
                if sign_region.size > 0:  # 确保区域不为空
                    avg_saliency = np.mean(sign_region)
                    
                    all_detections.append({
                        'class_name': f"warning_sign_{detection['color']}",  # 添加类别名称
                        'position': (x1, y1, x2, y2),
                        'saliency_score': avg_saliency,
                        'is_reasonable': avg_saliency > 0.3,
                        'confidence': detection['confidence'],
                        'source': 'color'
                    })
        
        # 选择最显著的结果
        if all_detections:
            best_detection = max(all_detections, key=lambda x: x['saliency_score'])
            return best_detection, saliency_map
            
        return None, saliency_map

    def visualize_results(self, image_path, result, saliency_map=None):
        # 读取原始图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图片")
        
        # 确保显著性图与原始图像尺寸相同
        if saliency_map is not None:
            saliency_map = cv2.resize(saliency_map, (img.shape[1], img.shape[0]))
        
        # 创建一个组合显示的画布
        height, width = img.shape[:2]
        canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # 将原始图像放在左侧
        canvas[:, :width] = img
        
        # 如果有显著性图，将其转换为热力图并放在右侧
        if saliency_map is not None:
            # 将显著性图转换为热力图
            saliency_heatmap = cv2.applyColorMap(
                (saliency_map * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            # 将热力图与原始图像混合
            overlay = cv2.addWeighted(img, 0.7, saliency_heatmap, 0.3, 0)
            canvas[:, width:] = overlay
        
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
