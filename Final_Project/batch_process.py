import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from SOD import get_saliency_map
from sign_analyzer import WarningSignAnalyzer

def process_images(input_dir, output_dir, csv_path):
    # 创建分析器实例
    analyzer = WarningSignAnalyzer()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备CSV数据
    results_data = []
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"开始处理 {len(image_files)} 张图片...")
    
    for image_file in image_files:
        try:
            image_path = os.path.join(input_dir, image_file)
            print(f"\n处理图片: {image_file}")
            
            # 分析图片
            result, saliency_map = analyzer.analyze_image(image_path)
            
            if result:
                # 保存检测结果图片
                result_img = analyzer.visualize_results(image_path, result, saliency_map)
                output_path = os.path.join(output_dir, f"result_{image_file}")
                cv2.imwrite(output_path, result_img)
                
                # 记录结果
                results_data.append({
                    'image_name': image_file,
                    'detected_class': result['class_name'],
                    'saliency_score': result['saliency_score'],
                    'is_reasonable': result['is_reasonable'],
                    'confidence': result['confidence'],
                    'detection_source': result['source'],
                    'position': f"({result['position'][0]}, {result['position'][1]}, {result['position'][2]}, {result['position'][3]})",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                print(f"检测到: {result['class_name']}")
                print(f"显著性得分: {result['saliency_score']:.2f}")
                print(f"位置是否合理: {'是' if result['is_reasonable'] else '否'}")
                print(f"结果已保存到: {output_path}")
            else:
                # 记录未检测到结果的情况
                results_data.append({
                    'image_name': image_file,
                    'detected_class': 'None',
                    'saliency_score': 0.0,
                    'is_reasonable': False,
                    'confidence': 0.0,
                    'detection_source': 'None',
                    'position': 'None',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                print("未检测到任何标识")
                
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {str(e)}")
            # 记录错误情况
            results_data.append({
                'image_name': image_file,
                'detected_class': 'Error',
                'saliency_score': 0.0,
                'is_reasonable': False,
                'confidence': 0.0,
                'detection_source': 'Error',
                'position': 'Error',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # 保存CSV文件
    df = pd.DataFrame(results_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n处理完成！结果已保存到: {csv_path}")

def main():
    # 设置输入输出路径
    input_dir = "test_images"  # 输入图片目录
    output_dir = "results"     # 输出结果目录
    csv_path = "detection_results.csv"  # CSV文件路径
    
    # 处理图片
    process_images(input_dir, output_dir, csv_path)

if __name__ == "__main__":
    main() 