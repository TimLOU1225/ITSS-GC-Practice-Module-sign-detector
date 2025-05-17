from __future__ import division
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
import os
import sys
import numpy as np
from config import *
from utilities import postprocess_predictions
from update_models import sam_vgg
import random
from imageio import imread
from PIL import Image
import cv2
from ultralytics import YOLO


def get_test(data):       
    Xims_224 = np.zeros((1, 224, 224, 3))
    img = imread(data['image'])
    img_name = os.path.basename(data['image'])
    gaussian = np.zeros((1, 14, 14, nb_gaussian))
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB
    r_img = tf.image.resize(img, (224, 224)).numpy()
    r_img -= np.array([img_channel_mean[0], img_channel_mean[1], img_channel_mean[2]])
    r_img = r_img[:, :, ::-1]  # Convert RGB to BGR
    Xims_224[0, :] = r_img
    return [Xims_224, gaussian], r_img, img_name


def prediction():
    if len(sys.argv) != 1:
        raise NotImplementedError
    else:
        seed = 7
        random.seed(seed)
        test_data = []

        testing_images = [datasest_path + f for f in os.listdir(datasest_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]
        testing_images.sort()

        for image in testing_images:
            annotation_data = {'image': image}
            test_data.append(annotation_data)

        phase = 'test'
        if phase == "test":
            x = Input(batch_shape=(1, 224, 224, 3))
            x_maps = Input(batch_shape=(1, 14, 14, nb_gaussian))
            m = Model(inputs=[x, x_maps], outputs=sam_vgg([x, x_maps]))

            print("Loading weights")
            m.load_weights('ASNet.h5')
            print("Making prediction")

            saliency_output = './ASNet/saliency_predictions/'
            if not os.path.exists(saliency_output):
                os.makedirs(saliency_output)

            for data in test_data:
                Ximg, original_image, img_name = get_test(data)
                predictions = m.predict(Ximg, batch_size=1)
                # 修正使用 original_image.size 获取宽度和高度的方式
                width, height = original_image.shape[1], original_image.shape[0]
                res_saliency = postprocess_predictions(predictions[6][0, :, :, 0], height, width)
                Image.fromarray(res_saliency.astype('uint8')).save(saliency_output + '%s.png' % img_name[0:-4])
                # m.reset_states()
        else:
            raise NotImplementedError

        print("Predicton Done")


def get_saliency_map(image_path, model_path):
    # 读取图像
    img = imread(image_path)
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    
    # 获取原始尺寸
    height, width = img.shape[:2]
    
    # 预处理
    Xims_224 = np.zeros((1, 224, 224, 3))
    gaussian = np.zeros((1, 14, 14, nb_gaussian))
    
    # 调整大小
    r_img = tf.image.resize(img, (224, 224)).numpy()
    r_img -= np.array([img_channel_mean[0], img_channel_mean[1], img_channel_mean[2]])
    r_img = r_img[:, :, ::-1]  # RGB to BGR
    Xims_224[0, :] = r_img
    
    # 加载模型
    x = Input(batch_shape=(1, 224, 224, 3))
    x_maps = Input(batch_shape=(1, 14, 14, nb_gaussian))
    m = Model(inputs=[x, x_maps], outputs=sam_vgg([x, x_maps]))
    m.load_weights(model_path)
    
    # 预测
    predictions = m.predict([Xims_224, gaussian], batch_size=1)
    
    # 后处理
    saliency_map = postprocess_predictions(predictions[6][0, :, :, 0], height, width)
    
    # 归一化到 0-1 范围
    saliency_map = saliency_map.astype(np.float32) / 255.0
    
    return saliency_map, img


class WarningSignAnalyzer:
    def __init__(self):
        # 初始化 YOLO 模型
        self.yolo_model = YOLO('yolov8n.pt')
        # 定义警告标识类别
        self.warning_classes = ['stop sign', 'traffic light', 'warning sign']
        # ASNet 模型路径
        self.asnet_model_path = os.path.join('Final_Project', 'ASNet.h5')
        
    def analyze_image(self, image_path):
        # 1. 获取显著性图
        saliency_map, original_image = get_saliency_map(image_path, self.asnet_model_path)
        
        # 2. 使用 YOLO 检测警告标识
        results = self.yolo_model(original_image)
        
        # 3. 分析结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                
                if class_name in self.warning_classes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 计算该区域的显著性值
                    sign_region = saliency_map[y1:y2, x1:x2]
                    avg_saliency = np.mean(sign_region)
                    
                    # 判断位置是否合理
                    is_reasonable = avg_saliency > 0.3
                    
                    return {
                        'class_name': class_name,
                        'position': (x1, y1, x2, y2),
                        'saliency_score': float(avg_saliency),
                        'is_reasonable': bool(is_reasonable),
                        'confidence': float(box.conf[0]),
                        'source': 'yolo'
                    }, saliency_map
        
        # 检测不到时兜底返回
        height, width = original_image.shape[:2]
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
        
        if result['class_name'] != 'No warning sign detected':
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

    # 所有处理逻辑完成后，再删除文件
    try:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error removing file: {str(e)}")
    except Exception as e:
        print(f"Error removing file: {str(e)}")

def main():
    # 设置输入输出路径
    input_dir = "test_images/Sign/train/images"
    output_dir = "Final_Project/results"
    
    # 处理目录中的所有图片
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()