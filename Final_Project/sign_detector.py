import cv2
import numpy as np

class SignDetector:
    def __init__(self):
        # 定义警告标识的典型颜色范围
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),  # 红色范围
            'yellow': ([20, 100, 100], [30, 255, 255]),  # 黄色范围
            'blue': ([100, 100, 100], [130, 255, 255])   # 蓝色范围
        }
        
    def detect_signs(self, image):
        # 转换到 HSV 颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 存储检测结果
        detections = []
        
        # 对每种颜色进行检测
        for color_name, (lower, upper) in self.color_ranges.items():
            # 创建颜色掩码
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # 形态学操作
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析每个轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 面积阈值
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h
                    
                    # 检查形状是否接近警告标识（通常是圆形或三角形）
                    if 0.8 <= aspect_ratio <= 1.2 or 0.5 <= aspect_ratio <= 2.0:
                        detections.append({
                            'position': (x, y, x+w, y+h),
                            'color': color_name,
                            'confidence': min(area/10000, 1.0)  # 简单的置信度计算
                        })
        
        return detections