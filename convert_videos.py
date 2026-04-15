import cv2
import os

# 要转换的视频文件
video_files = ['basketball1.avi', 'basketball2.avi']

for video_file in video_files:
    # 构建输入文件路径
    input_path = video_file
    # 构建输出文件路径
    output_path = video_file.replace('.avi', '.mp4')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f'输入文件 {input_path} 不存在')
        continue
    
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f'无法打开视频文件 {input_path}')
        continue
    
    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 定义编码器和创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 读取和写入视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f'视频 {input_path} 已成功转换为 {output_path}')
