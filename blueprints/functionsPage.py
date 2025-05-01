from PIL import Image
from flask import Blueprint, render_template, request, jsonify
import tensorflow as tf
import cv2
from static.methods.imageProcess import *
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
import os
import traceback

bp = Blueprint("functionsPage", __name__, url_prefix="/functions")
UPLOAD_FOLDER = 'static/DLFile/upload_folder'
OUTPUT_FOLDER = 'static/DLFile/output_folder'
OUTPUT_FOLDER_FOR_VIDEO = 'static/DLFile/videoPic/outputFrames'

# 全局模型变量
model = None

from static.DLFile.model import build_my_model


def load_model(model_path):
    """加载模型函数"""
    global model
    model = build_my_model(img_size=224, num_classes=5)
    model.load_weights(model_path)
    return model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    创建Grad-CAM热力图，用于可视化模型关注的区域

    参数:
    img_array - 经过预处理的图像数组，形状为 (1, height, width, channels)
    model - 训练好的模型
    last_conv_layer_name - 最后一个卷积层的名称
    pred_index - 要可视化的类别索引，如果为None，则使用模型预测的类别

    返回:
    热力图数组，值范围在 [0, 1]
    """
    # 确保图像已正确归一化
    if img_array.max() > 1.0:
        img_array = img_array / 255.0

    # 创建一个输出最后一个卷积层和预测结果的模型
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 记录梯度
    with tf.GradientTape() as tape:
        # 计算最后一个卷积层的输出和模型预测
        conv_outputs, predictions = grad_model(img_array)

        # 如果未指定预测索引，使用概率最高的类别
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # 获取目标类别的输出
        class_channel = predictions[:, pred_index]

    # 计算最后一个卷积层输出相对于目标类别的梯度
    grads = tape.gradient(class_channel, conv_outputs)

    # 对梯度进行全局平均池化
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 获取卷积层输出
    conv_outputs = conv_outputs[0]

    # 将池化后的梯度与卷积层输出相乘，获取类激活图
    # 使用维度扩展确保可以进行矩阵乘法
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 将热力图归一化到 [0, 1] 范围
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


@bp.route('/imageSeg')
def imageSegmentation():
    """图像分割页面路由"""
    # 加载模型
    load_model("static/DLFile/my_model.h5")

    # 打印模型层以进行调试
    print("模型层:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name}")

    return render_template("imageSegPage.html")




@bp.route('/startSeg', methods=['POST'])
def startSegment():
    file = request.files['file']
    image_pil = Image.open(file.stream).convert('RGB')
    image_np = np.array(image_pil)

    # ---------- 预处理部分 ----------
    def crop_single_channel(img, tol=7):
        """处理单通道图像"""
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]

    def crop_image_from_gray(img, tol=7):
        """基于灰度阈值裁剪图像"""
        if img.ndim == 2:
            return crop_single_channel(img, tol)
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            if not mask.any():
                return img
            cropped_channels = [
                channel[np.ix_(mask.any(1), mask.any(0))] for channel in cv2.split(img)
            ]
            return cv2.merge(cropped_channels)

    def crop_image(img, tol=7):
        """裁剪并调整大小以保持原始尺寸"""
        h, w = img.shape[:2]
        cropped_img = crop_image_from_gray(img, tol)
        return cv2.resize(cropped_img, (w, h))

    def crop_image_with_contours(image, threshold=1):
        """使用轮廓检测裁剪图像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return image[y:y + h, x:x + w]
        else:
            return image

    def universal_crop(image, method='gray', tol=7):
        """根据输入参数选择裁剪方法"""
        if method == 'gray':
            return crop_image(image, tol)
        elif method == 'contours':
            return crop_image_with_contours(image)
        else:
            raise ValueError(f"不支持的裁剪方法: {method}")

    def load_ben_color(img, sigmaX=10):
        """使用加权高斯模糊增强图像颜色"""
        if img is None:
            return None
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
        return img

    # 应用预处理: 裁剪 + 增强 + 调整大小 + 归一化
    print("开始预处理图像")
    image_np = universal_crop(image_np, method='gray', tol=7)
    image_np = load_ben_color(image_np, sigmaX=10)
    image_np = cv2.resize(image_np, (224, 224))  # 调整为模型输入尺寸

    # 保存处理后的图像用于显示
    display_image = image_np.copy()

    # 归一化用于模型输入
    image_np = image_np.astype('float32') / 255.0
    image_input = np.expand_dims(image_np, axis=0)

    # 获取模型预测
    print("开始模型预测")
    pred = model.predict(image_input)
    pred_class = np.argmax(pred, axis=-1)[0]
    pred_probabilities = pred[0].tolist()
    print(f"预测类别: {pred_class}, 概率: {pred_probabilities}")

    # 打印模型的所有层以进行调试
    print("模型所有层:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name}")

    try:
        print("生成热力图可视化")

        # 确保显示图像在0-255范围内
        if display_image.max() <= 1.0:
            display_image = (display_image * 255).astype('uint8')

        # 获取图像边缘
        gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)

        # 使用结构化滤波器增强图像结构
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        filtered = cv2.filter2D(gray, -1, kernel)

        # 对结构增强的图像使用自适应阈值
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 提取视网膜特征（血管、视盘等）
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)

        # 创建特征热力图
        heatmap_base = np.zeros_like(gray)

        # 添加边缘特征
        edges = cv2.Canny(gray, 30, 150)
        heatmap_base = cv2.add(heatmap_base, edges)

        # 添加阈值特征
        heatmap_base = cv2.add(heatmap_base, thresh)

        # 如果找到圆（可能是视盘），添加到热力图
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # 绘制圆心
                cv2.circle(heatmap_base, (i[0], i[1]), 2, 255, 3)
                # 绘制圆周
                cv2.circle(heatmap_base, (i[0], i[1]), i[2], 255, 3)

        # 应用高斯模糊使热力图更平滑
        heatmap_base = cv2.GaussianBlur(heatmap_base, (15, 15), 0)

        # 归一化热力图
        heatmap_base = cv2.normalize(heatmap_base, None, 0, 255, cv2.NORM_MINMAX)

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(heatmap_base.astype(np.uint8), cv2.COLORMAP_JET)

        # 将热力图叠加到原始图像上
        alpha = 0.7  # 热力图透明度
        beta = 0.3  # 原始图像透明度
        gamma = 0  # 标量加到每个总和

        # 叠加热力图到原图
        superimposed_img = cv2.addWeighted(display_image, beta, heatmap_colored, alpha, gamma)

        # 转换为RGB以便matplotlib显示
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        # 创建只显示热力图的图表
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(superimposed_img_rgb)
        ax.axis('off')


    except Exception as e:
        # handel error
        print(f"Heat diagram generate error: {str(e)}")
        traceback.print_exc()

        print("Change to default display mode")
        fig, ax = plt.subplots(figsize=(8, 8))

        # make sure the img formate is right
        if isinstance(display_image, np.ndarray):
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype('uint8')

            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            ax.imshow(display_image_rgb)
        else:
            ax.text(0.5, 0.5, "Image display error", ha='center', va='center')

        ax.set_title(f'Predicted Class: {pred_class}')
        ax.axis('off')

        # 添加预测概率
        class_probs = [f"Class {i}: {prob * 100:.2f}%" for i, prob in enumerate(pred_probabilities)]
        prob_text = '\n'.join(class_probs)
        ax.text(10, 20, prob_text, color='white', fontsize=12,
                backgroundcolor='black', alpha=0.7)

    # 保存可视化结果
    print("Save result img")
    filename_only = "output_" + secure_filename(file.filename)
    result_full_path = os.path.join(OUTPUT_FOLDER, filename_only)
    result_url_path = f"/static/DLFile/output_folder/{filename_only}"

    print(f"Save to: {result_full_path}")
    print(f"Front EndURL Path: {result_url_path}")

    plt.savefig(result_full_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 返回结果给前端
    return jsonify({
        'message': 'File uploaded and processed',
        'result_path': result_url_path,  # 使用URL路径而非服务器路径
        'predicted_class': int(pred_class),
        'predicted_probabilities': pred_probabilities
    })

@bp.route('/processEachImg', methods=['POST'])
def processEachImg():
    """处理单张图像的API端点"""
    image_stream = BytesIO(request.data)
    image_stream.seek(0)

    img = load_image_without_path(image_stream)

    pred_mask = model.predict(img)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = tf.squeeze(pred_mask, axis=0)
    img = tf.squeeze(img, axis=0)

    overlay_mask = tf.keras.preprocessing.image.array_to_img(pred_mask)
    overlay_image = tf.keras.preprocessing.image.array_to_img(img)

    fig, ax = plt.subplots()
    ax.imshow(overlay_image, alpha=0.5)
    ax.imshow(overlay_mask, alpha=0.5)
    ax.axis('off')

    # 将结果保存到BytesIO对象
    buffer = BytesIO()
    plt.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    plt.close()

    return jsonify({'processedImageData': img_str})


@bp.route('/predictDR', methods=['POST'])
def predictDR():
    """视网膜病变预测API端点"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))  # 确保尺寸和模型一致

        img_array = keras_image.img_to_array(img) / 255.0  # 归一化
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500