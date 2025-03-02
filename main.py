from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'original' not in request.files or 'compressed' not in request.files:
        return jsonify({'error': 'Both images are required'}), 400
    
    original = request.files['original']
    compressed = request.files['compressed']
    
    original_path = os.path.join(UPLOAD_FOLDER, original.filename)
    compressed_path = os.path.join(UPLOAD_FOLDER, compressed.filename)
    
    original.save(original_path)
    compressed.save(compressed_path)
    
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(compressed_path)
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    mse_val = np.mean((img1 - img2) ** 2)
    psnr_val = cv2.PSNR(img1, img2)
    ssim_val = ssim(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    
    original_size = round(os.path.getsize(original_path) / 1024, 2)  # KB
    compressed_size = round(os.path.getsize(compressed_path) / 1024, 2)  # KB
    compression_ratio = round(original_size / compressed_size, 2) if compressed_size > 0 else 0
    percentage_compression = round(((original_size - compressed_size) / original_size) * 100, 2) if original_size > 0 else 0
    
    response = {
        'original_url': '/' + original_path,
        'compressed_url': '/' + compressed_path,
        'original_name': original.filename,
        'compressed_name': compressed.filename,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'original_resolution': f"{img1.shape[1]}x{img1.shape[0]}",
        'compressed_resolution': f"{img2.shape[1]}x{img2.shape[0]}",
        'mse': round(mse_val, 2),
        'psnr': round(psnr_val, 2),
        'ssim': round(ssim_val, 2),
        'compression_ratio': compression_ratio,
        'percentage_compression': percentage_compression
    }
    print("JSON Response:", response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
