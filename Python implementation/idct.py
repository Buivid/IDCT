import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct

def idct_1d(X, N):
    out = np.zeros(N)
    for i in range(N):
        sum_val = 0
        for u in range(N):
            if u == 0:
                Cu = np.sqrt(1/N)
            else:
                Cu = np.sqrt(2/N)
            sum_val += Cu * X[u] * np.cos((2*i+1)*u*np.pi/(2*N))
        out[i] = sum_val
    return out

def idct_2d(dct_block):
    N = 8
    block = np.zeros((N,N))

    for i in range(N):
        block[i] = idct_1d(dct_block[i], N)
    for j in range(N):
        block[:,j] = idct_1d(block[:,j], N)
    return block

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)
    h, w = img_data.shape
    img_data = img_data[:h//8*8, :w//8*8]
    
    dct_blocks = np.zeros_like(img_data, dtype = np.float32)
    reconstructed = np.zeros_like(img_data, dtype = np.float32)
    
    for i in range(0, img_data.shape[0], 8):
        for j in range(0, img_data.shape[1], 8):
            block = img_data[i:i+8, j:j+8].astype(np.float32)
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho') 
            dct_blocks[i:i+8, j:j+8] = dct_block  

    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            dct_block = dct_blocks[i:i+8, j:j+8]
            reconstructed_block = idct_2d(dct_block)
            reconstructed[i:i+8, j:j+8] = reconstructed_block  
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("DCT")
    plt.imshow(dct_blocks, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("IDCT")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    
    plt.show()
    

if __name__ == "__main__":
    process_image("kit.jpg")