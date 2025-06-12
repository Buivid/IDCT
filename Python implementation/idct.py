import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct

def dct_1d(x, N):
    out = np.zeros(N)
    for u in range(N):
        sum_val = 0
        for i in range(N):
            sum_val += x[i] * np.cos((2*i+1)*u*np.pi/(2*N))
        if u == 0:
            Cu = np.sqrt(1/N)
        else:
            Cu = np.sqrt(2/N)
        #print(type(sum_val))
        out[u] = Cu * sum_val
        #print(type(out[u]))
    return out

def dct_2d(block):
    N = 8
    dct_block = np.zeros((N,N))
    
    for u in range(N):
        dct_block[u] = dct_1d(block[u], N)
    for v in range(N):
        dct_block[:, v] = dct_1d(dct_block[:,v], N)
    #print(type(dct_block[0,0]))
    return dct_block
        
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
    reconstructed = np.zeros_like(img_data, dtype = np.uint8)

    dct_blocks_py = np.zeros_like(img_data, dtype = np.float32)
    reconstructed_py = np.zeros_like(img_data, dtype = np.float32)
    
    for i in range(0, img_data.shape[0], 8):
        for j in range(0, img_data.shape[1], 8):
            block = img_data[i:i+8, j:j+8]
            
            dct_block_py = dct(dct(block.T, norm='ortho').T, norm='ortho') 
            dct_block = dct_2d(block).astype(np.float32)
            dct_blocks_py[i:i+8, j:j+8] = dct_block_py  
            dct_blocks[i:i+8, j:j+8] = dct_block  
    print(type(block[0,0]))
    print(type(dct_blocks[0,0]))
    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            dct_block = dct_blocks[i:i+8, j:j+8]
            dct_block_py = dct_blocks_py[i:i+8, j:j+8]
            #print(dct_block)
            reconstructed_block = idct_2d(dct_block)
            reconstructed_block_py = idct(idct(dct_block_py, axis = 0, norm='ortho'), axis = 1, norm='ortho') 
            reconstructed[i:i+8, j:j+8] = reconstructed_block  
            reconstructed_py[i:i+8, j:j+8] = reconstructed_block_py
    print(type(reconstructed[0,0]))
    plt.figure(figsize=(15, 5))
    
    plt.subplot(3, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 3)
    plt.title("DCT")
    plt.imshow(dct_blocks, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.title("IDCT")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')


    plt.subplot(3, 2, 2)
    plt.title("Block(0,0)")
    plt.imshow(dct_blocks[0:15,0:15], cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.title("DCT python")
    plt.imshow(dct_blocks_py, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title("IDCT python")
    plt.imshow(reconstructed_py, cmap='gray')
    plt.axis('off')
    
    x, y = h//8*8, w//8*8
    
    for i in range(x):
        sum_val_my, sum_val_py = 0, 0
        for j in range(y):
            sum_val_py += (abs(img_data[i,j]-reconstructed_py[i,j])**2).astype(np.float32)
            sum_val_my += (abs(img_data[i,j]-reconstructed[i,j])**2).astype(np.float32)
    RMSE_my = np.sqrt(1/(x*y) * sum_val_my) 
    RMSE_py = np.sqrt(1/(x*y) * sum_val_py) 
    #print(type(sum_val_py))
   
    print('RMSE_my = ', RMSE_my)
    PSNR_my = 20 * np.log10(255/RMSE_my)

    print('RMSE_py = ', RMSE_py)
    PSNR_py = 20 * np.log10(255/RMSE_py)

    print('PSNR_py = ', PSNR_py)
    print('PSNR_my = ', PSNR_my)
    print(type(PSNR_my))
    plt.show()
if __name__ == "__main__":
    process_image("baboon.jpg")