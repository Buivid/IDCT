import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct
from skimage import metrics

def dct_1d(x, N):
    out = np.zeros(N)
    for u in range(N):
        sum_val = 0
        for i in range(N):

            sum_val += (x[i]* np.cos((2*i+1)*u*np.pi/(2*N)))
        if u == 0:
            Cu = np.sqrt(1/N)
        else:
            Cu = np.sqrt(2/N)
        out[u] = (Cu*sum_val)
        
        
        # if(u == 0):
        #     print(np.binary_repr(np.int16(res)))
        #     print(np.binary_repr(np.int16(res2)))
        #     print('res =', res2)
        #     print('out =', out[u])
    return out

def dct_2d(block):
    N = 8
    dct_block = np.zeros((N,N)).astype(np.int16)
    out = np.zeros((N,N)).astype(np.int8)
    
    for u in range(N):
        dct_block[u] = dct_1d(block[u], N)
    for v in range(N):
        dct_block[:, v] = dct_1d(dct_block[:,v], N)
    # print(dct_block)
    for i in range(N):
        for j in range(N):
            out[i,j] = np.right_shift(dct_block[i,j], 5, casting='unsafe')
    
    return out
        
def idct_1d(X, N):
    out = np.zeros(N).astype(np.int16)
    for i in range(N):
        sum_val = 0
        for u in range(N):
            if u == 0:
                Cu = np.sqrt(1/N)
            else:
                Cu = np.sqrt(2/N)
            sum_val += (np.int16)(Cu * X[u] * np.cos((2*i+1)*u*np.pi/(2*N)))
        out[i] = sum_val
    return out

def idct_2d(dct_block):
    N = 8
    block = np.zeros((N,N)).astype(np.int16)

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
    
    dct_blocks = np.zeros_like(img_data, dtype = np.int8)
    reconstructed = np.zeros_like(img_data, dtype = np.uint8)

    # dct_blocks_py = np.zeros_like(img_data, dtype = np.int16)
    # reconstructed_py = np.zeros_like(img_data, dtype = np.uint8)
    
    for i in range(0, img_data.shape[0], 8):
        for j in range(0, img_data.shape[1], 8):
            block = img_data[i:i+8, j:j+8]
            dct_block = dct_2d(block)
            dct_blocks[i:i+8, j:j+8] = dct_block
            #dct_block_py = dct(dct(block, axis = 0, norm='ortho'), axis = 1, norm='ortho') 
            #dct_blocks_py[i:i+8, j:j+8] = dct_block_py  
    
    max_value = -128
    min_value = 127

    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            if(dct_blocks[i,j]>max_value):
                max_value = dct_blocks[i,j]
            if(dct_blocks[i,j]<min_value):
                min_value = dct_blocks[i,j]

    print('max value = ', max_value, '\n', 'min value = ', min_value)    
    # print(dct_block[0,0].dtype)
    # print(type(block[0,0]))
    # print(type(dct_blocks[0,0]))
    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            dct_block = dct_blocks[i:i+8, j:j+8]
            reconstructed_block = idct_2d(dct_block)
            reconstructed[i:i+8, j:j+8] = reconstructed_block 
            # dct_block_py = dct_blocks_py[i:i+8, j:j+8]
            # reconstructed_block_py = idct(idct(dct_block_py, axis = 0, norm='ortho'), axis = 1, norm='ortho')  
            # reconstructed_py[i:i+8, j:j+8] = reconstructed_block_py
    #print(reconstructed[0,0].dtype)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(2, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("DCT")
    plt.imshow(dct_blocks, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("IDCT")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')


    plt.subplot(2, 2, 4)
    plt.title("Block(0,0)")
    plt.imshow(dct_blocks[0:15,0:15], cmap='gray')
    plt.axis('off')

    # plt.subplot(3, 2, 5)
    # plt.title("DCT python")
    # plt.imshow(dct_blocks_py, cmap='gray')
    # plt.axis('off')

    # plt.subplot(3, 2, 6)
    # plt.title("IDCT python")
    # plt.imshow(reconstructed_py, cmap='gray')
    # plt.axis('off')
    
    x, y = h//8*8, w//8*8
    sum_val = 0
    for i in range(x):
        for j in range(y):
            sum_val += (abs(img_data[i,j].astype(np.float64)-reconstructed[i,j].astype(np.float64))**2)
    MSE_my = sum_val / (x*y)
    PSNR_my = 10 * np.log10(255**2/MSE_my)
    mse = metrics.mean_squared_error(img_data, reconstructed)
    psnr = metrics.peak_signal_noise_ratio(img_data, reconstructed)

    print('MSE = ', MSE_my)
    print('PSNR = ', PSNR_my)
    print('mse(библиотечная) = ', mse)
    print('psnr(библиотечная) = ', psnr)

    plt.show()
if __name__ == "__main__":
    process_image("lena.jpg")