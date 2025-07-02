import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct
from skimage import metrics

N = 8

def butterfly(I0, I1):
    O0 = (np.int16)(I0) + (np.int16)(I1)
    O1 = (np.int16)(I0) - (np.int16)(I1)
    return O0, O1

def rotation(I0, I1, k, n):
    #O0 = k*I1*(np.sin(np.pi*n/(2*N)) - np.cos(np.pi*n/(2*N))) + k*np.cos(np.pi*n/(2*N))*(I0+I1)
    #O1 = -k*I0*(np.cos(np.pi*n/(2*N)) + np.sin(np.pi*n/(2*N))) + k*np.cos(np.pi*n/(2*N))*(I0+I1)
    O0 = I0*k*np.cos(n*np.pi/(2*N)) + I1*k*np.sin(n*np.pi/(2*N))
    O1 = -I0*k*np.sin(n*np.pi/(2*N)) + I1*k*np.cos(n*np.pi/(2*N))
    return O0, O1

def scaleup(In, k):
    O = In * k
    return O

def loeffler_DCT(x):
    #stage 1:
    out = np.zeros(N)
    temp = np.zeros(N)
    temp[0], temp[7] = butterfly(x[0], x[7])
    temp[1], temp[6] = butterfly(x[1], x[6])
    temp[2], temp[5] = butterfly(x[2], x[5])
    temp[3], temp[4] = butterfly(x[3], x[4])
    #stage 2
    temp[0], temp[3] = butterfly(temp[0], temp[3])
    temp[1], temp[2] = butterfly(temp[1], temp[2])
    temp[4], temp[7] = rotation(temp[4], temp[7], 1, 3)
    temp[5], temp[6] = rotation(temp[5], temp[6], 1, 1)
    
    #stage 3
    out[0], out[4] = butterfly(temp[0], temp[1])
    out[2], out[6] = rotation(temp[2], temp[3], np.sqrt(2), 6)
    temp[4], temp[6] = butterfly(temp[4], temp[6])
    temp[7], temp[5] = butterfly(temp[7], temp[5])
    
    #stage 4
    out[1], out[7] = butterfly(temp[7], temp[4])
    out[3] = scaleup(temp[5], np.sqrt(2))
    out[5] = scaleup(temp[6], np.sqrt(2))

    return out

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
        dct_block[u] = loeffler_DCT(block[u])
        # dct_block[u] = dct_1d(block[u], N)
        
    dct_block = np.transpose(dct_block)
    for v in range(N):
        dct_block[v] = loeffler_DCT(dct_block[v]) 
        # dct_block[v] = dct_1d(dct_block[v], N)

    for i in range(8):
        for j in range(8):
            dct_block[i, j] *= 1/N
    # out = dct_block
    for i in range(N):
        for j in range(N):
            out[i,j] = np.right_shift(dct_block[i,j], 5, casting='unsafe')
    

    
    # print(dct_block)


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

def loeffler_IDCT(X):
    out = np.zeros(N)
    temp = np.zeros(N)
    
    temp = [X[0], X[4], X[2], X[6], X[7], X[3], X[5], X[1]]
    #stage 4
    temp[7], temp[4] = butterfly(temp[7], temp[4])
    temp[5] = scaleup(temp[5], np.sqrt(2))
    temp[6] = scaleup(temp[6], np.sqrt(2))

    #stage 3
    temp[0], temp[1] = butterfly(temp[0], temp[1])

    temp[2], temp[3] = rotation(temp[2], temp[3], np.sqrt(2), -6)
    temp[4], temp[6] = butterfly(temp[4], temp[6])

    temp[7], temp[5] = butterfly(temp[7], temp[5])

    #stage 2
    temp[0], temp[3] = butterfly(temp[0], temp[3])

    temp[1], temp[2] = butterfly(temp[1], temp[2])

    temp[4], temp[7] = rotation(temp[4], temp[7], 1, -3)
    temp[5], temp[6] = rotation(temp[5], temp[6], 1, -1)
    #stage 1
    out[0], out[7] = butterfly(temp[0], temp[7])
    out[1], out[6] = butterfly(temp[1], temp[6])
    out[2], out[5] = butterfly(temp[2], temp[5])
    out[3], out[4] = butterfly(temp[3], temp[4])
    # for i in range(N):
    #     out[i] /= np.sqrt(N)
    return out

def idct_2d(dct_block):
    N = 8
    block = np.zeros((N,N)).astype(np.int8)

    for i in range(N):
        # block[i] = idct_1d(dct_block[i], N)
        block[i] = loeffler_IDCT(dct_block[i])
    block = np.transpose(block)
    for j in range(N):
        # block[j] = idct_1d(block[j], N)
        block[j] = loeffler_IDCT(block[j])
    for i in range(N):
        for j in range(N):
            block[i,j] *= 1/N
    for i in range(N):
        for j in range(N):
            block[i,j] = np.left_shift(block[i,j], 5, casting='unsafe')
    return block

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)
    h, w = img_data.shape
    img_data = img_data[:h//8*8, :w//8*8]
    
    dct_blocks = np.zeros_like(img_data, dtype = np.int16)
    reconstructed = np.zeros_like(img_data, dtype = np.uint8)
    
    for i in range(0, img_data.shape[0], 8):
        for j in range(0, img_data.shape[1], 8):
            block = img_data[i:i+8, j:j+8]
            dct_block = dct_2d(block).astype(np.int8)
            dct_blocks[i:i+8, j:j+8] = dct_block
    
    max_value = -128
    min_value = 127

    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            if(dct_blocks[i,j]>max_value):
                max_value = dct_blocks[i,j]
            if(dct_blocks[i,j]<min_value):
                min_value = dct_blocks[i,j]

    print('max value = ', max_value, '\n', 'min value = ', min_value)    

    for i in range(0, dct_blocks.shape[0], 8):
        for j in range(0, dct_blocks.shape[1], 8):
            dct_block = dct_blocks[i:i+8, j:j+8]
            reconstructed_block = idct_2d(dct_block)
            #reconstructed_block = idct(idct(dct_block, axis = 0, norm='ortho'), axis = 1, norm='ortho')
            reconstructed[i:i+8, j:j+8] = reconstructed_block 
            
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
    # x=[0, 1, 2, 3, 4, 5, 6, 7]
    # y1 = dct_1d(x, 8)
    # print('DCT-II\n',np.around(y1,4))

    # q1 = idct(y1, norm = 'ortho')
    # q2 = idct_1d(y1, N)
    # q3 = loeffler_IDCT(y1)
    # print('IDCT(scipy)\n',np.around(q1,4))
    # print('IDCT-II\n',np.around(q2,4))
    # print('IDCT(loeffler)\n',np.around(q3,4))

    # y4 = dct(x, norm = 'ortho')
    # print('DCT(scipy)\n',np.around(y4,4))

    # y2 = loeffler_DCT(x)
    # print('DCT loeffler\n',np.around(y2,4))

    # y3 = loeffler_DCT_2(x)
    # print('DCT loeffler (presentation)\n',np.around(y3,4))

