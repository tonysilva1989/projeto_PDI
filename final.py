import cv2
import matplotlib.pyplot as plt
import numpy as np


pyr_scale = 0.5
levels = 3
winsize = 15
iterations = 3
p_n = 5
p_sigma = 1.2
flags = 0
default_step_local = 16
w_intensidade   = 0.30
w_color       = 0.30
w_orientacao = 0.20
w_motion      = 0.20

#mascaras com rotacao 0, 45, 90 e 135
GaborKernel_0 = [\
    [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ],\
    [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],\
    [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],\
    [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],\
    [ 0.000921261, 0.006375831, -0.174308068, -0.067914552, 1.000000000, -0.067914552, -0.174308068, 0.006375831, 0.000921261 ],\
    [ 0.000624940, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.000624940 ],\
    [ 0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076 ],\
    [ 2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05 ],\
    [ 1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06 ]\
]
GaborKernel_45 = [\
    [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05,  0.000744712,  0.000132863, -9.04408E-06, -1.01551E-06 ],\
    [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700,  0.000389916,  0.003516954,  0.000288732, -9.04408E-06 ],\
    [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072,  0.000847346,  0.003516954,  0.000132863 ],\
    [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072,  0.000389916,  0.000744712 ],\
    [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000,  0.249959607, -0.139178011, -0.022947700,  3.79931E-05 ],\
    [  0.000744712,  0.003899160, -0.108372072, -0.302454279,  0.249959607,  0.460162150,  0.052928748, -0.013561362, -0.001028923 ],\
    [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011,  0.052928748,  0.044837725,  0.002373205, -0.000279806 ],\
    [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362,  0.002373205,  0.000925120,  2.25320E-05 ],\
    [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806,  2.25320E-05,  4.04180E-06 ]\
]
GaborKernel_90 = [\
    [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ],\
    [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],\
    [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],\
    [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],\
    [  0.002010422,  0.030415784,  0.211749204,  0.678352526,  1.000000000,  0.678352526,  0.211749204,  0.030415784,  0.002010422 ],\
    [ -0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537 ],\
    [ -0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433 ],\
    [  1.28181E-05,  0.000193926,  0.001350077,  0.004325061,  0.006375831,  0.004325061,  0.001350077,  0.000193926,  1.28181E-05 ],\
    [  1.85212E-06,  2.80209E-05,  0.000195076,  0.000624940,  0.000921261,  0.000624940,  0.000195076,  2.80209E-05,  1.85212E-06 ]
]
GaborKernel_135 = [\
    [ -1.01551E-06, -9.04408E-06,  0.000132863,  0.000744712,  3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06 ],\
    [ -9.04408E-06,  0.000288732,  0.003516954,  0.000389916, -0.022947700, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05 ],\
    [  0.000132863,  0.003516954,  0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806 ],\
    [  0.000744712,  0.000389916, -0.108372072, -0.302454279,  0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923 ],\
    [  3.79931E-05, -0.022947700, -0.139178011,  0.249959607,  1.000000000, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05 ],\
    [ -0.001028923, -0.013561362,  0.052928748,  0.460162150,  0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712 ],\
    [ -0.000279806,  0.002373205,  0.044837725,  0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863 ],\
    [  2.25320E-05,  0.000925120,  0.002373205, -0.013561362, -0.022947700, 0.000389916, 0.003516954, 0.000288732, -9.04408E-06 ],\
    [  4.04180E-06,  2.25320E-05, -0.000279806, -0.001028923,  3.79931E-05 , 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06 ]\
]


class saliencia_class:

    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.prev_frame = None
        self.SM = None
        self.GaborKernel0   = np.array(GaborKernel_0)
        self.GaborKernel45  = np.array(GaborKernel_45)
        self.GaborKernel90  = np.array(GaborKernel_90)
        self.GaborKernel135 = np.array(GaborKernel_135)


    def sal_mapa_RGBI(self, inputImage):
        src = np.float32(inputImage) * 1.0/255
        (B, G, R) = cv2.split(src)
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return R, G, B, I

    def gera_piram_gauss(self, src):
        dst = list()
        dst.append(src)
        for i in range(1,9):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst


    def mapa_caract_OF(self, src):
        GaussianI = self.gera_piram_gauss(src)
        GaborOutput0   = [ np.empty((1,1)), np.empty((1,1)) ]  
        GaborOutput45  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput90  = [ np.empty((1,1)), np.empty((1,1)) ]
        GaborOutput135 = [ np.empty((1,1)), np.empty((1,1)) ]
        for j in range(2,9):
            GaborOutput0.append(   cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel0) )
            GaborOutput45.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel45) )
            GaborOutput90.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel90) )
            GaborOutput135.append( cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel135) )
        CSD0   = self.difer_center_surr(GaborOutput0)
        CSD45  = self.difer_center_surr(GaborOutput45)
        CSD90  = self.difer_center_surr(GaborOutput90)
        CSD135 = self.difer_center_surr(GaborOutput135)
        dst = list(CSD0)
        dst.extend(CSD45)
        dst.extend(CSD90)
        dst.extend(CSD135)

        return dst

    def gera_mapa_caract_MFM(self, src):

        I8U = np.uint8(255 * src)
        cv2.waitKey(10)

        if self.prev_frame is not None:
            pyr_scale= pyr_scale
            levels = levels
            winsize = winsize
            iterations = iterations
            p_n = p_n
            p_sigma = p_sigma
            flags = flags
            flow = cv2.calcOpticalFlowFarneback(prev = self.prev_frame,next = I8U,pyr_scale = pyr_scale,levels = levels,winsize = winsize,iterations = iterations,p_n = p_n,p_sigma = p_sigma,flags = flags,flow = None)
            flowx = flow[...,0]
            flowy = flow[...,1]
        else:
            flowx = np.zeros(I8U.shape)
            flowy = np.zeros(I8U.shape)
        dst_x = self.piram_gauss_CSD(flowx)
        dst_y = self.piram_gauss_CSD(flowy)
        self.prev_frame = np.uint8(I8U)
        return dst_x, dst_y

    def difer_center_surr(self, GaussianMaps):
        dst = list()
        for s in range(2,5):
            now_size = GaussianMaps[s].shape
            now_size = (now_size[1], now_size[0]) 
            tmp = cv2.resize(GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
            tmp = cv2.resize(GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
        return dst

    def piram_gauss_CSD(self, src):
        GaussianMaps = self.gera_piram_gauss(src)
        dst = self.difer_center_surr(GaussianMaps)
        return dst

    def mapa_caract(self, I):
        return self.piram_gauss_CSD(I)

    def gera_mapa_caract(self, R, G, B):
        tmp1 = cv2.max(R, G)
        RGBMax = cv2.max(B, tmp1)
        RGBMax[RGBMax <= 0] = 0.0001   
        RGMin = cv2.min(R, G)
        RG = (R - G) / RGBMax
        BY = (B - RGMin) / RGBMax
        RG[RG < 0] = 0
        BY[BY < 0] = 0
        RGFM = self.piram_gauss_CSD(RG)
        BYFM = self.piram_gauss_CSD(BY)
        return RGFM, BYFM



    def normalizar(self, src):
        minn, maxx, dummy1, dummy2 = cv2.minMaxLoc(src)
        if maxx!=minn:
            dst = src/(maxx-minn) + minn/(minn-maxx)
        else:
            dst = src - minn
        return dst

    def normalizacao(self, src):
        dst = self.normalizar(src)
        lmaxmean = self.max_local_media(dst)
        normcoeff = (1-lmaxmean)*(1-lmaxmean)
        return dst * normcoeff


    def max_local_media(self, src):
        stepsize = default_step_local
        width = src.shape[1]
        height = src.shape[0]
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height-stepsize, stepsize):
            for x in range(0, width-stepsize, stepsize):
                localimg = src[y:y+stepsize, x:x+stepsize]
                lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1
        return lmaxmean / numlocal


    def regiao_saliencia_mapa(self, src):

        binarized_SM = self.binarizado_sal(src)
        img = src.copy()
        mask =  np.where((binarized_SM!=0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        rect = (0,0,1,1) 
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel, fgdModel=fgdmodel, iterCount=iterCount, mode=cv2.GC_INIT_WITH_MASK)
        mask_out = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask_out)
        return output

    def normal_mapas(self, FM):
        NFM = list()
        for i in range(0,6):
            normalizada = self.normalizacao(FM[i])
            nownfm = cv2.resize(normalizada, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            NFM.append(nownfm)
        return NFM

    def gera_mapa_consp_int(self, IFM):
        NIFM = self.normal_mapas(IFM)
        ICM = sum(NIFM)
        return ICM

    def gera_mapa_consp_CCM(self, CFM_RG, CFM_BY):

        CCM_RG = self.gera_mapa_consp_int(CFM_RG)
        CCM_BY = self.gera_mapa_consp_int(CFM_BY)

        CCM = CCM_RG + CCM_BY

        return CCM

    def gera_mapa_consp_MC(self, MFM_X, MFM_Y):
        return self.gera_mapa_consp_CCM(MFM_X, MFM_Y)

    def mapa_consp(self, OFM):
        OCM = np.zeros((self.height, self.width))
        for i in range (0,4):
            nowofm = OFM[i*6:(i+1)*6]  
            NOFM = self.gera_mapa_consp_int(nowofm)
            NOFM2 = self.normalizacao(NOFM)
            OCM += NOFM2
        return OCM


    def mapa_sal(self, src):
        size = src.shape
        width  = size[1]
        height = size[0]
        R, G, B, I = self.sal_mapa_RGBI(src)
        IFM = self.mapa_caract(I)
        CFM_RG, CFM_BY = self.gera_mapa_caract(R,G,B)
        OFM = self.mapa_caract_OF(I)
        MFM_X, MFM_Y = self.gera_mapa_caract_MFM(I)
        ICM = self.gera_mapa_consp_int(IFM)
        CCM = self.gera_mapa_consp_CCM(CFM_RG, CFM_BY)
        OCM = self.mapa_consp(OFM)
        MCM = self.gera_mapa_consp_MC(MFM_X, MFM_Y)
        wi = w_intensidade
        wc = w_color
        wo = w_orientacao
        wm = w_motion
        SMMat = wi*ICM + wc*CCM + wo*OCM + wm*MCM
        sm_norm = self.normalizar(SMMat)
        sm_norm2 = sm_norm.astype(np.float32)
        suavizado = cv2.bilateralFilter(sm_norm2, 7, 3, 1.55)
        self.SM = cv2.resize(suavizado, (width,height), interpolation=cv2.INTER_NEAREST)
        return self.SM

    def binarizado_sal(self, src):
        if self.SM is None:
            self.SM = self.mapa_sal(src)
        SM_I8U = np.uint8(255 * self.SM)
        thresh, binarized_SM = cv2.threshold(SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binarized_SM    

  
#funcao "main()"
if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    tamanho = img.shape
    altura = tamanho[0]
    largura  = tamanho[1]
    sm = saliencia_class(largura, altura)
    mapaSal = sm.mapa_sal(img)
    binarizado = sm.binarizado_sal(img)
    saliencias = sm.regiao_saliencia_mapa(img)
    plt.subplot(2,2,1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2,2,2), plt.imshow(mapaSal, 'gray')
    plt.subplot(2,2,3), plt.imshow(cv2.cvtColor(saliencias, cv2.COLOR_BGR2RGB))
    plt.show()
    ##cv2.imshow("plot",img)
