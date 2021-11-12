import cv2 as cv
bg=cv.imread('OIP-C.jfif',cv.IMREAD_COLOR)
fg=cv.imread('R-C.jfif',cv.IMREAD_COLOR)
print('背景图片的大小{},前景图片的大小{}'.format(bg.shape,fg.shape))
dim=(1200,800)
resized_bg=cv.resize(bg,dim,interpolation=cv.INTER_AREA)
resized_fg=cv.resize(fg,dim,interpolation=cv.INTER_AREA)
print('调整后背景图片的大小{},调整后前景图片的大小{}'.format(resized_bg.shape,resized_fg.shape))
blend=cv.addWeighted(resized_bg,0.5,resized_fg,0.8,0.0)
cv.imwrite('blended.png',blend)



