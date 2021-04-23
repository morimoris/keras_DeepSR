import cv2

img_high = cv2.imread("result/model_b/high_b_38.jpg")
img_low = cv2.imread("result/model_b/low_b_38.jpg")
img_pred = cv2.imread("result/model_b/pred_b_38.jpg")

print("low : " + str(cv2.PSNR(img_high, img_low)))
print("pred : " + str(cv2.PSNR(img_high, img_pred)))