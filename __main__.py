# from TIFFany import foreground_mask, gaussian_blur, median_blur, mse, ssim
import TIFFany as tf

src = "TIFFany/img/Lena_unedited.png"

print(tf.mse(src, src)) # expecting a value of 0.0
print(tf.ssim(src, src)) # expecting a value of 1.0
#tf.gaussian_blur(src, radius=5, display=True)
#tf.median_blur(src, radius=5, display=True)
#tf.foreground_mask(src, src, threshold=128)
#tf.show(tf.foreground_mask(src, src, threshold=128))
tf.show(tf.contours(src, src, threshold=120))
