import TIFFany as tf

original = "TIFFany/img/Lena_unedited.png"
filtered = "TIFFany/img/Lena_filtered.png"
modified = "TIFFany/img/Lena_modified.png"
modified_heavy = "TIFFany/img/Lena_heavily_modified.png"

print(tf.mse(original, filtered))
print(tf.mse(original, modified_heavy))

print(tf.ssim(original, filtered))
print(tf.ssim(original, modified))

tf.gaussian_blur(original, radius=5, display=True)
tf.median_blur(original, radius=10, display=True)
tf.median_blur(original, radius=5, display=False)

tf.show(tf.foreground_mask(original, modified, threshold=128))
tf.show(tf.foreground_mask(original, modified_heavy, threshold=128))
tf.show(tf.contours(original, modified_heavy, threshold=120))
