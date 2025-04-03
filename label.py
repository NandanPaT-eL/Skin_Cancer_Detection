import cv2
import matplotlib.pyplot as plt

image_path = "Skin cancer ISIC The International Skin Imaging Collaboration/Train/actinic keratosis/ISIC_0025780.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

plt.imshow(image)
plt.title("Label: Melanoma")  # Ensure this matches the actual label
plt.axis("off")
plt.show()
