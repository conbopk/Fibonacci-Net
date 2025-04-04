import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

path = "Dataset/brain-tumor-multimodal-image-ct-and-mri/Brain Tumor MRI images"
categories = ["Healthy", "Tumor"]

image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(path, category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image_paths.append(image_path)
        labels.append(category)

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

# df.to_csv("brain_tumor_mri.csv", index=False)

#Data exploding
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.columns)
# print(df.duplicated().sum())
# print(df.isnull().sum())
# print(df.info())
# print(df['label'].unique())
# print(df['label'].value_counts())


#Data visualization

# sns.set_style("whitegrid")
#
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.countplot(data=df, x="label", hue='label',palette="viridis", ax=ax, legend=False)
#
# ax.set_title("Distribution of Tumor Types", fontsize=14, fontweight='bold')
# ax.set_xlabel("Tumor Type", fontsize=12)
# ax.set_xlabel("Count", fontsize=12)
#
# for p in ax.patches:
#     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
#                 ha='center', va='bottom', fontsize=11, color='black',
#                 xytext=(0, 5), textcoords='offset points')
#
# plt.savefig("tumor_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()


# label_counts = df["label"].value_counts()
#
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = sns.color_palette("viridis", n_colors=len(label_counts))
#
# ax.pie(label_counts, labels=label_counts.index, colors=colors, autopct='%1.1f%%', startangle=140,
#        textprops={'fontsize': 12, 'weight': 'bold'}, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
#
# ax.set_title("Distribution of Tumor Type - Pie Chart", fontsize=14, fontweight='bold')
#
# plt.savefig("tumor_distribution_pie.png", dpi=300, bbox_inches="tight")
# plt.show()


# num_images = 5

# plt.figure(figsize=(15, 12))

# for i, category in enumerate(categories):
#     category_images = df[df['label'] == category]['image_path'].iloc[:num_images]
#
#     for j, img_path in enumerate(category_images):
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         plt.subplot(len(categories), num_images, i * num_images + j + 1)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(category)
#
# plt.tight_layout()
# plt.savefig("data_visualization.png", dpi=300, bbox_inches="tight")
# plt.show()

# label_encoder = LabelEncoder()
# df['category_encoded'] = label_encoder.fit_transform(df['label'])
#
# df = df[['image_path', 'category_encoded']]
# df.to_csv('data_encoded.csv', index=False)
#
# ros = RandomOverSampler(random_state=42)
# x_resampled, y_resampled = ros.fit_resample(df[['image_path']], df['category_encoded'])
#
# df_resampled = pd.DataFrame(x_resampled, columns=['image_path'])
# df_resampled['category_encoded'] = y_resampled
#
# print("\nClass distribution after oversampling:")
# print(df_resampled['category_encoded'].value_counts())
#
# df_resampled['category_encoded'] = df_resampled['category_encoded'].astype(str)
# df_resampled.to_csv('data_resampled.csv', index=False)

