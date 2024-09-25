---
title: Military Aircraft Detection
description: Military aircraft detection in images and videos using YOLOv8.
author: <author_id>
date: 2024-06-01 11:33:00 +0800
categories: [Personal Project, Deep Learning]
tags: [project, computer vision, YOLOv8, Deep Learning, Object Detection, classification]
pin: true
math: true
mermaid: true
image:
  path: /assets/img/AirCraft/presentation.jpg
  alt: F18 airplanes detection
---

# Military Aircraft Detection in Images and Videos using YOLOv8

## Introduction

In this project, I used a **YOLOv8** model to **detect and classify military aircraft** in images and videos. The goal is to demonstrate the power of Object Detection in the field of **Computer Vision**, particularly for the recognition of complex and moving objects, especially objects with similar characteristics like military aircraft.

> The notebook associated with this project is available at [this Kaggle link](https://www.kaggle.com/code/hisakaa/yolov8-aircraft-detection/).
{: .prompt-info }

## YOLOv8

<div class="box-tip" markdown="1">
<div class="title">YOLOv8</div>

YOLOv8 is an Object Detection model based on the YOLO (You Only Look Once) neural network. It is the eighth version of this model, which has been improved to be faster and more accurate than its predecessors. Developed by Ultralytics, YOLOv8 offers significant improvements in terms of accuracy and speed compared to its previous versions.

**Main Features of YOLOv8:** 

- **Real-time Detection**: Capable of processing live videos to detect objects instantly.
- **High Accuracy**: Uses advanced deep learning techniques to provide precise predictions.
- **Efficiency**: Designed to be used even on machines with limited resources, such as laptops without a powerful GPU.

**Advantages of YOLOv8:**

- **Speed**: Optimized for speed, which is crucial for applications like aerial surveillance.
- **Versatility**: Can be applied to various types of objects and contexts, including aircraft detection in images and videos.
- **Ease of Use**: Integrated with popular development tools and well-documented, making it easy to implement.

</div>

By using YOLOv8, this project aims to demonstrate how modern computer vision technologies can be effectively applied for detecting specific objects in images and videos, contributing to fields such as security and aerial surveillance.

## Dataset

We have a rich dataset containing 14,500 images, each containing one or more aircraft. For each image, an associated CSV file provides detailed annotations of the aircraft present, including the coordinates of their positions (xmin, ymin, xmax, ymax) and their classification.

<div class="box-info" markdown="1">

<div class="title"> Example of a CSV File </div>

| filename                         | width | height | class | xmin | ymin | xmax | ymax |
|----------------------------------|-------|--------|-------|------|------|------|------|
| 000aa01b25574f28b654718db0700f72 | 2048  | 1365   | F35   | 852  | 177  | 1998 | 503  |
| 000aa01b25574f28b654718db0700f72 | 2048  | 1365   | JAS39 | 169  | 769  | 549  | 893  |
| 000aa01b25574f28b654718db0700f72 | 2048  | 1365   | JAS39 | 125  | 908  | 440  | 1009 |
| 000aa01b25574f28b654718db0700f72 | 2048  | 1365   | B52   | 277  | 901  | 1288 | 1177 |

This data provides a solid foundation for training and **testing** our YOLOv8 model, enabling it to accurately recognize various types of military aircraft in different contexts.

</div>

### Sample Images from the Dataset

Here are some sample images from the dataset used to train and test the YOLOv8 model, along with their corresponding annotations:

![C5](/assets/img/AirCraft/C5.jpg)

This image shows a **C5 Galaxy** aircraft, a heavy military transport aircraft used by the US Air Force. Military aircraft can have various shapes and sizes, making their detection and classification challenging for Object Detection models.

![Mirage2000](/assets/img/AirCraft/Mirage2000.jpg)

This image shows a **Mirage 2000** aircraft, a fighter jet designed by the French company Dassault Aviation in the late 1970s. The **Mirage 2000** is primarily used by the French Air Force, which received 315 units, while 286 others were exported to eight different countries.

The dataset contains all types of images—some clearer than others, images with one or multiple aircraft, and others with aircraft from different classes. This allows us to test the model's ability to detect and classify aircraft in **varied contexts**.

## Model Training

To train a **YOLOv8 model**, it is essential to have a labeled dataset that will serve as the basis for the model’s learning. In this project, we used a dataset containing 14,500 images of military aircraft, each annotated with the `coordinates` of the aircraft present and their `classification`.

### Train Validation Test Split

Before starting the training, we split our dataset into three distinct parts: a training set **(70%)**, a validation set **(15%)**, and a test set **(15%)**. This division allows us to check the model's performance on unseen data and ensure it generalizes well to images it has not seen before.


```mermaid
graph TD
    B1 -->|70%| B[Ensemble d'entraînement]
    C1 -->|15%| C[Ensemble de validation]
    D1 -->|15%| D[Ensemble de test]

    subgraph Dataset
        direction LR
        B1[████████████████████████████████████████████████████████████████████████████████████████████████████████]
        C1[██████████████████]
        D1[██████████████████]
    end
```


### Training Steps

#### 1. **Model Configuration**  
A YAML configuration file is created to specify:

- The paths to the training, validation, and test sets.
- The number of classes to detect.
- The class names.

<div class="box-info" markdown="1">

<div class="title"> Example Configuration: </div>

```yaml 
train: ../data/train.txt
val: ../data/val.txt
test: ../data/test.txt
nc: 3
names: ['F35', 'JAS39', 'B52']
```

</div>

#### 2. **Model Architecture**  
YOLOv8 uses a convolutional neural network (CNN) architecture with several layers:  

- **Convolutional layers**: To extract features from the images.  
- **Activation layers**: To introduce non-linearity.  
- **Pooling layers**: To reduce dimensionality.  
- **Prediction layers**: To generate predictions for bounding boxes and classes.  

Each layer is designed to capture specific information from the images and combine it to produce accurate predictions.  

#### 3. **Training Process**   
During training, the model goes through the following steps:   

- **Forward propagation**: The image passes through the model's layers, producing predictions.  
- **Loss calculation**: The difference between the predictions and the actual annotations is calculated. YOLO's loss function combines classification errors, localization errors, and missing objects.  
- **Backward propagation**: The loss gradients are calculated and used to update the model's weights through the gradient descent algorithm.  
- **Validation**: After each epoch, the model is evaluated on the validation set to adjust hyperparameters and prevent overfitting.  


## Model Validation

After training, the model is evaluated on the test set to measure its performance. The following metrics are used to assess the quality of the predictions:

- **Box(P)**: Precision of the bounding box detections.
- **R**: Recall, measuring the model's ability to find all relevant instances.
- **mAP50**: Mean Average Precision at 50% IoU (Intersection over Union), measuring the average precision at a 50% IoU threshold.
- **mAP50-95**: Mean Average Precision at different IoU thresholds, from 50% to 95%.

These metrics evaluate the model's ability to detect and classify military aircraft with precision and recall, while minimizing false positives and false negatives.

<div class="box-info" markdown="1">

<div class="title"> Here is an example code to evaluate the model on the test set: </div>

```python
!yolo val \
model='/kaggle/input/yolov7-military-plane/yolov8-m-best.pt' \
data='/kaggle/working/ultralytics/data/mad.yaml'\
augment \
batch=12 \
imgsz=1280
```

</div>

This code is used to evaluate the model on the validation set, using the trained model and the parameters specified in the YAML configuration file.

Here are the results of the model evaluation on the validation set:

<div class="box-warning" markdown="1">

<div class="title"> Validation Results (Scroll to see all classes) </div>

<div style="overflow-x:auto; max-height:600px;">
  <table>
    <thead>
      <tr>
        <th>Class</th>
        <th>Images</th>
        <th>Instances</th>
        <th>Box(P)</th>
        <th>R</th>
        <th>mAP50</th>
        <th>m</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>all</td><td>2024</td><td>3408</td><td>0.944</td><td>0.86</td><td>0.938</td><td>0.89</td></tr>
      <tr><td>A10</td><td>48</td><td>82</td><td>0.961</td><td>0.927</td><td>0.953</td><td>0.902</td></tr>
      <tr><td>A400M</td><td>46</td><td>67</td><td>0.938</td><td>0.866</td><td>0.951</td><td>0.883</td></tr>
      <tr><td>AG600</td><td>34</td><td>35</td><td>0.994</td><td>1.0</td><td>0.995</td><td>0.981</td></tr>
      <tr><td>AV8B</td><td>37</td><td>65</td><td>0.963</td><td>0.985</td><td>0.986</td><td>0.97</td></tr>
      <tr><td>B1</td><td>49</td><td>67</td><td>0.929</td><td>0.91</td><td>0.951</td><td>0.912</td></tr>
      <tr><td>B2</td><td>52</td><td>68</td><td>0.95</td><td>0.844</td><td>0.96</td><td>0.855</td></tr>
      <tr><td>B52</td><td>50</td><td>64</td><td>0.979</td><td>0.922</td><td>0.967</td><td>0.929</td></tr>
      <tr><td>Be200</td><td>32</td><td>35</td><td>0.981</td><td>1.0</td><td>0.995</td><td>0.928</td></tr>
      <tr><td>C130</td><td>93</td><td>180</td><td>0.87</td><td>0.928</td><td>0.945</td><td>0.885</td></tr>
      <tr><td>C2</td><td>107</td><td>146</td><td>0.966</td><td>0.966</td><td>0.991</td><td>0.973</td></tr>
      <tr><td>C17</td><td>59</td><td>88</td><td>0.9</td><td>0.82</td><td>0.927</td><td>0.844</td></tr>
      <tr><td>C5</td><td>50</td><td>50</td><td>0.944</td><td>0.88</td><td>0.947</td><td>0.925</td></tr>
      <tr><td>E2</td><td>47</td><td>67</td><td>0.938</td><td>0.905</td><td>0.936</td><td>0.898</td></tr>
      <tr><td>E7</td><td>16</td><td>18</td><td>1.0</td><td>0.976</td><td>0.995</td><td>0.974</td></tr>
      <tr><td>EF2000</td><td>56</td><td>84</td><td>0.929</td><td>0.726</td><td>0.885</td><td>0.832</td></tr>
      <tr><td>F117</td><td>35</td><td>46</td><td>1.0</td><td>0.822</td><td>0.904</td><td>0.852</td></tr>
      <tr><td>F14</td><td>38</td><td>68</td><td>0.922</td><td>0.824</td><td>0.884</td><td>0.843</td></tr>
      <tr><td>F15</td><td>102</td><td>196</td><td>0.9</td><td>0.934</td><td>0.957</td><td>0.914</td></tr>
      <tr><td>F16</td><td>135</td><td>223</td><td>0.889</td><td>0.78</td><td>0.873</td><td>0.793</td></tr>
      <tr><td>F18</td><td>95</td><td>207</td><td>0.945</td><td>0.865</td><td>0.948</td><td>0.869</td></tr>
      <tr><td>F22</td><td>56</td><td>90</td><td>0.905</td><td>0.842</td><td>0.918</td><td>0.893</td></tr>
      <tr><td>F35</td><td>113</td><td>147</td><td>0.913</td><td>0.862</td><td>0.937</td><td>0.871</td></tr>
      <tr><td>F4</td><td>58</td><td>86</td><td>0.944</td><td>0.849</td><td>0.916</td><td>0.854</td></tr>
      <tr><td>JAS39</td><td>51</td><td>81</td><td>0.935</td><td>0.852</td><td>0.943</td><td>0.885</td></tr>
      <tr><td>MQ9</td><td>34</td><td>36</td><td>0.898</td><td>0.732</td><td>0.899</td><td>0.852</td></tr>
      <tr><td>Mig31</td><td>39</td><td>65</td><td>0.947</td><td>0.825</td><td>0.938</td><td>0.897</td></tr>
      <tr><td>Mirage2000</td><td>30</td><td>75</td><td>0.95</td><td>0.853</td><td>0.87</td><td>0.841</td></tr>
      <tr><td>P3</td><td>34</td><td>63</td><td>0.971</td><td>0.529</td><td>0.799</td><td>0.756</td></tr>
      <tr><td>RQ4</td><td>52</td><td>65</td><td>0.918</td><td>0.864</td><td>0.948</td><td>0.869</td></tr>
      <tr><td>Rafale</td><td>59</td><td>98</td><td>0.887</td><td>0.857</td><td>0.931</td><td>0.89</td></tr>
      <tr><td>SR71</td><td>25</td><td>42</td><td>0.896</td><td>0.786</td><td>0.955</td><td>0.887</td></tr>
      <tr><td>Su34</td><td>48</td><td>62</td><td>0.937</td><td>0.871</td><td>0.953</td><td>0.907</td></tr>
      <tr><td>Su57</td><td>41</td><td>72</td><td>0.971</td><td>0.929</td><td>0.979</td><td>0.945</td></tr>
      <tr><td>Tu160</td><td>40</td><td>54</td><td>0.959</td><td>0.926</td><td>0.97</td><td>0.921</td></tr>
      <tr><td>Tu95</td><td>26</td><td>36</td><td>0.965</td><td>0.758</td><td>0.91</td><td>0.877</td></tr>
      <tr><td>Tornado</td><td>43</td><td>63</td><td>0.961</td><td>0.774</td><td>0.924</td><td>0.882</td></tr>
      <tr><td>U2</td><td>38</td><td>44</td><td>0.976</td><td>0.934</td><td>0.987</td><td>0.956</td></tr>
      <tr><td>US2</td><td>84</td><td>90</td><td>0.981</td><td>0.944</td><td>0.981</td><td>0.942</td></tr>
      <tr><td>V22</td><td>68</td><td>100</td><td>0.988</td><td>0.847</td><td>0.947</td><td>0.853</td></tr>
      <tr><td>XB70</td><td>20</td><td>20</td><td>0.98</td><td>0.9</td><td>0.938</td><td>0.88</td></tr>
      <tr><td>YF23</td><td>13</td><td>18</td><td>0.968</td><td>0.833</td><td>0.953</td><td>0.941</td></tr>
      <tr><td>Vulcan</td><td>46</td><td>69</td><td>0.961</td><td>0.754</td><td>0.911</td><td>0.851</td></tr>
      <tr><td>J20</td><td>47</td><td>76</td><td>0.906</td><td>0.776</td><td>0.896</td><td>0.856</td></tr>
    </tbody>
  </table>
</div>

</div>


**Overall Performance:**

The metrics show that the model has **good precision** (0.944) and **recall** (0.86).

- The **mAP50** and **mAP50-95** values indicate high performance for most aircraft classes, with scores close to or above **0.9**, demonstrating the model’s effectiveness in detecting aircraft in images.
  
- However, it's important to note that some classes have **lower scores**, which could be due to:
  - Variations in the training data.
  - Specific characteristics of certain aircraft.

**Opportunities for Improvement:**

These results suggest that we can **further improve the model** by:
  
- Adjusting the **hyperparameters**.
- Collecting more data for **underrepresented classes**.

### Validation Results Visualization

To better understand the model's performance, we can visualize the results of the predictions on the validation set.

<div class="box-info" markdown="1">

<div class="title"> Confusion Matrix </div>

![Confusion Matrix](/assets/img/AirCraft/conf_matrix_normalized.png)

The confusion matrix shows the model's performance in classifying aircraft classes. The diagonal elements represent the true positives, while the off-diagonal elements represent the false positives and false negatives. The matrix provides insights into the model's ability to distinguish between different aircraft classes and identify common errors. 

</div>

<div class="box-info" markdown="1">

<div class="title"> Precision-Recall Curve </div>

![Precision-Recall Curve](/assets/img/AirCraft/PR_curve.png)

The precision-recall curve illustrates the trade-off between precision and recall for different classification thresholds. A higher area under the curve (AUC) indicates better performance, with a balance between precision and recall. The curve helps evaluate the model's ability to detect aircraft while minimizing false positives.

</div>

## Testing the Model on Images

### Test on Test Dataset

After training and validating the model, we can test it on real images to evaluate its performance under real-world conditions. Here is an example of code to test the model on some images from the test dataset

<details class="details-block" markdown="1">
<summary> Click to see the code </summary>

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import glob
import random
import pandas as pd

model = YOLO('/kaggle/input/yolov7-military-plane/yolov8-m-best.pt')

test_image_paths = sorted(glob.glob('/kaggle/working/ultralytics/data/test/images/*.jpg'))
test_annotation_paths = sorted(glob.glob('/kaggle/working/ultralytics/data/test/labels/*.txt'))

sampled_indices = random.sample(range(len(test_image_paths)), 5)
sampled_image_paths = [test_image_paths[i] for i in sampled_indices]
sampled_annotation_paths = [test_annotation_paths[i] for i in sampled_indices]

results = model(sampled_image_paths)

for img_path, ann_path, result in zip(sampled_image_paths, sampled_annotation_paths, results):
    image = cv2.imread(img_path)[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    with open(ann_path, 'r') as f:
        annotations = f.readlines()
    
    print(f"Real Annotations for {img_path}:")
    for annotation in annotations:
        class_num, x_center, y_center, b_width, b_height = map(float, annotation.split())
        class_name = model.names[int(class_num)]
        print(f"Class: {class_name}")

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() 
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
        ax.add_patch(rect)
        plt.text(x1, y1, model.names[int(box.cls[0])], color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()
```

The code loads the **trained YOLOv8 model**, selects a few random images from the test dataset, makes predictions on these images, and displays the results. The ground truth annotations are also printed for comparison with the model's predictions.

</details>

### Test on Real Images

In addition to the test dataset images, we can also test the model on **out-of-sample images** to assess its ability to generalize to new data. I selected an image of a Rafale from Google Images to test the model.

Here is the Rafale image used for the test and the model's prediction:

![Rafale](/assets/img/AirCraft/RafalePred.jpg)

We can see that the model correctly detected and classified the two Rafale aircraft in the image, with accurate bounding boxes and correct class predictions.

## Testing the Model on Videos

In addition to images, the YOLOv8 model can also be used to detect aircraft **in videos**. I tested the model on a promotional video of the Rafale on YouTube. Here is the link to the video: [Rafale Video](https://www.youtube.com/watch?v=OCghuDF5sec)

After downloading the video, I extracted frames at regular intervals and used the YOLOv8 model to detect the aircraft in each frame. Here is an example of code to detect aircraft in a video:

<details class="details-block" markdown="1">
<summary> Click to see the code </summary>


```python
import cv2
import numpy as np
from ultralytics import YOLO

F22_video_path = "/kaggle/input/videos/F-22 Raptor with Wall of Fire - Dayton Air Show 72323.mp4"

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_width = 640
    new_height = int(new_width * height / width)

    fourcc = cv2.VideoWriter_fourcc(*'VP90') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (new_width, new_height))

        results = model(resized_frame)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

output_path = '/kaggle/working/annotated_F22_video.mp4'

process_video(F22_video_path, output_path)
```

</details>

Here is an example of the video annotated with YOLOv8 detections:

<div style="text-align: center;">
<video width="640" height="360" controls>
  <source src="/assets/vid/AirCraft/rafalevideo.mp4" type="video/mp4">
</video>
</div>

The video shows **YOLOv8 detections** on the **Rafale** presentation video, with **bounding boxes** and **class predictions** for each detected aircraft. The model is capable of detecting moving aircraft in the video, demonstrating its ability to **process video sequences in real-time**.

However, we can observe that the model sometimes struggles to detect aircraft when they are **partially hidden** or **seen from behind**. For example, in the video, the model has difficulty detecting Rafale aircraft when they are viewed from behind, and it tends to **confuse them with other aircraft**, particularly the **EF2000** and the **Tornado**.

This highlights the **importance of training data quality** and the **diversity of examples** to improve the model's performance under various conditions.

We can try the model on **another video** where the **aircraft visibility is lower** to see if the model can still detect them.

<div style="text-align: center;">
<video width="640" height="360" controls>
  <source src="/assets/vid/AirCraft/F22video.mp4" type="video/mp4">
</video>
</div>

This is an example of a video annotated with YOLOv8 detections on a **F22** presentation video. The model performed relatively well in detecting the aircraft in the video, but it encountered **difficulties** in correctly classifying the aircraft. It seems that the model **confused** the **F22** with other aircraft, notably the **F35**. This may be due to similarities in the **visual characteristics** of the aircraft or variations in the training data.

By looking at the results of the model on the videos, we can see that the model is capable of detecting aircraft in videos, but it could be better by hadding a **temporal component** to the model to improve the tracking of the aircraft, so the model can better understand the **context** of the video and **track the aircraft** more accurately, by looking at the previous probabilities of the classes.

## Conclusion

In this project, I used a **YOLOv8** model to **detect** and **classify** **military aircraft** in **images** and **videos**. The model was **trained** on a dataset containing **14,500 images of military aircraft**, with **detailed annotations** for each image.

After training and validation, the model demonstrated **strong performance** in terms of **precision** and **recall**, with **high scores** for most aircraft classes. The **tests on real images and videos** also showed that the model is capable of **detecting aircraft under various conditions**, although it may sometimes face **challenges with partially hidden aircraft** or when they are captured from **unusual angles**.


