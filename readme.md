# YOLOv5 Object Detection and Instance Segmentation

This repository contains custom implementations of YOLOv5 for object detection and instance segmentation, showcasing the versatility and power of YOLOv5 in handling various computer vision tasks. The project is structured into two main Jupyter notebooks: one for object detection on Road Sign Dtataset and another for instance segmentation on a Sign Recognition Dataset.

## Project Structure

### Object_Detection.ipynb
Demonstrates the use of YOLOv5 for detecting objects within images. This notebook covers the entire workflow from dataset preparation, model training, evaluation, and inference.

### Instance_Segmentation.ipynb

Applies YOLOv5 for instance segmentation tasks, focusing on segmenting individual signs from images. The notebook includes steps for data preprocessing, model adaptation for segmentation, training with transfer learning, and result visualization.

## Datasets

This project utilizes two distinct datasets to tackle the challenges of object detection and instance segmentation within the context of sign recognition:

1. Road Traffic Sign Data for Detection
The first dataset focuses on the detection of road traffic signs. It contains a wide range of traffic sign images annotated with bounding boxes, facilitating the training and evaluation of our object detection model.

2. Sign Recognition Data for Segmentation
The second dataset is tailored for instance segmentation tasks, concentrating on sign recognition. This dataset includes images of various signs with detailed pixel-level annotations, enabling the precise segmentation of individual signs.

Integration with [Roboflow](https://roboflow.com)
We leverage Roboflow to manage, preprocess, and augment our datasets. Roboflow is a comprehensive tool that simplifies the dataset management process, offering functionalities like annotation conversion, image augmentation, and version control. By using Roboflow, we ensure that our datasets are optimized for training, which is crucial for achieving high model performance.

To access and use the datasets directly within our project, Roboflow provides an API key. This allows for seamless integration of dataset downloading and preparation steps into our Jupyter Notebooks. By using the Roboflow API, we can easily fetch the latest version of our datasets formatted specifically for YOLOv5, ensuring our models are trained on up-to-date and well-prepared data.

![Ground Truth Labels](/yolo_v5_detection/images/label.png)

![Augmented Training Data](/yolo_v5_detection/images/aug_train_data.png)

## Getting Started

- To get started with this project, clone the repository to your local machine or directly to your preferred development environment:
git clone https://github.com/suniash/yolov5.git

- Install the required dependencies listed in requirements.txt

- Run the Jupyter Notebook to train the model

## Results and Metrics

### Object Detection
![](/yolo_v5_detection/images/metrics.png)

![](/yolo_v5_detection/images/infer2.png)

### Instance Segmentation
![](/yolov5_segmentation/images/metrics.png)

![](/yolov5_segmentation/images/results.png)


## Contributing
Contributions to this project are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The YOLOv5 team for the powerful and efficient model. The creators of the  datasets used in this project and Roboflow platform for seamless integration of datasets in notebooks. 