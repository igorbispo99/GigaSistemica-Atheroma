# GigaSistemica-Atheroma

- **Description:** This study introduced a hybrid deep learning model to detect and segment carotid atheroma calcifications (CACs) in dental panoramic radiographs (PR). The proposed method combines the efficiency of an attention mechanism with the precision of the UNet architecture. It enables the identification and segmentation of CACs by utilizing an automated two-step process, offering a reliable and practical tool for opportunistic screening in dental settings. The model facilitates the detection of potential atherosclerosis-related risks, providing an accessible, non-invasive, and cost-effective diagnostic approach.

- **Status:** Under review at *IEEE Journal of Biomedical and Health Informatics (JBHI)*.
- **Publication:**
  - Correia, I. B. M. C., Ferreira, M. V. S., Chini, C. F., Dias, B. S. S., Costa, L. R., Caetano, M. F., Leite, A. F., de Melo, N. S., & Farias, M. C. Q. (2024). *Detection and segmentation of carotid atheroma calcification in dental panoramic radiographs using a hybrid deep learning model*. Submitted to **IEEE Journal of Biomedical and Health Informatics**.

![](https://i.imghippo.com/files/hr1992jZI.png)

## How to Run

Use the following command structure to run the model:

```sh
python main.py --input_path <path_to_image_or_folder>
--output_path <directory_for_saving_results>
[--proceed_when_negative]
--classifier_model_path <path_to_classifier_model>
--detection_model_path <path_to_detection_model>
--segmentation_model_path <path_to_segmentation_model>
--class_names <class_name_1> <class_name_2> ...
```

Where:
- **--input_path**: Required path to a single image or a folder of images.  
- **--output_path**: Where results will be saved (defaults to `results`).  
- **--proceed_when_negative**: Optional flag indicating whether to continue with detection and segmentation even if the classification result is negative.  
- **--classifier_model_path**: Path to the classification model (default provided).  
- **--detection_model_path**: Path to the detection model (default provided).  
- **--segmentation_model_path**: Path to the segmentation model (default provided).  
- **--class_names**: Names of classes used by the classifier (default is `Nao_Ateroma Ateroma`).


## Output Details

Upon execution, the program generates two main outputs for each image processed. An output image (`_output.png`) is saved, where detected regions are outlined and potential segments are highlighted in red with adjustable transparency. In addition, a corresponding JSON file (`_output.json`) is created to record classification results, detection bounding boxes, detection scores, and class labels. This combination offers a visual reference in the form of modified images and a structured data report of predictions. 