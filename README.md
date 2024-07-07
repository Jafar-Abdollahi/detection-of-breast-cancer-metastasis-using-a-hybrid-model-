<h2> Discription </h2>
# detection-of-breast-cancer-metastasis-using-a-hybrid-model
Breast cancer (BC) is a prevalent disease and major cause of mortality among women worldwide. A substantial number of BC patients experience metastasis which in turn leads to treatment failure and death. The survival rate has been significantly increased thanks to the state of the art technologies and detection tools. In this study, we cross-compared the application of advanced artificial intelligence algorithms such as Logistic Regression, K-Nearest Neighbors, Discrete Cosine Transform, Random Forest Classifier, Support Vector Machines, Multilayer Perceptron, and Ensemble to diagnose BC metastasis. We further combined MLP with genetic algorithm (GA) as a hybrid method of intelligent analysis. The core data we used for comparison belonged to the images of both benign and malignant tumors and were taken from Wisconsin Breast Cancer dataset from the UCI repository. Our findings indicate that our MLP-GA hybrid algorithm can speed up diagnosis with higher accuracy rate than the individual patterns of algorithm. Two methods of comparison (Cross-Validation and Holdout) were applied to this study which produced consistent results.

Background: Metastasis is the main cause of death toll among breast cancer patients. Since current approaches for diagnosis of lymph node metastases are time-consuming, deep learning (DL) algorithms with more speed and accuracy are explored for effective alternatives. 
Methods: A total of 220025 whole-slide pictures from patient lymph nodes were classified into two cohorts: testing and training. For metastatic cancer identification, we employed hybrid convolutional network models. The performance of our diagnostic system was verified using 57458 unlabeled images that utilized criteria that included accuracy, sensitivity, specificity, and p-value. 
Results: The DL-based system that was automatically and exclusively capable of quantifying and identifying metastatic lymph nodes was engineered. Quantification was made with 98.84% accuracy. Moreover, the precision of VGG16 and Recall was 92.42% and 91.25%, respectively. Further experiments demonstrated that metastatic cancer differentiation levelS could influence the recognition performance. 
Conclusions: Our engineered diagnostic complex showed an elevated level of precision and efficiency for lymph node diagnosis. Our innovative DL-based system has a potential to simplify pathological screening for metastasis in breast cancer patients.


<h2> Dataset </h2>
<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/Picture1.jpg"> 
<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/Picture2.jpg"> 
<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/Picture3.jpg"> 
<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/Picture4.jpg"> 


<h2> Methods </h2>
In this study, we applied a hybrid method of CNN-based classification for classifying our images. Two well-known deep and pre-trained CNN models (Resnet50, VGG16) and two fully-trained CNNs (Mobile-net, Google net) were employed for transfer learning and full training. In order to train the CNN, we used open-source DL-based Tensorflow and Keras libraries. The models’ performance was analyzed in terms of accuracy, sensitivity, specificity, receiver operating characteristic curves, areas under the receiver operating characteristic curve, and heat maps.
Datasets
We utilized the Camelyon16 dataset consisting of 400 hematoxlyin and eosin (H&E) whole-slide images of lymph nodes, with metastatic regions labeled. The images are in Portable network graphics (PNG) format and can be downloaded here: “https://www.kaggle.com/c/histopathologic-cancer-detection/data.” Furthermore, the data has two folders related to testing and training images as well as a training labels file. There are 220k training images, with a roughly 60/40 split between negatives and positives, and 57k evaluation images.  
In the dataset, scientists are endowed with plenty of small pathological images in order to categorize. An image ID is assigned to each file. The ground truth specifically for the images located in the train folder is provided by a file named train_labels.csv. Indeed, scholars forecast the labels that are for the images within the test folder. To be more specific, a positive label demonstrates that the patch center region (32x32 px) includes minimum one pixel of tumor-associated tissue. Moreover, tumor tissue located in the patch’s outer area does not affect the label.
Preprocessing
One of the essential elements for categorizing histological images is pre-processing. The dataset images are fairly large, while CNNs are normally designed in order to receive significantly smaller inputs. Hence, the images’ resolution needs to be diminished, thereby being able to take in the input while keeping the key features. The dataset size is considerably lower than what is commonly needed to train a DL model appropriately; therefore, data augmentation is employed to raise the unique data amount in the set. In fact, this method greatly contributes to avoiding overfitting that is a phenomenon by which the model absorbs the training data properly, albeit being completely unable to categorize and generalize unseen images 23, 24.
Data Augmentation
The combinations of approaches supplied by the Keras library were examined to see the influence on overfitting and their contribution to enhancing categorization accuracy. Analyzing histological images is rotationally invariant, meaning that it does not take into account the angle at which a microscopy image is viewed. Consequently, employing rotation augmentation for the image should not affect the architecture training negatively 23.
Ensemble Deep-Learning Approach for Detecting Metastatic Breast Cancer: Proposed Method
Innovation: Layer-wise fine-tuning and different weight initialization schemes.
In this study, we propose an autonomous classification method for BC classification. We used two pre-trained methods (VGG16, Resnet50) and two fully-trained ones (Google net and Mobile net) for our classification study. The models have been previously trained using the ImageNet database, which can be retrieved from the image-classification library of TensorFlow-Slim (http://tensorflow.org). Finally, we compared the outputs of the algorithms used in the pre-trained period and those of fully trained period in order to adequately evaluate the function of all models applied for classification of breast histopathologic images into benign (B) and malignant (M) in precise diagnosis of BC metastasis. 



<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/Picture5.png"> 


<h2> Conlusion </h2>
The comparison of four various CNN models possessing depths between three to thirteen convolutional layers was performed in this research. First of all, our empirical outcomes indicated that initializing parameters of a network with transferred features could enhance the categorization performance for any model. Nevertheless, deeper architectures that were trained on larger datasets converged rapidly. Furthermore, learning from scratch needs more training time than a pre-trained network model. Considering this matter, fine-tuned pre-trained VGG16 produced the best performance with 98.84% precision, 96.01% AUC, and 92.42 % APS for 90%–10% testing-training data splitting.  

<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/Picture6.png"> 
<img src="https://github.com/Jafar-Abdollahi/detection-of-breast-cancer-metastasis-using-a-hybrid-model-/blob/main/2024-07-07_19-58-39.png"> 

<h2> Paper </h2>
Abdollahi, J., Keshandehghan, A., Gardaneh, M., Panahi, Y., & Gardaneh, M. (2020). Accurate Detection of Breast Cancer Metastasis Using a Hybrid Model of Artificial Intelligence Algorithm. Archives of Breast Cancer, 18-24.


<h2> Contact me </h2>
You can reach me at:

Email: ja.abdollahi77@gmail.com
<br>
LinkedIn: https://www.linkedin.com/in/jafar-abdollahi-7647971b3/
<br>
Google Scholar: https://scholar.google.com/citations?user=2dK8kPwAAAAJ&hl=en
<br>
researchgate: https://www.researchgate.net/profile/Jafar-Abdollahi?ev=hdr_xprf&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImxvZ2luIiwicGFnZSI6ImhvbWUiLCJwcmV2aW91c1BhZ2UiOiJsb2dpbiIsInBvc2l0aW9uIjoiZ2xvYmFsSGVhZGVyIn19
<br>
youtube: https://www.youtube.com/@jafarabdollahi/featured
<br>
<img src="https://github.com/Jafar-Abdollahi/cuffless-bp-master-in-python-jupyter-/blob/main/2024-07-07_19-45-22.png"> 
