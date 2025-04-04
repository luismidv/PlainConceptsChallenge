<h2>PLAIN CONCEPTS TECHNICAL CHALLENGE</h2>

The challenge consists on implemening a solution for an electrical company that needs to automate the identification of electric components in circuits schema.

<h3>Example of images and the items to locate:</h3> <br/>

![296_png rf 2fa9d963874a376b48a1d10fa6b685f8](https://github.com/user-attachments/assets/fc6eb4c2-c46e-41ec-a1fe-2f246faf3fa9)

<h3>The project was developed following the next steps:</h3></br>
  1. Analysis and first view on the dataset.</br>
  &nbsp;&nbsp;&nbsp;1.1 Class distribution</br>
  &nbsp;&nbsp;&nbsp;1.2 Image width-height differences and resizing</br>
  2. Labeling</br>
    &nbsp;&nbsp;&nbsp;2.1 Usage of label-studio tool to get each image labelled in XML files (PASCAL VOC FORMAT).</br>
  3. Data preparation</br>
     &nbsp;&nbsp;&nbsp;3.1 Prepare a dataset class to iterate using dataloder(First approximation)</br>
     &nbsp;&nbsp;&nbsp;3.2 Prepare a folder ("dataset") with images and labels for train, valid and test steps.</br>
     &nbsp;&nbsp;&nbsp;3.3 Transform the XML files to .txt files that include class, x_min, y_min, w, h</br>
  4. Fine-tune yolov8 model</br>
     &nbsp;&nbsp;&nbsp;4.1 First fine-tune.</br>
     &nbsp;&nbsp;&nbsp;4.2 Test the model on unseen data.</br>
     &nbsp;&nbsp;&nbsp;4.3 Second fine-tune</br>
     &nbsp;&nbsp;&nbsp;4.4 Re-test the model on unseen data.</br>
</br>
The algorithm was trained in Google Colab since HuggingFace free plan doesn't, stops the service because of too high-load for that plan.<br/>

<h3> About each .py file structure</h3>
data_analysis:  where the main data analysis and data preparation work is done. <br/>
dataprepare: dataset preparation for the first aproximation, not used in the final project.<br/>
train: where the training, testing steps are done.<br/>
app.py: file created to implement API part, not finished.<br/>
  
     

BEST TRAINING METRICS:</br>

LOSS METRICS</br>

![Unknown](https://github.com/user-attachments/assets/5d009406-8405-4846-af9c-cdbf19b4e637)

ACCURACY METRICS</br>

![Unknown-2](https://github.com/user-attachments/assets/4fd3d48b-2970-413a-88dd-b15fe959efff)

LEARNING RATE METRICS</br>

![Unknown-3](https://github.com/user-attachments/assets/021c7d9c-f520-4a95-8c10-b6da2f25c8d5)

UNSEEN DATA TESTING:</br>

![Unknown-4](https://github.com/user-attachments/assets/5dc785d2-1e57-48d1-a44e-82d9ff9c8827)

![Unknown-5](https://github.com/user-attachments/assets/94fa71ea-7430-48d4-bca0-c19528988688)

