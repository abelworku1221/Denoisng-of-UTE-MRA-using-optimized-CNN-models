# Denoisng-of-UTE-MRA-using-optimized-CNN-models
This Repository contains the python code and a high resolution pre-clinical contrast-enhanced UTE-MRA dataset used for image denoising as well as other research topics.  

Convolutional neurtal networks used

![image](https://github.com/user-attachments/assets/f3782f22-ff14-494c-9308-efa187af126a)

Sample result

![image](https://github.com/user-attachments/assets/e98f2268-55d5-4299-b555-42a4cd7ee52b)


Steps to use the repository

step1. download the data provided at: https://unistackr0-my.sharepoint.com/:u:/g/personal/abelworku1221_unist_ac_kr/EXHm3MP02ihHpEW8mpPegeUB_Bxm2FaQPrSaaqkuzE4s2w?e=4Fti0Q.
       extract it and place it in side the corresponding folders "Data/Training/" and "Data/Validation/"  for training and testing. check the path in the "dataloader.py" file and confirm if it is okay.

Step2. Open the "Train_Denoising_models.py" file and import any model among 5 modesl created inside "Models/" folder. you can change training parameters such as batch and number of epoches.

step3. After training. open the "testing denoisng.py" file and read the corresponding trained file and run the script. it will generate the result files in the folder "Result/..." depending on which model used and the imput data name.

lastly, if you want to cite our work, please cite as follows:

