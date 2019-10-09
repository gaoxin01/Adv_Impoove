# Adv_Improve
A Method of Using Existing Database to Improve the Effect of Bi-classification Model(PyTorch implementations).  
# Database Download
Link: https://pan.baidu.com/s/1LlAXxOjx7arl58ne6mtd1w   Extraction code: rxyk 
# Specific steps
## Data storage  
create a folder named `data` in the project, download the database file, and put the extracted folder `cifar-10-batches-py` into the `data` folder.  

## Converting Binary-style databases to image and TXT tag formats
Modify the path of `root` in the DataConvert.py file to make `root = "the path of your own data storage"`.  
Modify the path of `image` to make `image = "the path of your own database image storage"`.  
Modify the path of `labelTxt` to make `label = "the path you want your label to store"`.  
run `DataConvert.py`.  
If the database is the original form of the image without conversion, you do not need to run this file for conversion.  

## Select the required two-category label from the label file  
The tag file generated in the previous step is stored under the `dir` folder, and the txt tag file is processed to get the two-class tag file we need.  
Modify `label` to `label = "Address of the total tag file of the database you generated in the previous step"`.  
run `labelProcess.py`.  
Through this program, we get the tag file `train0.txt` of the first category and the tag file `train1.txt` of the second category from the total tag file of a database, and the tag file `test0_1.txt` of the required two-class test set.

## Train the model and use the method of generating against the sample to improve the model effect
In the program, it is set to generate the anti-samples 50 times. Each time the anti-samples are generated, the samples are added to the existing training set re-training model, and the total iteration is 50 times.  
The generated pictures are placed in the `fooling_img` folder under `data`, the first kind of pictures are placed in the `fooling_img` `class0` folder, and the second kind of pictures are placed in the `fooling_img` `Class1` folder.  
The `fooling_img` folder is automatically generated, but two paths need to be modified because the path of the `data` folder will be different.  
Modify `gen_class0` to `gen_class0 = "Your data folder directory + /fooling_img/class0/"`  
Modify `gen_class1` to `gen_class1 = "Your data folder directory + /fooling_img/class1/"`  
In the terminal, type `python Adv_Improve.py` to run `Adv_Improve.py`.  

## Train the model through all data
Because the same database is used, our method does not expand the data set by initially generating new data separately. However, some anti-samples are modified during the training process, which is different from the original method using gan. Training after the data set. So we need to compare the effects of the models directly trained without this method.  
Regular training model through this program.  
run `normal.py`
