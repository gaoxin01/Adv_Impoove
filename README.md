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
