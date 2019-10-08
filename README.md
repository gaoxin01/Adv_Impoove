# Adv_Improve
A Method of Using Existing Database to Improve the Effect of Bi-classification Model(PyTorch implementations).
# Database Download
Link: https://pan.baidu.com/s/1LlAXxOjx7arl58ne6mtd1w   Extraction code: rxyk 
# Specific steps
Data storage
create a folder named "data" in the project, download the database file, and put the extracted folder "cifar-10-batches-py" into the "data" folder.

Converting Binary-style databases to image and TXT tag formats (DataConvert.py)
Modify the path of root in the DataConvert.py file to make root = "the path of your own data storage". 
Modify the path of image to make image = "the path of your own database image storage". 
Modify the path of labelTxt to make label = "the path you want your label to store".
run DataConvert.py.
If the database is the original form of the image without conversion, you do not need to run this file for conversion.

Extract the label data we need from the label files of all the generated data and modify the label values (labelProcess.py)
The tag file generated in the previous step is stored under the dir folder, and the txt tag file is processed to get the two-class tag file we need.
Modify "label = " to label = "Address of the total tag file of the database you generated in the previous step".
run labelProcess.py.
