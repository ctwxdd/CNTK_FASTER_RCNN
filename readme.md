####Nick Wu 13/Oct./2017####

##########################
### Training the model ###
##########################

#download vgg-16 by model by run 
python ./model/downloadmodel.py VGG16_ImageNet_Caffe

#download dataset from CNTK　repo

#place the dataset under DataSets/building100 and run the annotation tool to generate class_map file
python ./DataSets/annotations/annotations_helper.py

#Start training the model by run
python ./train.py

#if the code report error on cannot import Utils.Cython_modual/*
clone from https://github.com/rbgirshick/py-faster-rcnn and run py-faster-rcnn/lib/setup.py build_ext --inplace
and copy the generated .so or pyd file into ./utils/Cython_modules


##########################
#### Testing the model ###
##########################


#test on image files
place the images to be tested in test_img and run python eval_from_file.py

#Use gpu for training
comment out "C.device.try_set_default_device(C.device.cpu())"

#Run Demo page
Run python ./server/app.py and open 127.0.0.1:3000 to test it