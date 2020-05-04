#import deeplabcut
import os
import h5py
import numpy as np
import pandas as pd
import tables
import statistics
import cv2
###############User input
baseFilePath = 'D:/DLCTrainOpenPose/ForwardRoll'
DLCfolder = 'ForwardRoll-CJC-2020-05-04'
openposeFilePath = 'D:/DLCTrainOpenPose/CJC/ForwardRoll0001_20200501/Intermediate/OpenPoseOutput/OP_FowardRoll.npy'
videoname = 'ForwardRoll'
amountOpenPosePoints = 25
scorerName = 'CJC'
pvalueCutoff = .8

################ Create path varibles based on User Input
DLCpath = baseFilePath +'/'+DLCfolder
videoPath = baseFilePath 
configPath = baseFilePath+'/'+DLCfolder+'/config.yaml'

###################Load in Openpose data
openposeOutput = np.load(openposeFilePath) 

##################Parse through OpenPose data to obtain high pvalue frame numbers
highOpenPosePvals =[]#Create empty high pvalue list
for jj in range(len(openposeOutput)):#iterates through the amount of frames of openpose output
    avg = statistics.mean(openposeOutput[jj,:,2]) #Takes average of each body part p-value
    if avg > pvalueCutoff: #If the average is higher than specified pvalue 
        highOpenPosePvals.append(jj)#Add that frame number to the high pvalue list
#highOpenPosePvals = np.array(highOpenPosePvals) #Make the list an array (may not be necessary)
 
###################Grab Frames from video with high pvalue
datadir =[videoPath]#create directory variable for video folder
for dir in  datadir: #iterates through folder of videos
    for video in os.listdir(dir):#iterates through each video
        vidcap = cv2.VideoCapture(os.path.join(dir,video))#Open video
        vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) #Get video height
        vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) #Get video width
        video_resolution = (int(vidWidth),int(vidHeight)) #Create variable for video resolution
        success,image = vidcap.read() #read a frame
        count = 0 #Intialize a counter variable
        while success: #While there is frames to read
            success,image = vidcap.read() #read a frame
            if success: #If frame is correctly read
                if count in highOpenPosePvals: #If the current frame being read has a high pvalue
                    resize = cv2.resize(image, video_resolution) #Set image to same resolution of video
                    #Save images, this can be improved
                    if str(count) == '0':
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img000000.png' , image)     # save frame as png file   
                    if len(str(count)) == 1:                      
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img00000%d.png' %count , image)     # save frame as png file
                    if len(str(count)) == 2:
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img0000%d.png' %count , image)     # save frame as png file
                    if len(str(count)) == 3:
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img000%d.png' %count , image)     # save frame as png file
                    if len(str(count)) == 4:
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img00%d.png' %count , image)     # save frame as png file
                    if len(str(count)) == 5:
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img0%d.png' %count , image)     # save frame as png file
                    if len(str(count)) == 6:
                        cv2.imwrite(DLCpath+'/labeled-data/'+videoname+'/img%d.png' %count , image)     # save frame as png file
            else: #if the frame is incorrectly read
                continue#Go to nect frame
            count+=1#Increase counter variable
        
###################Sort and reshape high value openpose data to the correct format for DLC 
openPoseFramesSelected = openposeOutput[highOpenPosePvals,:,0:2]#Take the x and y columns of the frames with a high pval 
openPoseFramesSelected = np.array(openPoseFramesSelected)#change from list to array
openPoseSorted =[]#create empty list of the sorted data
for jj in range(len(openPoseFramesSelected)):#iterates through all high pval frames
    openPoseSorting =[]#create empty list for X and Y vals
    for ii in range(amountOpenPosePoints):#iterates through the amount of body points tracked
        openPoseSorting.append(openPoseFramesSelected[jj,ii,0])#grab the x value
        openPoseSorting.append(openPoseFramesSelected[jj,ii,1])#grab the y value
        openPoseTranspose = np.transpose(openPoseSorting)#transpose points so x and y are in columns
    openPoseSorted.append(openPoseTranspose)#add that frame of all openpose points to the sorted list
openPoseSorted = np.array(openPoseSorted)#convert to array
print(openPoseSorted.shape)#Print the shape as a check, should be (amountofhighpvalFrames,(amountofOpenposePoints*2)) 

###################Format the frame numbers for DLC, can be improved
frameIndex =[]#create an empty list for frame names
for jj in highOpenPosePvals:
    if str(jj) == '0':
        number = '000000'
    if len(str(jj)) == 1:
        number = '00000'+str(jj)
    if len(str(jj)) == 2:
        number = '0000'+str(jj)
    if len(str(jj)) == 3:
        number = '000'+ str(jj)  
    if len(str(jj)) == 4:
        number = '00'+str(jj)
    if len(str(jj)) == 5:
        number = '0'+str(jj)
    if len(str(jj)) == 6:
        number = str(jj)  

    frameIndex.append('labeled-data/'+videoname+'/img'+number+'.png') #add the frame number in the correct DLC format to the list

####################Create lists for correct DLC labels
XYName = []# create an empty list for x and y values
IntialName =[]# create an empty initials list
for jj in range(2*amountOpenPosePoints):#iterates through the amount of openpose points*2(*2 because x and y val for each point)
    if jj == 0:#if first point
        XYName.append('x')#it is X value
        IntialName.append(scorerName)#add scorer name
        continue#skip rest of loop
    if (jj%2) ==0:#If it is an even point
        XYName.append('x')#it is an x value 
        IntialName.append(scorerName)#add scorer name
    if not (jj%2) ==0: #if it is an odd point
        XYName.append('y')#it is a y value
        IntialName.append(scorerName)#add scorer name

##################Create list for body points, could be improved instead of hardcoding by importing from config file        
bodypartNames = ['nose','nose','chest','chest','rightShoulder','rightShoulder','rightElbow','rightElbow','rightWrist','rightWrist','leftShoulder','leftShoulder','leftElbow','leftElbow','leftWrist','leftWrist','centerPelvis','centerPelvis','rightHip','rightHip','rightShin','rightShin','rightAnkle','rightAnkle','leftHip','leftHip','leftShin','leftShin','leftAnkle','leftAnkle','rightEye','rightEye','leftEye','leftEye','rightEar','rightEar','leftEar','leftEar','leftInnerFoot','leftInnerFoot','leftOuterFoot','leftOuterFoot','leftHeel','leftHeel','rightInnerFoot','rightInnerFoot','rightOuterFoot','rightOuterFoot','rightHeel','rightHeel' ]
#################Create a list with for the three different rows needed in the DLC format
cols = [IntialName,bodypartNames,XYName]
################ make the openposesorted data into a dataframe
openPoseDF = pd.DataFrame(openPoseSorted, index = frameIndex)
openPoseDF.columns = pd.MultiIndex.from_arrays(cols, names = ['scorer','bodyparts','coords'])#add the three row names to top of datafra

################ save data
filename = baseFilePath+'/'+DLCfolder+'/labeled-data/'+videoname+'/CollectedData_CJC' #filepath to where to save sorted data
openPoseDF.to_hdf(filename+'.h5', key='df_with_missing',format = 'table', mode='w')#write dataframe to h5 file
openPoseDF.to_csv(filename+'.csv')#write dataframe to csv file


