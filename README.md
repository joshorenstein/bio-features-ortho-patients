### Biomechanical features impacting presence of hernia in orthopedic patients
Mixed effect neural network model based on biomechanical features of orthopedic patients. Six features are given <br>
1) Pelvic incidence is defined as the angle between a line perpendicular to the sacral plate at its midpoint and a line connecting this point to the femoral head axis.
2) Pelvic tilt is the orientation of the pelvis in respect to the thighbones and the rest of the body <br>
3) Lumbar lordosis refers to the natural inward curve of your lower back. It's a key element of posture, whether good or bad. <br>
4) Sacral slope is defined as the angle between the horizontal and the sacral plate, and is a critical spinal parameter in the analysis of sagittal balance that has been correlated to the progression of spondylolisthesis <br>
5) The degree of spondylolisthesis is defined as the ratio of the anterior displacement of the affected vertebra to its sagittal diameter, 
multiplied by 100%. <br>
        
###### The model:
An unsupervised machine learning model to classify each of the patients into four clusters. These clusters are used as a fixed effect in the supervised machine learning models. <br>
6 popular machine learning algorithms are fit to the training and test data set with the goal of predicting whether a hernia is present which is designated by being 
either normal or abnormal <br>

The following models were fit: Regression Trees, Random Forest, Support Vector Machine, C5.0, Gradient Boost and Neural Network <br>
Support Vector Machine was chosen as the final model <br>

##### Results:
Support Vector Machine had balanced accuracy of about 90%.
