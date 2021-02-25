### Biomechanical features impacting presence of hernia in orthopedic patients
A mixed effect neural network model based on biomechanical features of orthopedic patients. Five features are given; pelvic incidence,
pelvic tilt, lumbar lordosis, sacral slope and the degree of spondylolisthesis  <br>

###### The model:
An unsupervised machine learning model is used to classify each of the patients into four clusters. A quick exploratory analysis of the data shows that degree of spondylolisthesis has a major impact on the dependent variable. These clusters are then used as a fixed effect in the supervised machine learning models with the goal of reducing overfit that could be caused by the degree of spondylolisthesis having great variable importance. <br>. Six popular machine learning algorithms are fit to the training and test data set with the goal of predicting whether a hernia is present which is designated by being 
either normal or abnormal. Those models are Regression Trees, Random Forest, Support Vector Machine (with and without fixed effect), C5.0, Gradient Boost and Neural Network. <br>

##### Results:
```
model             training_accuracy test_accuracy difference
  <chr>                         <dbl>         <dbl>      <dbl>
1 C50                           100            87       -13   
2 Decision_Tree                  86.2          81.8      -4.4 
3 Gradient_Boost                 88.8          83.1      -5.7 
4 Neural_Net                     85.3          85.7       0.4 
5 Random_Forest                 100            87       -13   
6 Support_Vector                 82.3          88.3       6   
7 Support_Vector_No              81.9          85.7       3.80
```

[Median Biomechanics by Person Type](https://github.com/joshorenstein/bio-features-ortho-patients/blob/master/results/results.csv) <br/>
[Regression Tree](https://github.com/joshorenstein/bio-features-ortho-patients/blob/master/charts/regression-tree-model.pdf) <br/>
[SVM Variable Importance](https://github.com/joshorenstein/bio-features-ortho-patients/blob/master/results/variable_importance.pdf) <br/>

###### Conclusions:
* In the supervised algorithms, the best models on predicting the test data were Support Vector Machine, C5.0 and Random Forest. 
* The C5.0 and Random Forest had 100% accuracy on the training data likely due to the models being overfit. 
* The neural network had the most stable accuracy, ie, the smallest difference between the predictions of the training and test sets. 
* A regression tree model is always worth considering as it is the easiest model to interpret; however, it's performance on the test set was near the bottom. 
* It's fairly easy to extract variable importance from the Support Vector Machine models which makes it useful for explaining findings to non-technical people as well. <br/>

The Neural Network is the model I would most likely fit on this data moving forward. It is not the easiest data to visualize, but given its stable results and high accuracy, among the models tested, I think it is the best fit for this data. <br/>


###### The data:
1) Pelvic incidence is defined as the angle between a line perpendicular to the sacral plate at its midpoint and a line connecting this point to the femoral head axis. <br>
2) Pelvic tilt is the orientation of the pelvis in respect to the thighbones and the rest of the body <br>
3) Lumbar lordosis refers to the natural inward curve of your lower back. It's a key element of posture, whether good or bad. <br>
4) Sacral slope is defined as the angle between the horizontal and the sacral plate, and is a critical spinal parameter in the analysis of sagittal balance that has been correlated to the progression of spondylolisthesis <br>
5) The degree of spondylolisthesis is defined as the ratio of the anterior displacement of the affected vertebra to its sagittal diameter, 
multiplied by 100%. <br>
        

