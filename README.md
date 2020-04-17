# Deceiving Autonomous Cars with Toxic Signs #

### Abstract ###
Recent studies show that the cutting edge deep neural networks are vulnerable to contradictory examples,
deriving from small magnitude perturbations added to the input. With the advent of self-driving machines,
the contradictory example, as we can imagine, can generate many complications: a car can interpret a signal
incorrectly and generate an accident. In our project, we want to analyze and test the problem,
showing that it is possible to generate specific perturbations to the input images to confuse the model and, in
some way, force the network prediction.

### Requirements ###
To test our code you can simply install a virtual environment on your machine ([Turorial](https://www.tensorflow.org/install/pip)). 
Then, you can run the following code to install all the necessary libraries:
```
python setup.py
```

### Explanation ###
* **Adversarial_img**: It contains all the disturbed images and the relative CSV in the appropriate folders, based on the last attack made (read the paper for more details) <br/>
* **Blank_samples**: It contains all the empty signs used for the *blank_signs_attack* <br/>
* **Dataset**:  It contains the link to the dataset used in the project <br/>
* **Detector_samples**: It contains the original road images and the respective images of the detected signals (read the paper for more details) <br/>
* **Logo_samples**:  It contains all the logo samples used for the *logo_attack* <br/>
* **Model**:  It contains the link to the trained model<br/>
* **Original_samples**:  It contains a set of high definition images used to generate contradictory images <br/>
* **Aug_examples**: It contains some sample images generated by the augmented functions <br/>
* **Attack.py**:  It contains the general code of the attacks <br/>
* **Call_model.py**:  It contains the architecture of the network with other useful functions to call it from other files <br/>
* **Data_augmentayion.py**:  It contains the code to modify the images and balance the dataset <br/>
* **Detector_phase.py**:  It contains the code for the detection phase <br/>
* **Fg_attack.py**:  It contains the code of the Fast Gradient Attack <br/>
* **Histogram.py**:  It contains the code to plot the histogram of the unbalanced dataset <br/>
* **Iterative_attack.py**:  It contains the code of the Iterative Attack <br/>
* **Parameters.py**:  It contains the list of all parameters used on the project <br/>
* **Real_time_detection.py**:  It contains the code for real-time detection and classification using the webcam  <br/>
* **Requirements.txt**:  It contains the list of all libraries needed to execute the code <br/>
* **Setup.py**:  It contains the code to install all libraries needed to execute the code <br/>
* **Sign_name.csv**:  it contains the correspondence between the class of the dataset, the class of the model and the relative label  <br/>
* **Test.py**: It contains the code to test the model <br/>
* **Train.py**:  It contains the code to train the model <br/>
* **Utils.py**:  It contains all the useful functions in the project <br/>

### Team ###
* [Antonino Di Maggio](https://www.linkedin.com/in/antonino-di-maggio/) 
* [Leonardo Salvucci](https://www.linkedin.com/in/leonardo-salvucci/)  

### Useful links ###
Slide presentation: In progress <br/>
Paper: https://drive.google.com/open?id=1vDJjxpUsyqt0kizyF87VwtZb8NFrJxrr <br/>
