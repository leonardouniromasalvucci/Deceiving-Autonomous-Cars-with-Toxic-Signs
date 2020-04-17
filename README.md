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
* **Blank_samples**: Contains all the empty signs used for the *blank_signs_attack* <br/>
* **Dataset**:  Contains the link to the dataset used in the project <br/>
* **Detector_samples**: It contains the original road images and the respective images of the detected signals (read the paper for more details) <br/>
* **Logo_samples**:  Contains all the logo samples used for the *logo_attack* <br/>
* **Model**:  Contains the link to the trained model<br/>
* **Original_samples**:  Contains a set of high definition images used to generate contradictory images <br/>
* **Aug_examples**:  4t4h <br/>
* **Attack.py**:  4t4h <br/>
* **Call_model.py**:  4t4h <br/>
* **Data_augmentayion.py**:  4t4h <br/>
* **Detector_phase.py**:  4t4h <br/>
* **Fg_attack.py**:  4t4h <br/>
* **Histogram.py**:  4t4h <br/>
* **Iterative_attack.py**:  4t4h <br/>
* **Parameters.py**:  4t4h <br/>
* **Real_time_detection.py**:  4t4h <br/>
* **Requirements.txt**:  4t4h <br/>
* **Setup.py**:  4t4h <br/>
* **Sign_name.csv**:  4t4h <br/>
* **Test.py**:  4t4h <br/>
* **Train.py**:  4t4h <br/>
* **Utils.py**:  4t4h <br/>

### Team ###
* [Antonino Di Maggio](https://www.linkedin.com/in/antonino-di-maggio/) 
* [Leonardo Salvucci](https://www.linkedin.com/in/leonardo-salvucci/)  

### Useful links ###
Slide presentation: In progress <br/>
Paper: https://drive.google.com/open?id=1vDJjxpUsyqt0kizyF87VwtZb8NFrJxrr <br/>
