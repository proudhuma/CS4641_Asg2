# GaTech CS 4641 Machine Learning
## Project 2. Randomized Optimization

Student Name: HU Heng
  
GT ID: 903478698

[github repo](https://github.com/proudhuma/CS4641_Asg2)
-------------

### Files:
folder data contains following files:
 - adult.data: training data for adult dataset
 - adult.test: test data for adult dataset
  
folder out contains output files, the files are in csv format.

folder report contains the .tex file for the report.

folder plot contains .jpg figures.

folder problems contains java file for three self defined problems.

randomized.py is for training neural networks

plot.py is for ploting figures

------------
### Requirements:
 - make sure the python version is 3.6
 - make sure mlrose, numpy, pandas and matplotlib are installed

------------
### Run:

    # For part 1
    python3 ./randomized.py  <algorithm> # random_hill_climb OR simulated_annealing OR genetic_alg
    /* For part 2
    * install ABAGAIL
    * copy .java files folder to /opt/mytest
    * ant to build the project
    */
    java -cp ABAGAIL.jar mytest.<Classname> // run the test


