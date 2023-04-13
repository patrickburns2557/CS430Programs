===================================
K-Means Clustering Classification
===================================

external libraries used:
scipy
scikit-learn
matplotlib
numpy

Program was created using Python 3.10

Program can be run with python by running the CS430Homework4KMeans.py file

==========================================================================

===================================
LibSVM Classification
===================================

Couldn't get LibSVM to run on Windows, so run on linux instead
downloaded libsvm-331 from https://github.com/cjlin1/libsvm/releases/tag/v331
downloaded the a1a and a1a.t files and put them into the "tools" folder inside libsvm-331
run make command in the folder called "libsvm-331" to compile "svm.cpp", "svm-predict.cpp", "svm-scale.cpp", and "svm-train.cpp" for linux
had to apt-get install gnuplot
needed to change the #! line at the top of easy.py and grid.py to be "#!/usr/bin/env python3" instead of "#!/usr/bin/env python"
Added the line "print("Optimal parameters: c={0}, gamma={1}".format(c, g))" at the end of easy.py right above the line that says "print('Output prediction: {0}'.format(predict_test_file))" to print the optimal values again, right after the printed prediction to easily see both next to one another.
Then went to the tools folder inside the libsvm-331 folder and ran the following command from linux terminal:
	./easy.py a1a a1a.t


Downloaded the diabetes.txt dataset.
To split the diabetes dataset into 80-20, 60-40, 70-30, and 90-10 splits for training and testing data respectively, wrote and ran the file "split.py" inside the tools folder in the libsvm-331 folder. This created all of the files seen in the commands below.

To run each of them, go to the tools folder and run each command in the linux terminal:
	./easy.py diabetes80-20.train diabetes80-20.test
	./easy.py diabetes60-40.train diabetes60-40.test
	./easy.py diabetes70-30.train diabetes70-30.test
	./easy.py diabetes90-10.train diabetes90-10.test