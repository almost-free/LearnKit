LearnKit
========

LearnKit is a Cocoa framework for Machine Learning. It currently runs on top of the Accelerate framework on iOS and OS X.

Supported Algorithms
--------------------

- k-Means
- k-Nearest Neighbors
- Linear Regression
- Logistic Regression
- Naive Bayes
- Neural Networks
- Principal Component Analysis

Example
-------

In this example, we have a matrix that contains 5000 20x20 digits. Each 20x20 digit has been flattened into a row of 400 pixel intensities. We load it as such:

	NSURL *matrixURL = [NSURL fileURLWithPath:matrixPath];
	NSURL *matrixOutputURL = [NSURL fileURLWithPath:outputVectorPath];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:matrixURL
													 matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:matrixOutputURL
											   outputVectorValueType:LNKValueTypeUInt8
														exampleCount:5000
														 columnCount:400
													addingOnesColumn:YES];

Next, we set up a conjugate gradient optimization algorithm to train the neural network.

	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	algorithm.iterationCount = 400;

Now we initialize a neural network classifier with our matrix and optimization algorithm. We also indicate the possible outputs are digits ranging from 1 to 10.

	LNKNeuralNetClassifier *classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:matrix 
																	 implementationType:LNKImplementationTypeAccelerate
																  optimizationAlgorithm:algorithm
																				classes:[LNKClasses withRange:NSMakeRange(1, 10)]];
	classifier.hiddenLayerCount = 1;
	classifier.hiddenLayerUnitCount = 25;

The neural network parameters above were picked with performance in mind. They can be fine-tuned to increase the accuracy of the classifier. Finally, we train the neural network classifer and predict the class of a previously-unseen digit.

	[classifier train];
	
	LNKClass *someDigit = [classifier predictValueForFeatureVector:someImage length:someImageLength];

With the right parameters, classification accuracy rates of over 99% can be attained.

Future Tasks
------------

- Support collaborative filtering
- Support decision trees
- Support SVMs
- Port to Metal and OpenCL

License
-------

LearnKit is available under the MIT license.

Credit
------

LearnKit uses:

- `fmincg` by Carl Edward Rasmussen
- `liblbfgs`
- Data prepared by Andrew Ng for [Machine Learning](https://www.coursera.org/course/ml)
