//: ## Finding Primary Colors of an Image with K-Means
//: [Visit](http://www.mattrajca.com/2016/03/20/finding-primary-colors-in-images-with-learnkit.html) the companion article.

import Cocoa
import LearnKit

//: We start by loading the image we're going to analyze. The goal is to programatically extract the two primary colors of the image – a dark shade of blue for the sky and tan for the building.

let url = NSBundle.mainBundle().URLForResource("Palace", withExtension: "jpg")!

//: ![Schonbrunn Palace](Palace.jpg)

//: All data in LearnKit gets imported through matrices represented with `LNKMatrix`. We load the image into a matrix where rows represent pixels and columns represent the R, G, and B channels.

guard let matrix = LNKMatrix(imageAtURL: url, format: .RGB) else {
	fatalError("The image cannot be loaded.")
}

//: Now we construct a K-Means classifier that will find the two primary clusters, which will correspond to the two primary colors when working with RGB data.

let classifier = LNKKMeansClassifier(matrix: matrix, implementationType:.Accelerate, optimizationAlgorithm: nil, classes: LNKClasses.withCount(2))

//: Passing in `LNKSizeMax` for the iteration count causes the algorithm to run until convergence. A fixed number of iterations can be passed in here to improve performance, though that's not a problem in this example.

classifier.iterationCount = LNKSizeMax

//: Now we run the algorithm, iteratively finding the key colors.

classifier.train()

//: Once training is complete, we obtain the two primary colors, represented as 3-component vectors containing R, G, and B values.

let color1 = classifier.centroidForClusterAtIndex(0)
let color2 = classifier.centroidForClusterAtIndex(1)

//: To visualize the colors, we convert them into `NSColor` objects.

func NSColorFromRGBVector(vector: LNKVector) -> NSColor {
	return NSColor(SRGBRed: CGFloat(vector.data[0]), green: CGFloat(vector.data[1]), blue: CGFloat(vector.data[2]), alpha: 1)
}

let nscolor1 = NSColorFromRGBVector(color1)
let nscolor2 = NSColorFromRGBVector(color2)

//: Sure enough, these are the two primary colors of our image!
