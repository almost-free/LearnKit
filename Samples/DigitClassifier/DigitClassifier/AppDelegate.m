//
//  AppDelegate.m
//  DigitClassifier
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "AppDelegate.h"

#import "CanvasView.h"

#import <LearnKit/LearnKit.h>

@interface AppDelegate () <CanvasViewDelegate>

@property (nonatomic, strong) IBOutlet NSWindow *window;
@property (nonatomic, weak) IBOutlet CanvasView *canvasView;
@property (nonatomic, weak) IBOutlet NSTextField *labelField;

@end


@implementation AppDelegate {
	LNKNeuralNetClassifier *_classifier;
	LNKMatrix *_matrix;
}

- (void)awakeFromNib {
	self.canvasView.delegate = self;
}

- (void)_trainClassifier {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *matrixPath = [bundle pathForResource:@"Examples" ofType:@"dat"];
	NSString *outputVectorPath = [bundle pathForResource:@"Labels" ofType:@"dat"];
	
	_matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:matrixPath]
										   matrixValueType:LNKValueTypeDouble
										 outputVectorAtURL:[NSURL fileURLWithPath:outputVectorPath]
									 outputVectorValueType:LNKValueTypeUInt8
											  exampleCount:5000 columnCount:400 addingOnesColumn:YES];
	
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	algorithm.iterationCount = 400;
	
	_classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:_matrix
											  implementationType:LNKImplementationTypeAccelerate
										   optimizationAlgorithm:algorithm
														 classes:[LNKClasses withRange:NSMakeRange(1, 10)]];
	_classifier.hiddenLayerCount = 1;
	_classifier.hiddenLayerUnitCount = 25;
	
	[_classifier train];
	
	[self.canvasView loadFeatureVector:[_matrix exampleAtIndex:2000]];
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
	[self _trainClassifier];
}

- (void)canvasViewRecognizeFeatureVector:(const LNKFloat *)vector {
	LNKFloat *paddedVector = LNKFloatAlloc(401);
	LNKFloatCopy(paddedVector+1, vector, 400);
	paddedVector[0] = 1;
	
	LNKClass *class = [_classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(paddedVector, 401)];
	self.labelField.stringValue = [NSString stringWithFormat:@"%ld", class.unsignedIntegerValue];
}

@end
