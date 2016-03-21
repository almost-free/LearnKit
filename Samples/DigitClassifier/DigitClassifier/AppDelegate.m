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

static const LNKSize exampleCount = 5000;
static const LNKSize columnCount = 400;
static const LNKSize paddedColumnCount = columnCount + 1;
static const LNKFloat tolerance = 0.4;

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
											  rowCount:exampleCount
											   columnCount:columnCount
										  addingOnesColumn:YES];
	
	LNKFloat *matrixBuffer = (LNKFloat *)_matrix.matrixBuffer;
	
	// Pre-process entries to 1-bit.
	for (LNKSize n = 0; n < exampleCount * paddedColumnCount; n++) {
		matrixBuffer[n] = matrixBuffer[n] < tolerance ? 0 : 1;
	}
	
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	algorithm.iterationCount = 400;

	NSArray<LNKNeuralNetLayer *> *hiddenLayers = @[ [[LNKNeuralNetSigmoidLayer alloc] initWithUnitCount:25] ];
	LNKNeuralNetLayer *outputLayer = [[LNKNeuralNetSigmoidLayer alloc] initWithClasses:[LNKClasses withRange:NSMakeRange(1, 10)]];
	_classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:_matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm hiddenLayers:hiddenLayers outputLayer:outputLayer];
	
	[_classifier train];
	
	[self.canvasView loadFeatureVector:[_matrix rowAtIndex:2000]];
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
	[self _trainClassifier];
}

- (void)canvasViewRecognizeFeatureVector:(const LNKFloat *)vector {
	LNKFloat *paddedVector = LNKFloatAlloc(paddedColumnCount);
	LNKFloatCopy(paddedVector+1, vector, columnCount);
	paddedVector[0] = 1;
	
	LNKClass *class = [_classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(paddedVector, paddedColumnCount)];
	self.labelField.stringValue = [NSString stringWithFormat:@"%ld", class.unsignedIntegerValue];
}

@end
