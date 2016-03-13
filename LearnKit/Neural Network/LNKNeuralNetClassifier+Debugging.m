//
//  LNKNeuralNetClassifier+Debugging.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKNeuralNetClassifier+Debugging.h"

#if TARGET_OS_MAC
#import <AppKit/AppKit.h>
#else
#import <UIKit/UIKit.h>
#endif

@interface LNKNeuralNetLayer (Debugging)
- (void)drawPreviewInRect:(CGRect)rect context:(CGContextRef)context;
@end

@implementation LNKNeuralNetLayer (Debugging)

static const CGFloat kPreviewVerticalScale = 6;
static const CGFloat kPreviewHalfRange = 6;

- (void)drawPreviewInRect:(CGRect)rect context:(CGContextRef)context {
	NSParameterAssert(context != NULL);

	CGContextBeginPath(context);

	BOOL first = YES;

	for (LNKFloat x = -kPreviewHalfRange; x < kPreviewHalfRange; x++) {
		LNKFloat output = x;
		self.activationFunction(&output, 1);

		const CGFloat screenX = rect.origin.x + x + kPreviewHalfRange;
		const CGFloat screenY = round(output * kPreviewVerticalScale) + CGRectGetMidY(rect);

		if (first) {
			CGContextMoveToPoint(context, screenX, screenY);
			first = NO;
		}
		else {
			CGContextAddLineToPoint(context, screenX, screenY);
		}
	}

	CGContextSetStrokeColorWithColor(context, CGColorGetConstantColor(kCGColorWhite));
	CGContextStrokePath(context);
}

@end

@implementation LNKNeuralNetClassifier (Debugging)

static const CGFloat kLayerWidth = 108;
static const CGFloat kLayerHeight = 82;
static const CGFloat kLayerPadding = 16;
static const CGFloat kLayerMargin = 8;

static const CGFloat kDataBlockSide = 40;
static const CGFloat kDataBlockTextInset = 4;

static const CGFloat kVisualizationHeight = 150;
static const CGFloat kMargin = 8;
static const CGFloat kInputLayerAreaWidth = kLayerMargin * 2 + kDataBlockSide;
static const CGFloat kInnerMargin = 8;

static const CGFloat kFunctionPreviewSide = 12;
static const CGFloat kFunctionPreviewRadius = kFunctionPreviewSide / 2;
static const CGFloat kFunctionPreviewInset = 4;
static const CGFloat kFunctionPreviewVerticalAlignment = 3;

static const CGFloat kArcPadding = 16;
static const CGFloat kArcRadius = kArcPadding / 2;
static const CGFloat kArcRightInflection = 6;
static const CGFloat kArcLeftInflection = kArcPadding - kArcRightInflection;

static const CGFloat kTextBlockHeight = 18;
static const CGFloat kFontSize = 14;
static const CGFloat kWeightTextInset = 3;
static const CGFloat kWeightTextVerticalAlignment = 5;
static const CGFloat kCircleTextInset = 6;
static const CGFloat kPlusTextVerticalAlignment = 4;

static const CGFloat kLayerShapeSide = 20;
static const CGFloat kLayerShapeRadius = kLayerShapeSide / 2;
static const CGFloat kLayerShapeMargin = 10;
static const CGFloat kLayerShapeHalfMargin = kLayerShapeMargin / 2;

static const char *inputText = "Input";
static const char *hiddenText = "Hidden";
static const char *outputText = "Output";
static const char *weightText = "W";
static const char *biasText = "b";
static const char *plusText = "+";

static CGColorRef boxColor() {
	static dispatch_once_t onceToken;
	static CGColorRef boxColor;
	dispatch_once(&onceToken, ^{
		boxColor = CGColorCreateGenericRGB(0.49, 0.54, 0.63, 0.5);
	});
	return boxColor;
}

static CGColorRef lineColor() {
	static dispatch_once_t onceToken;
	static CGColorRef lineColor;
	dispatch_once(&onceToken, ^{
		lineColor = CGColorCreateGenericRGB(0.49, 0.54, 0.63, 1);
	});
	return lineColor;
}

static CGColorRef dataColor() {
	static dispatch_once_t onceToken;
	static CGColorRef dataColor;
	dispatch_once(&onceToken, ^{
		dataColor = CGColorCreateGenericRGB(0.15, 0.20, 0.28, 1);
	});
	return dataColor;
}

static CGColorRef operationColor() {
	static dispatch_once_t onceToken;
	static CGColorRef operationColor;
	dispatch_once(&onceToken, ^{
		operationColor = CGColorCreateGenericRGB(0.99, 0.59, 0.15, 1);
	});
	return operationColor;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

- (void)_drawDataBlockInRect:(CGRect)rect context:(CGContextRef)context withText:(NSString *)text {
	NSParameterAssert(context != NULL);

	CGContextSetFillColorWithColor(context, dataColor());
	CGContextFillRect(context, rect);

	CGContextSetFillColorWithColor(context, CGColorGetConstantColor(kCGColorWhite));
	CGContextShowTextAtPoint(context, rect.origin.x + kDataBlockTextInset, CGRectGetMidY(rect) - kDataBlockTextInset, text.UTF8String, strlen(text.UTF8String));
}

- (void)_drawLayer:(LNKNeuralNetLayer *)layer inContext:(CGContextRef)context position:(CGPoint)position {
	NSParameterAssert(context != NULL);
	NSParameterAssert(layer != nil);

	const CGFloat midY = floor(kVisualizationHeight / 2);
	const CGFloat squareStart = position.x + kLayerPadding;
	const CGFloat squareEnd = squareStart + kLayerShapeSide;
	const CGFloat positionYPad = position.y + kLayerPadding;
	const CGFloat midTop = midY + kLayerShapeHalfMargin + kLayerShapeRadius;
	const CGFloat midBottom = midY - kLayerShapeHalfMargin - kLayerShapeRadius;
	const CGFloat plusStart = squareEnd + kArcPadding;
	const CGFloat plusEnd = plusStart + kLayerShapeSide;
	const CGFloat functionStart = plusEnd + kInnerMargin;
	const CGFloat functionEnd = functionStart + kLayerShapeSide;

	CGContextSetStrokeColorWithColor(context, lineColor());

	CGContextBeginPath(context);
	CGContextMoveToPoint(context, position.x, midY);
	CGContextAddArcToPoint(context, position.x + kArcLeftInflection, midTop, squareStart, midTop, kArcRadius);
	CGContextAddLineToPoint(context, squareStart, midTop);
	CGContextStrokePath(context);

	CGContextBeginPath(context);
	CGContextMoveToPoint(context, squareEnd, midTop);
	CGContextAddArcToPoint(context, squareEnd + kArcRightInflection, midTop, plusStart, midY, kArcRadius);
	CGContextAddLineToPoint(context, plusStart, midY);
	CGContextStrokePath(context);

	CGContextBeginPath(context);
	CGContextMoveToPoint(context, squareEnd, midBottom);
	CGContextAddArcToPoint(context, squareEnd + kArcRightInflection, midBottom, plusStart, midY, kArcRadius);
	CGContextAddLineToPoint(context, plusStart, midY);
	CGContextStrokePath(context);

	CGContextBeginPath(context);
	CGContextMoveToPoint(context, plusEnd, midY);
	CGContextAddLineToPoint(context, functionStart, midY);
	CGContextStrokePath(context);

	CGContextBeginPath(context);
	CGContextMoveToPoint(context, functionEnd, midY);
	CGContextAddLineToPoint(context, functionEnd + kInnerMargin + kLayerMargin, midY);
	CGContextStrokePath(context);

	CGContextSetStrokeColorWithColor(context, boxColor());
	CGContextStrokeRect(context, CGRectMake(position.x, position.y, kLayerWidth, kLayerHeight));

	CGContextSetFillColorWithColor(context, dataColor());
	CGContextFillRect(context, CGRectMake(squareStart, positionYPad, kLayerShapeSide, kLayerShapeSide));
	CGContextFillRect(context, CGRectMake(squareStart, positionYPad + kLayerShapeSide + kLayerShapeMargin, kLayerShapeSide, kLayerShapeSide));

	CGContextSetFillColorWithColor(context, operationColor());
	CGContextFillEllipseInRect(context, CGRectMake(plusStart, midY - kLayerShapeRadius, kLayerShapeSide, kLayerShapeSide));
	CGContextFillEllipseInRect(context, CGRectMake(functionStart, midY - kLayerShapeRadius, kLayerShapeSide, kLayerShapeSide));

	CGContextSetFillColorWithColor(context, CGColorGetConstantColor(kCGColorWhite));
	CGContextShowTextAtPoint(context, squareStart + kWeightTextInset, positionYPad + kWeightTextVerticalAlignment, biasText, strlen(biasText));
	CGContextShowTextAtPoint(context, squareStart + kWeightTextInset, positionYPad + kLayerShapeSide + kLayerShapeMargin + kWeightTextVerticalAlignment, weightText, strlen(weightText));
	CGContextShowTextAtPoint(context, plusStart + kCircleTextInset, midY - kPlusTextVerticalAlignment, plusText, strlen(plusText));

	[layer drawPreviewInRect:CGRectMake(functionStart + kFunctionPreviewInset, midY - kFunctionPreviewRadius - kFunctionPreviewVerticalAlignment, kFunctionPreviewSide, kFunctionPreviewSide) context:context];
}

- (CGImageRef)_createImageVisualization CF_RETURNS_RETAINED {
	const LNKSize hiddenLayerCount = self.hiddenLayerCount;

	const CGFloat width = kInputLayerAreaWidth + kMargin + hiddenLayerCount * (kLayerWidth + kLayerMargin) + kLayerWidth + kMargin + kDataBlockSide + kMargin;
	const CGSize size = CGSizeMake(width, kVisualizationHeight);

	const CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	const CGContextRef context = CGBitmapContextCreate(NULL, (size_t)size.width, (size_t)size.height, 8, 4 * width, colorSpace, kCGImageAlphaPremultipliedLast);
	CGColorSpaceRelease(colorSpace);

	CGContextSetFillColorWithColor(context, CGColorGetConstantColor(kCGColorWhite));
	CGContextFillRect(context, CGRectMake(0, 0, size.width, size.height));

	CGContextSelectFont(context, "Helvetica", kFontSize, kCGEncodingMacRoman);

	const CGFloat labelPosition = kVisualizationHeight - kTextBlockHeight - kMargin;

	CGContextSetFillColorWithColor(context, CGColorGetConstantColor(kCGColorBlack));
	CGContextShowTextAtPoint(context, kMargin, labelPosition, inputText, strlen(inputText));

	const CGFloat midY = floor(kVisualizationHeight / 2);
	const CGRect inputBox = CGRectMake(kMargin, floor((kVisualizationHeight - kDataBlockSide) / 2), kDataBlockSide, kDataBlockSide);
	[self _drawDataBlockInRect:inputBox context:context withText:[NSString stringWithFormat:@"%llu", self.inputLayer.unitCount - 1 /* exclude bias */]];

	CGContextSetStrokeColorWithColor(context, lineColor());
	CGContextBeginPath(context);
	CGContextMoveToPoint(context, CGRectGetMaxX(inputBox), midY);
	CGContextAddLineToPoint(context, kInputLayerAreaWidth + kMargin, midY);
	CGContextStrokePath(context);

	const CGFloat centeredLayerY = floor((kVisualizationHeight - kLayerHeight) / 2);

	for (LNKSize index = 0; index < hiddenLayerCount; index++) {
		const CGFloat x = kMargin + kInputLayerAreaWidth + index * (kLayerWidth + kLayerMargin);
		LNKNeuralNetLayer *const hiddenLayer = [self hiddenLayerAtIndex:index];

		[self _drawLayer:hiddenLayer inContext:context position:CGPointMake(x, centeredLayerY)];

		CGContextSetFillColorWithColor(context, CGColorGetConstantColor(kCGColorBlack));
		CGContextShowTextAtPoint(context, x, labelPosition, hiddenText, strlen(hiddenText));
	}

	LNKNeuralNetLayer *const outputLayer = self.outputLayer;
	const CGFloat outputX = kMargin + kInputLayerAreaWidth + hiddenLayerCount * (kLayerWidth + kLayerMargin);

	[self _drawLayer:outputLayer inContext:context position:CGPointMake(outputX, centeredLayerY)];

	CGContextSetFillColorWithColor(context, CGColorGetConstantColor(kCGColorBlack));
	CGContextShowTextAtPoint(context, outputX, labelPosition, outputText, strlen(outputText));

	const CGRect outputBox = CGRectMake(width - kMargin - kDataBlockSide, floor((kVisualizationHeight - kDataBlockSide) / 2), kDataBlockSide, kDataBlockSide);
	[self _drawDataBlockInRect:outputBox context:context withText:[NSString stringWithFormat:@"%llu", outputLayer.unitCount]];

	const CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	return image;
}

#pragma clang diagnostic pop

- (id)debugQuickLookObject {
	const CGImageRef image = [self _createImageVisualization];

#if TARGET_OS_MAC
	id result = [[NSImage alloc] initWithCGImage:image size:NSMakeSize(CGImageGetWidth(image), CGImageGetHeight(image))];
#else
	id result = [[UIImage alloc] initWithCGImage:image];
#endif

	CGImageRelease(image);

	return [result autorelease];
}

@end
