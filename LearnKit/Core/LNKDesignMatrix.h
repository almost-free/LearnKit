//
//  LNKDesignMatrix.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

typedef NS_ENUM(NSUInteger, LNKValueType) {
	LNKValueTypeDouble,
	LNKValueTypeUInt8,
	
	LNKValueTypeNone = NSUIntegerMax
};

@interface LNKDesignMatrix : NSObject <NSCopying>

- (instancetype)init NS_UNAVAILABLE;

/// Initializes a design matrix by loading a CSV file. The file should not contain headings.
/// Optionally, a ones column may be added to the beginning of the matrix. The last column will be mapped to the output vector.
/// This initializer may return `nil`.
- (instancetype)initWithCSVFileAtURL:(NSURL *)url addingOnesColumn:(BOOL)addOnesColumn;

/// Initializes a design matrix by loading a binary matrix of values and a corresponding output vector.
/// Values are parsed in column order. The column count should not include the ones column.
/// If there is no output vector, pass `nil` for the output vector URL and `LNKValueTypeNone` for the output vector value type.
/// This initializer may return `nil`.
- (instancetype)initWithBinaryMatrixAtURL:(NSURL *)matrixURL matrixValueType:(LNKValueType)matrixValueType
						outputVectorAtURL:(NSURL *)outputVectorURL outputVectorValueType:(LNKValueType)outputVectorValueType
							 exampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn;

/// Initializes a design matrix by filling the given buffers.
/// The column count should not include the ones column.
- (instancetype)initWithExampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn prepareBuffers:(BOOL (^)(LNKFloat *matrix, LNKFloat *outputVector))preparationBlock;

@property (nonatomic, readonly) LNKSize exampleCount;
@property (nonatomic, readonly) LNKSize columnCount;

@property (nonatomic, readonly) BOOL hasBiasColumn;

- (const LNKFloat *)matrixBuffer NS_RETURNS_INNER_POINTER;
- (const LNKFloat *)outputVector NS_RETURNS_INNER_POINTER;

- (const LNKFloat *)exampleAtIndex:(LNKSize)index NS_RETURNS_INNER_POINTER;

- (void)clipExampleCountTo:(LNKSize)exampleCount;

@property (nonatomic, readonly, getter=isNormalized) BOOL normalized;

- (void)normalize;
- (void)normalizeWithMeanVector:(const LNKFloat *)meanVector standardDeviationVector:(const LNKFloat *)sdVector;

/// An exception will be thrown if these methods are called prior to normalizing the design matrix.
- (const LNKFloat *)normalizationMeanVector NS_RETURNS_INNER_POINTER;
- (const LNKFloat *)normalizationStandardDeviationVector NS_RETURNS_INNER_POINTER;

/// Normalizes a vector in-place.
/// `normalizeFeatures` must be called prior to normalizing a vector.
- (void)normalizeVector:(LNKFloat *)input;

/// Provides mutable access to the output vector.
- (void)modifyOutputVector:(void(^)(LNKFloat *outputVector, LNKSize m))transformationBlock;

/// For debugging.
- (void)printMatrix;

@end
