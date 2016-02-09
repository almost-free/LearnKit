//
//  LNKMatrix.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, LNKValueType) {
	LNKValueTypeDouble,
	LNKValueTypeUInt8,
	
	LNKValueTypeNone = NSUIntegerMax
};

@interface LNKMatrix : NSObject <NSCopying>

- (instancetype)init NS_UNAVAILABLE;

/// Initializes a matrix by loading a CSV file. The file should not contain headings.
/// Optionally, a ones column may be added to the beginning of the matrix. The last column will be mapped to the output vector.
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url addingOnesColumn:(BOOL)addOnesColumn;
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter addingOnesColumn:(BOOL)addOnesColumn;

/// Initializes a matrix by loading a binary matrix of values and a corresponding output vector.
/// Values are parsed in column order. The column count should not include the ones column.
/// If there is no output vector, pass `nil` for the output vector URL and `LNKValueTypeNone` for the output vector value type.
- (nullable instancetype)initWithBinaryMatrixAtURL:(NSURL *)matrixURL matrixValueType:(LNKValueType)matrixValueType
								 outputVectorAtURL:(nullable NSURL *)outputVectorURL outputVectorValueType:(LNKValueType)outputVectorValueType
									  exampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn;

/// Initializes a matrix by filling the given buffers.
/// The column count should not include the ones column.
- (instancetype)initWithExampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount
					addingOnesColumn:(BOOL)addOnesColumn
					  prepareBuffers:(BOOL (^)(LNKFloat *matrix, LNKFloat *outputVector))preparationBlock;

@property (nonatomic, readonly) LNKSize exampleCount;
@property (nonatomic, readonly) LNKSize columnCount;

@property (nonatomic, readonly) BOOL hasBiasColumn;

- (const LNKFloat *)matrixBuffer NS_RETURNS_INNER_POINTER;
- (const LNKFloat *)outputVector NS_RETURNS_INNER_POINTER;

- (const LNKFloat *)exampleAtIndex:(LNKSize)index NS_RETURNS_INNER_POINTER;

/// The result must be freed by the caller.
- (LNKVector)copyOfColumnAtIndex:(LNKSize)columnIndex;

- (void)clipExampleCountTo:(LNKSize)exampleCount;

/// Returns a copy of the current matrix with its rows reshuffled.
- (LNKMatrix *)copyShuffledMatrix;

/// Returns a copy of the current matrix with `exampleCount` of its rows reshuffled.
- (LNKMatrix *)copyShuffledSubmatrixWithExampleCount:(LNKSize)exampleCount;

- (void)splitIntoTrainingMatrix:(LNKMatrix *__nonnull *__nonnull)trainingMatrix testMatrix:(LNKMatrix *__nonnull *__nonnull)testMatrix trainingBias:(LNKFloat)trainingBias;

- (LNKMatrix *)submatrixWithExampleRange:(NSRange)range;
- (LNKMatrix *)submatrixWithExampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount;

@property (nonatomic, readonly, getter=isNormalized) BOOL normalized;

- (void)normalize;
- (void)normalizeWithMeanVector:(const LNKFloat *)meanVector standardDeviationVector:(const LNKFloat *)sdVector;

/// An exception will be thrown if these methods are called prior to normalizing the matrix.
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

NS_ASSUME_NONNULL_END
