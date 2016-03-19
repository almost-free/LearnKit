//
//  LNKMatrix.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

NS_ASSUME_NONNULL_BEGIN

@class LNKCSVColumnRule;

typedef NS_ENUM(NSUInteger, LNKValueType) {
	LNKValueTypeDouble,
	LNKValueTypeUInt8,
	
	LNKValueTypeNone = NSUIntegerMax
};

@interface LNKMatrix : NSObject <NSCopying>

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initIdentityWithColumnCount:(LNKSize)columnCount;

/// Initializes a matrix by loading a CSV file. The file should not contain headings.
/// Optionally, a ones column may be added to the beginning of the matrix. The last column will be mapped to the output vector.
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url addingOnesColumn:(BOOL)addOnesColumn;
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter addingOnesColumn:(BOOL)addOnesColumn;

/// Columns may be deleted or transformed (e.g. mapping strings representing categorical data to numerical entries) by passing a dictionary of preprocessing rules, indexed by the column index.
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter addingOnesColumn:(BOOL)addOnesColumn columnPreprocessingRules:(NSDictionary<NSNumber *, LNKCSVColumnRule *> *)preprocessingRules;

/// Initializes a matrix by loading a binary matrix of values and a corresponding output vector.
/// Values are parsed in column order. The column count should not include the ones column.
/// If there is no output vector, pass `nil` for the output vector URL and `LNKValueTypeNone` for the output vector value type.
- (nullable instancetype)initWithBinaryMatrixAtURL:(NSURL *)matrixURL matrixValueType:(LNKValueType)matrixValueType
								 outputVectorAtURL:(nullable NSURL *)outputVectorURL outputVectorValueType:(LNKValueType)outputVectorValueType
										  rowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn;

/// Initializes a matrix by filling the given buffers.
/// The column count should not include the ones column.
- (instancetype)initWithRowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount
				addingOnesColumn:(BOOL)addOnesColumn
				  prepareBuffers:(BOOL (^)(LNKFloat *matrix, LNKFloat *outputVector))preparationBlock;

@property (nonatomic, readonly) LNKSize rowCount;
@property (nonatomic, readonly) LNKSize columnCount;

@property (nonatomic, readonly) BOOL hasBiasColumn;

- (const LNKFloat *)matrixBuffer NS_RETURNS_INNER_POINTER;
- (const LNKFloat *)outputVector NS_RETURNS_INNER_POINTER;

- (const LNKFloat *)rowAtIndex:(LNKSize)index NS_RETURNS_INNER_POINTER;

/// The result must be freed by the caller.
- (LNKVector)copyOfColumnAtIndex:(LNKSize)columnIndex;

- (void)clipRowCountTo:(LNKSize)rowCount;

/// Throws an exception if the passed-in matrix is `nil` or if matrix dimensions are incompatible.
- (LNKMatrix *)multiplyByMatrix:(LNKMatrix *)matrix;
- (LNKMatrix *)transposedMatrix;

- (LNKMatrix *)covarianceMatrix;
- (nullable LNKMatrix *)invertedMatrix;

/// Returns a copy of the current matrix with its rows reshuffled.
- (LNKMatrix *)copyShuffledMatrix;

/// Returns a copy of the current matrix with `rowCount` of its rows reshuffled.
- (LNKMatrix *)copyShuffledSubmatrixWithRowCount:(LNKSize)rowCount;

- (void)splitIntoTrainingMatrix:(LNKMatrix *__nonnull *__nonnull)trainingMatrix testMatrix:(LNKMatrix *__nonnull *__nonnull)testMatrix trainingBias:(LNKFloat)trainingBias;

- (LNKMatrix *)submatrixWithRowRange:(NSRange)range;
- (LNKMatrix *)submatrixWithRowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount;

@property (nonatomic, readonly, getter=isNormalized) BOOL normalized;

/// If the matrix has already been normalized, `self` is returned.
- (LNKMatrix *)normalizedMatrix;
- (LNKMatrix *)normalizedMatrixWithMeanVector:(const LNKFloat *)meanVector standardDeviationVector:(const LNKFloat *)sdVector;

/// An exception will be thrown if these methods are called prior to normalizing the matrix.
- (const LNKFloat *)normalizationMeanVector NS_RETURNS_INNER_POINTER;
- (const LNKFloat *)normalizationStandardDeviationVector NS_RETURNS_INNER_POINTER;

/// Normalizes a vector in-place by subtracting the mean and dividing by the standard deviation of the matrix.
/// This method throws an exception if the matrix has not been normalized.
- (void)normalizeVector:(LNKFloat *)vector;

/// Provides mutable access to the output vector.
- (void)modifyOutputVector:(void(^)(LNKFloat *outputVector, LNKSize m))transformationBlock;

/// For debugging.
- (void)printMatrix;

@end

NS_ASSUME_NONNULL_END
