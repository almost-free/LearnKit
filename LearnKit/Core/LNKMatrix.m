//
//  LNKMatrix.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

#import "LNKAccelerate.h"
#import "LNKCSVColumnRule.h"
#import "LNKFastArray.h"
#import "LNKUtilities.h"

@implementation LNKMatrix {
	LNKFloat *_matrix, *_outputVector;
	LNKFloat *_columnToMu, *_columnToSD;
	BOOL _weakMatrixReference;
}

#define NUMBER_BUFFER_SIZE 2048
#define OFFSET_ROW(exampleIndex) ((exampleIndex) * _columnCount)

static LNKSize _sizeOfLNKValueType(LNKValueType type) {
	if (type == LNKValueTypeDouble)
		return 8;
	else if (type == LNKValueTypeUInt8)
		return 1;
	else if (type == LNKValueTypeNone)
		return 0;
	
	NSCAssert(0, @"Invalid type - not reached");
	return 0;
}

- (instancetype)init {
	NSAssertNotReachable(@"Use one of the initWith... initializers", nil);
	return nil;
}

- (instancetype)initIdentityWithColumnCount:(LNKSize)columnCount {
	return [self initWithRowCount:columnCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		for (LNKSize i = 0; i < columnCount; i++) {
			matrix[i * columnCount + i] = 1;
		}
		return YES;
	}];
}

- (instancetype)initWithCSVFileAtURL:(NSURL *)url {
	return [self initWithCSVFileAtURL:url delimiter:','];
}

- (instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter {
	return [self initWithCSVFileAtURL:url delimiter:delimiter ignoringHeader:NO columnPreprocessingRules:@{}];
}

- (instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter ignoringHeader:(BOOL)ignoreHeader columnPreprocessingRules:(NSDictionary<NSNumber *, LNKCSVColumnRule *> *)preprocessingRules {
	if (preprocessingRules == nil) {
		[NSException raise:NSInvalidArgumentException format:@"The dictionary of preprocessing rules must not be nil"];
	}

	NSParameterAssert(url);
	
	if (!(self = [super init]))
		return nil;
	
	NSError *error = nil;
	NSString *stringContents = [[NSString alloc] initWithContentsOfURL:url encoding:NSUTF8StringEncoding error:&error];
	
	if (!stringContents) {
		NSLog(@"Error while loading matrix: could not load the file at the given URL: %@", error);
		return nil;
	}
	
	if (![self _parseMatrixCSVString:stringContents delimiter:delimiter ignoringHeader:ignoreHeader columnPreprocessingRules:preprocessingRules]) {
		[stringContents release];
		return nil;
	}
	
	[stringContents release];
	
	return self;
}

- (instancetype)initWithBinaryMatrixAtURL:(NSURL *)matrixURL matrixValueType:(LNKValueType)matrixValueType
						outputVectorAtURL:(NSURL *)outputVectorURL outputVectorValueType:(LNKValueType)outputVectorValueType
							 rowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn {
	NSParameterAssert(matrixURL);
	NSParameterAssert(rowCount);
	NSParameterAssert(columnCount);
	NSParameterAssert(matrixValueType != LNKValueTypeNone);
	
	if (!(self = [super init]))
		return nil;
	
	const LNKSize matrixValueSize = _sizeOfLNKValueType(matrixValueType);
	const LNKSize expectedMatrixSize = rowCount * columnCount * matrixValueSize;
	
	_rowCount = rowCount;
	_columnCount = columnCount + (addOnesColumn ? 1 : 0);
	_hasBiasColumn = addOnesColumn;
	
	const LNKSize columnCountWithoutOnes = columnCount;
	
	NSError *error = nil;
	NSData *matrixData = [NSData dataWithContentsOfURL:matrixURL options:0 error:&error];
	
	if (!matrixData) {
		NSLog(@"Error while loading matrix: could not load the matrix file at the given URL: %@", error);
		return nil;
	}
	
	const char *matrixValues = [matrixData bytes];
	
	[self _allocateBuffers];
	
	for (LNKSize m = 0; m < _rowCount; m++) {
		for (LNKSize n = 0; n < columnCountWithoutOnes; n++) {
			const char *matrixValue = &matrixValues[(n * _rowCount + m) * matrixValueSize];
			const LNKSize index = m * _columnCount + n + (addOnesColumn ? 1 : 0);
			
			switch (matrixValueType) {
				case LNKValueTypeDouble:
					_matrix[index] = *(LNKFloat *)matrixValue;
					break;
				case LNKValueTypeUInt8:
					_matrix[index] = *(uint8_t *)matrixValue;
					break;
				default:
					break;
			}
		}
	}
	
	if (outputVectorURL && outputVectorValueType != LNKValueTypeNone) {
		const LNKSize outputVectorValueSize = _sizeOfLNKValueType(outputVectorValueType);
		const LNKSize expectedOutputVectorSize = rowCount * outputVectorValueSize;
		
		NSData *outputVectorData = [NSData dataWithContentsOfURL:(NSURL *__nonnull)outputVectorURL options:0 error:&error];
		
		if (!outputVectorData) {
			NSLog(@"Error while loading matrix: could not load the output vector file at the given URL: %@", error);
			return nil;
		}
		
		if (matrixData.length != expectedMatrixSize) {
			NSLog(@"Error while loading matrix: invalid matrix file size");
			return nil;
		}
		
		if (outputVectorData.length != expectedOutputVectorSize) {
			NSLog(@"Error while loading matrix: invalid output vector file size");
			return nil;
		}
		
		const char *outputVectorValues = [outputVectorData bytes];
		
		for (LNKSize m = 0; m < _rowCount; m++) {
			const char *outputVectorValue = &outputVectorValues[m * outputVectorValueSize];
			
			switch (outputVectorValueType) {
				case LNKValueTypeDouble:
					_outputVector[m] = *(LNKFloat *)outputVectorValue;
					break;
				case LNKValueTypeUInt8:
					_outputVector[m] = *(uint8_t *)outputVectorValue;
					break;
				default:
					break;
			}
		}
	}
	
	return self;
}

- (instancetype)initWithRowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn prepareBuffers:(BOOL (^)(LNKFloat *, LNKFloat *))preparationBlock {
	NSParameterAssert(rowCount);
	NSParameterAssert(columnCount);
	NSParameterAssert(preparationBlock);
	
	if (!(self = [super init]))
		return nil;
	
	_rowCount = rowCount;
	_columnCount = columnCount + (addOnesColumn ? 1 : 0);
	_hasBiasColumn = addOnesColumn;
	
	[self _allocateBuffers];
	
	if (!preparationBlock(_matrix, _outputVector)) {
		[self _freeBuffers];
		[self release];
		return nil;
	}
	
	return self;
}

// This initializer does not make a copy of the matrix data. Rather, it establishes a weak reference.
- (instancetype)_initWithRowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn matrix:(LNKFloat *)matrix prepareOutputBuffer:(BOOL (^)(LNKFloat *))preparationBlock {
	NSParameterAssert(rowCount);
	NSParameterAssert(columnCount);
	NSParameterAssert(preparationBlock);
	
	if (!(self = [super init]))
		return nil;
	
	_rowCount = rowCount;
	_columnCount = columnCount + (addOnesColumn ? 1 : 0);
	_hasBiasColumn = addOnesColumn;
	
	[self _allocateBuffersIncludingMatrix:NO];
	_matrix = matrix;
	_weakMatrixReference = YES;
	
	if (!preparationBlock(_outputVector)) {
		[self _freeBuffers];
		return nil;
	}
	
	return self;
}

- (id)copyWithZone:(NSZone *)zone {
#pragma unused(zone)
	
	// Even if we already have a ones column, we set addingOnesColumn to NO because we don't want one to be added again.
	LNKMatrix *matrix = [[LNKMatrix alloc] _initWithRowCount:_rowCount columnCount:_columnCount addingOnesColumn:NO matrix:_matrix prepareOutputBuffer:^BOOL(LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, _outputVector, _rowCount);
		return YES;
	}];
	
	NSAssert(matrix->_rowCount == _rowCount, @"Incorrect example count");
	NSAssert(matrix->_columnCount == _columnCount, @"Incorrect column count");
	
	matrix->_normalized = _normalized;
	matrix->_hasBiasColumn = _hasBiasColumn;
	LNKFloatCopy(matrix->_columnToMu, _columnToMu, _columnCount);
	LNKFloatCopy(matrix->_columnToSD, _columnToSD, _columnCount);
	
	return matrix;
}

- (const LNKFloat *)matrixBuffer {
	return _matrix;
}

- (const LNKFloat *)outputVector {
	return _outputVector;
}

- (const LNKFloat *)rowAtIndex:(LNKSize)index {
	NSParameterAssert(index < _rowCount);
	return _matrix + (index * _columnCount);
}

- (void)clipRowCountTo:(LNKSize)rowCount {
	NSParameterAssert(rowCount);
	_rowCount = rowCount;
}

- (LNKVector)copyOfColumnAtIndex:(LNKSize)columnIndex {
	LNKFloat *values = LNKFloatAlloc(_rowCount);

	for (LNKSize index = 0; index < _rowCount; index++) {
		values[index] = _matrix[index * _columnCount + columnIndex];
	}

	return LNKVectorMakeUnsafe(values, _rowCount);
}

- (LNKMatrix *)multiplyByMatrix:(LNKMatrix *)matrix {
	if (matrix == nil) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The passed in matrix must not be nil" userInfo:nil];
	}

	const LNKSize columnCount = self.columnCount;

	if (columnCount != matrix.rowCount) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The matrix sizes do not match" userInfo:nil];
	}

	const LNKSize rowCount = self.rowCount;
	const LNKSize matrixColumnCount = matrix.columnCount;

	LNKFloat *const result = LNKFloatAlloc(rowCount * matrixColumnCount);
	LNK_mmul(_matrix, UNIT_STRIDE, matrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, rowCount, matrixColumnCount, columnCount);

	return [[[LNKMatrix alloc] initWithRowCount:rowCount columnCount:matrixColumnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *localMatrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		LNKFloatCopy(localMatrix, result, rowCount * matrixColumnCount);
		return YES;
	}] autorelease];
}

- (LNKMatrix *)transposedMatrix {
	const LNKSize columnCount = self.columnCount;
	const LNKSize rowCount = self.rowCount;

	return [[[LNKMatrix alloc] initWithRowCount:columnCount columnCount:rowCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		LNK_mtrans(_matrix, matrix, columnCount, rowCount);
		return YES;
	}] autorelease];
}

- (LNKMatrix *)covarianceMatrix {
	if (!self.normalized) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"Only covariance of normalized matrices is supported" userInfo:nil];
	}

	return [[[LNKMatrix alloc] initWithRowCount:_columnCount columnCount:_columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)

		// S = 1/m X' X
		LNKFloat *const transposeMatrix = LNKFloatAlloc(_columnCount * _rowCount);
		LNK_mtrans(_matrix, transposeMatrix, _columnCount, _rowCount);

		LNK_mmul(transposeMatrix, UNIT_STRIDE, _matrix, UNIT_STRIDE, matrix, UNIT_STRIDE, _columnCount, _columnCount, _rowCount);
		free(transposeMatrix);

		const LNKFloat m = (LNKFloat)_rowCount;
		LNK_vsdiv(matrix, UNIT_STRIDE, &m, matrix, UNIT_STRIDE, _columnCount * _columnCount);

		return YES;
	}] autorelease];
}

- (LNKMatrix *)invertedMatrix {
	if (_columnCount != _rowCount) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"Only square matrices can be inverted" userInfo:nil];
		return nil;
	}

	return [[[LNKMatrix alloc] initWithRowCount:_columnCount columnCount:_columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)

		LNKFloatCopy(matrix, _matrix, _columnCount * _columnCount);

		__CLPK_integer error = 0;
		__CLPK_integer *const pivot = malloc(_columnCount * sizeof(__CLPK_integer));
		__CLPK_doublereal *const workspace = malloc(_columnCount * sizeof(__CLPK_doublereal));

		// Factorization
		__CLPK_integer N = (__CLPK_integer)_columnCount;
		dgetrf_(&N, &N, matrix, &N, pivot, &error);

		if (error != 0) {
			free(pivot);
			free(workspace);
			return NO;
		}

		// Inversion
		dgetri_(&N, matrix, &N, pivot, workspace, &N, &error);

		free(pivot);
		free(workspace);

		return (error == 0);
	}] autorelease];
}

- (BOOL)isEqual:(id)object {
	if (![object isKindOfClass:[LNKMatrix class]]) {
		return NO;
	}

	LNKMatrix *const otherMatrix = object;

	if (_rowCount != otherMatrix.rowCount || _columnCount != otherMatrix.columnCount) {
		return NO;
	}

	const LNKFloat *otherBuffer = otherMatrix.matrixBuffer;
	const LNKSize items = _rowCount * _columnCount;
	const LNKFloat threshold = 0.0001;

	for (LNKSize i = 0; i < items; i++) {
		if (fabs(_matrix[i] - otherBuffer[i]) > threshold) {
			return NO;
		}
	}

	return YES;
}

- (LNKMatrix *)matrixByAddingBiasColumn {
	const LNKSize rowCount = self.rowCount;
	const LNKSize columnCount = self.columnCount;
	const LNKSize biasOffset = 1;
	const LNKSize columnCountWithBias = columnCount + biasOffset;
	const LNKFloat *const matrixBuffer = self.matrixBuffer;

	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithRowCount:rowCount columnCount:columnCount addingOnesColumn:YES prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		for (LNKSize m = 0; m < rowCount; m++) {
			LNKFloatCopy(matrix + m * columnCountWithBias + biasOffset, matrixBuffer + m * columnCount, columnCount);
		}
		LNKFloatCopy(outputVector, self.outputVector, rowCount);
		return YES;
	}];
	matrix->_normalized = _normalized;
	LNKFloatCopy(matrix->_columnToMu + biasOffset, _columnToMu, columnCount);
	LNKFloatCopy(matrix->_columnToSD + biasOffset, _columnToSD, columnCount);
	return [matrix autorelease];
}

// The result must be freed by the caller.
- (LNKSize *)_shuffleIndices {
	LNKSize *indices = malloc(sizeof(LNKSize) * _rowCount);

	for (LNKSize index = 0; index < _rowCount; index++) {
		indices[index] = index;
	}

	// Shuffle the indices.
	for (LNKSize index = 0; index < _rowCount; index++) {
		const LNKSize leftElements = _rowCount - index;
		const LNKSize otherIndex = arc4random_uniform((uint32_t)leftElements) + index;

		const LNKSize temp = indices[otherIndex];
		indices[otherIndex] = indices[index];
		indices[index] = temp;
	}

	return indices;
}

- (LNKMatrix *)copyShuffledMatrix {
	return [self copyShuffledSubmatrixWithRowCount:_rowCount];
}

- (LNKMatrix *)copyShuffledSubmatrixWithRowCount:(LNKSize)rowCount {
	if (rowCount > _rowCount)
		[NSException raise:NSInvalidArgumentException format:@"The number of examples in the submatrix cannot be greater than the number of examples in the current matrix"];
	
	return [[LNKMatrix alloc] initWithRowCount:rowCount columnCount:_columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKSize *const indices = [self _shuffleIndices];
		
		for (LNKSize index = 0; index < rowCount; index++) {
			const LNKSize actualIndex = indices[index];
			
			outputVector[index] = _outputVector[actualIndex];
			LNKFloatCopy(matrix + index * _columnCount, _matrix + actualIndex * _columnCount, _columnCount);
		}
		
		free(indices);
		
		return YES;
	}];
}

- (void)splitIntoTrainingMatrix:(LNKMatrix **)trainingMatrix testMatrix:(LNKMatrix **)testMatrix trainingBias:(LNKFloat)trainingBias {
	const LNKSize rowCount = self.rowCount;
	const LNKSize trainingSize = rowCount * trainingBias;
	const LNKSize testSize = rowCount - trainingSize;

	LNKMatrix *const shuffledMatrix = [self copyShuffledMatrix];
	*trainingMatrix = [shuffledMatrix submatrixWithRowRange:NSMakeRange(0, trainingSize)];
	*testMatrix = [shuffledMatrix submatrixWithRowRange:NSMakeRange(trainingSize, testSize)];

	[shuffledMatrix release];
}

- (LNKMatrix *)submatrixWithRowRange:(NSRange)range {
	const LNKSize columnCount = self.columnCount;

	LNKMatrix *const submatrix = [[LNKMatrix alloc] initWithRowCount:range.length columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		for (LNKSize example = range.location; example < NSMaxRange(range); example++) {
			const LNKFloat *inputExample = [self rowAtIndex:example];
			LNKFloatCopy(matrix + (example - range.location) * columnCount, inputExample, columnCount);

			outputVector[example - range.location] = _outputVector[example];
		}

		return YES;
	}];

	return [submatrix autorelease];
}

- (LNKMatrix *)submatrixWithRowCount:(LNKSize)rowCount columnCount:(LNKSize)columnCount {
	if (rowCount >= _rowCount)
		[NSException raise:NSInvalidArgumentException format:@"The number of examples in the submatrix must be less than the number of examples in the current matrix"];
	
	if (columnCount >= _columnCount)
		[NSException raise:NSInvalidArgumentException format:@"The number of columns in the submatrix must be less than the number of columns in the current matrix"];
	
	LNKMatrix *submatrix = [[LNKMatrix alloc] initWithRowCount:rowCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		for (LNKSize row = 0; row < rowCount; row++) {
			const LNKFloat *inputExample = [self rowAtIndex:row];
			LNKFloatCopy(matrix + row * columnCount, inputExample, columnCount);
		}
		
		return YES;
	}];
	
	return [submatrix autorelease];
}

- (LNKSize)_outputColumnIndexForColumnPreprocessingRules:(NSDictionary<NSNumber *, LNKCSVColumnRule *> *)rules {
	__block LNKSize outputColumnIndex = LNKSizeMax;

	[rules enumerateKeysAndObjectsUsingBlock:^(NSNumber *key, LNKCSVColumnRule *obj, BOOL *stop) {
		if (obj.type == LNKCSVColumnRuleTypeOutput) {
			outputColumnIndex = key.LNKSizeValue;
			*stop = YES;
		}
	}];

	return outputColumnIndex;
}

- (BOOL)_parseMatrixCSVString:(NSString *)stringContents delimiter:(unichar)delimiter ignoringHeader:(BOOL)ignoreHeader columnPreprocessingRules:(NSDictionary<NSNumber *, LNKCSVColumnRule *> *)preprocessingRules {
	LNKSize fileColumnCount = LNKSizeMax;
	LNKFastArrayRef lines = LNKFastArrayCreate();
	const char *rawString = stringContents.UTF8String;
	const NSUInteger stringLength = stringContents.length;
	
	__block LNKFastArrayRef currentLine = NULL;
	NSUInteger startIndex = NSNotFound;
	
	void (^cleanupLines)() = ^{
		if (currentLine != NULL) {
			for (LNKSize i = 0; i < LNKFastArrayElementCount(currentLine); i++) {
				char *buffer = LNKFastArrayElementAtIndex(currentLine, i);
				free(buffer);
			}

			LNKFastArrayFree(currentLine);
		}

		const LNKSize m = LNKFastArrayElementCount(lines);
		for (LNKSize i = 0; i < m; i++) {
			LNKFastArrayRef line = LNKFastArrayElementAtIndex(lines, i);
			NSAssert(line != currentLine, @"Anticipating a double-free");

			for (LNKSize j = 0; j < LNKFastArrayElementCount(line); j++) {
				char *buffer = LNKFastArrayElementAtIndex(line, j);
				free(buffer);
			}

			LNKFastArrayFree(line);
		}

		currentLine = NULL;
		LNKFastArrayFree(lines);
	};
	
	for (NSUInteger n = 0; n < stringLength; n++) {
		if (!currentLine) {
			currentLine = LNKFastArrayCreate();
			startIndex = n;
		}
		
		const char c = rawString[n];
		
		if (startIndex != NSNotFound && (c == delimiter || c == '\n' || c == '\r')) {
			LNKSize length = n - startIndex;

			if (length > 0) {
				char *buffer = calloc(length + 1 /* NULL terminator */, sizeof(char));
				memcpy(buffer, rawString + startIndex, length);
				LNKFastArrayAddElement(currentLine, buffer);
			}
			
			startIndex = NSNotFound;
			
			if (c == '\n' || c == '\r') {
				// Skips empty lines.
				if (LNKFastArrayElementCount(currentLine) > 0) {
					if (fileColumnCount == LNKSizeMax) {
						fileColumnCount = LNKFastArrayElementCount(currentLine);
						
						if (fileColumnCount < 2) {
							NSLog(@"Error while loading matrix: the matrix must have at least two columns");
							cleanupLines();
							return NO;
						}
					}
					else if (fileColumnCount != LNKFastArrayElementCount(currentLine)) {
						NSLog(@"Error while loading matrix: lines have varying numbers of columns");
						cleanupLines();
						return NO;
					}

					LNKFastArrayAddElement(lines, currentLine);
				} else {
					LNKFastArrayFree(currentLine);
				}

				currentLine = NULL;
			}
		}
		else if (startIndex == NSNotFound) {
			startIndex = n;
		}
	}
	
	if (LNKFastArrayElementCount(lines) == 0) {
		NSLog(@"Error while loading matrix: the matrix does not contain any examples");
		cleanupLines();
		return NO;
	}
	
	if (fileColumnCount == LNKSizeMax) {
		NSLog(@"Error while loading matrix: the matrix does not contain any columns");
		cleanupLines();
		return NO;
	}

	LNKFastArrayRef firstLine = LNKFastArrayElementAtIndex(lines, 0);
	const LNKSize firstLineColumns = LNKFastArrayElementCount(firstLine);
	__block LNKSize deletedColumns = 0;

	[preprocessingRules enumerateKeysAndObjectsUsingBlock:^(NSNumber *column, LNKCSVColumnRule *rule, BOOL *stop) {
#pragma unused(stop)
		if (rule.type == LNKCSVColumnRuleTypeDelete && column.LNKSizeValue < firstLineColumns) {
			deletedColumns += 1;
		}
	}];

	const LNKSize lineCount = LNKFastArrayElementCount(lines);
	const LNKSize startRow = (ignoreHeader ? 1 : 0);
	
	_rowCount = ignoreHeader ? lineCount - 1 : lineCount;
	
	// The matrix's column count does not include the output vector, but should include the optional ones column.
	_columnCount = fileColumnCount - 1 - deletedColumns;
	_hasBiasColumn = NO;
	
	[self _allocateBuffers];

	LNKSize localRow = 0;
	
	for (LNKSize m = startRow; m < lineCount; m++) {
		// The last column contains the output vector.
		LNKFastArrayRef line = LNKFastArrayElementAtIndex(lines, m);
		const LNKSize ruleOutputColumnIndex = [self _outputColumnIndexForColumnPreprocessingRules:preprocessingRules];
		const LNKSize outputColumnIndex = ruleOutputColumnIndex == LNKSizeMax ? fileColumnCount - 1 : ruleOutputColumnIndex;
		char *outputString = LNKFastArrayElementAtIndex(line, outputColumnIndex);

		LNKCSVColumnRule *outputRule = preprocessingRules[@(outputColumnIndex)];
		if (outputRule && outputRule.type == LNKCSVColumnRuleTypeConversion) {
			LNKCSVColumnRuleTypeConversionHandler handler = outputRule.object;
			NSAssert(handler != nil, @"Conversions should always have a handler");

			_outputVector[localRow] = handler([NSString stringWithUTF8String:outputString]);
		} else if (outputRule && outputRule.type == LNKCSVColumnRuleTypeDelete) {
			NSLog(@"Error while loading the matrix: the output column cannot be deleted");
			cleanupLines();
			return NO;
		} else {
			_outputVector[localRow] = LNK_strtoflt(outputString);
		}

		LNKSize localColumn = 0;

		// Ignore the last column since it's actually our output vector.
		for (LNKSize n = 0; n < fileColumnCount; n++) {
			if (n == outputColumnIndex) {
				// Skip the output column, but continue because it's not necessarily the last one!
				continue;
			}

			char *columnString = LNKFastArrayElementAtIndex(line, n);

			LNKCSVColumnRule *outputRule = preprocessingRules[@(n)];
			if (outputRule && outputRule.type == LNKCSVColumnRuleTypeConversion) {
				LNKCSVColumnRuleTypeConversionHandler handler = outputRule.object;
				NSAssert(handler != nil, @"Conversions should always have a handler");

				_matrix[OFFSET_ROW(localRow) + localColumn] = handler([NSString stringWithUTF8String:columnString]);
				localColumn += 1;
			} else if (outputRule && outputRule.type == LNKCSVColumnRuleTypeDelete) {
				// Skips to the next column without storing anything.
			} else {
				_matrix[OFFSET_ROW(localRow) + localColumn] = LNK_strtoflt(columnString);
				localColumn += 1;
			}
		}

		localRow++;
	}

	cleanupLines();
	return YES;
}

- (void)_allocateBuffersIncludingMatrix:(BOOL)allocateMatrix {
	if (allocateMatrix)
		_matrix = LNKFloatAlloc(_columnCount * _rowCount);
	
	_outputVector = LNKFloatAlloc(_rowCount);
	_columnToMu = LNKFloatAlloc(_columnCount);
	_columnToSD = LNKFloatAlloc(_columnCount);
	
	if (_hasBiasColumn) {
		_columnToSD[0] = 1;
		_columnToMu[0] = 0;
		
		if (allocateMatrix) {
			const LNKFloat one = 1;
			LNK_vfill(&one, _matrix, _columnCount, _rowCount);
		}
	}
}

- (void)_allocateBuffers {
	[self _allocateBuffersIncludingMatrix:YES];
}

- (void)_fillBuffersWithValue:(LNKFloat)fillValue {
	LNK_vfill(&fillValue, _matrix, UNIT_STRIDE, _rowCount * _columnCount);
	LNK_vfill(&fillValue, _outputVector, UNIT_STRIDE, _rowCount);
}

- (void)_freeBuffers {
	if (!_weakMatrixReference)
		free(_matrix);
	
	free(_outputVector);
	free(_columnToMu);
	free(_columnToSD);
}

- (void)dealloc {
	[self _freeBuffers];
	[super dealloc];
}

- (LNKMatrix *)normalizedMatrix {
	if (_normalized) {
		return self;
	}
	
	LNKFloat *const workgroup = LNKFloatAlloc(_rowCount);
	LNKFloat *const columnToMu = LNKFloatAlloc(_columnCount);
	LNKFloat *const columnToSD = LNKFloatAlloc(_columnCount);

	if (_hasBiasColumn) {
		columnToSD[0] = 1;
		columnToMu[0] = 0;
	}
	
	for (LNKSize n = _hasBiasColumn; n < _columnCount; n++) {
		LNKFloat *const columnPointer = _matrix + n;
		
		LNKFloat mean;
		LNK_vmean(columnPointer, _columnCount, &mean, _rowCount);
		
		columnToMu[n] = mean;
		columnToSD[n] = LNK_vsd(LNKVectorMakeUnsafe(columnPointer, _rowCount), _columnCount, workgroup, mean, YES);
	}

	free(workgroup);

	LNKMatrix *const matrix = [self normalizedMatrixWithMeanVector:columnToMu standardDeviationVector:columnToSD];
	free(columnToMu);
	free(columnToSD);
	
	return matrix;
}

- (LNKMatrix *)normalizedMatrixWithMeanVector:(const LNKFloat *)meanVector standardDeviationVector:(const LNKFloat *)sdVector {
	if (_normalized) {
		return self;
	}

	NSParameterAssert(meanVector);
	NSParameterAssert(sdVector);

	LNKMatrix *const normalizedMatrix = [[LNKMatrix alloc] initWithRowCount:_rowCount columnCount:_columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(matrix, _matrix, _columnCount * _rowCount);
		LNKFloatCopy(outputVector, _outputVector, _rowCount);
		return YES;
	}];

	LNKFloatCopy(normalizedMatrix->_columnToMu, meanVector, _columnCount);
	LNKFloatCopy(normalizedMatrix->_columnToSD, sdVector, _columnCount);
	
	for (LNKSize n = _hasBiasColumn; n < _columnCount; n++) {
		LNKFloat *const columnPointer = normalizedMatrix->_matrix + n;
		
		const LNKFloat minusMean = -meanVector[n];
		const LNKFloat sd = sdVector[n];
		
		// (column - mean) / standardDeviation
		LNK_vsadd(columnPointer, _columnCount, &minusMean, columnPointer, _columnCount, _rowCount);
		LNK_vsdiv(columnPointer, _columnCount, &sd, columnPointer, _columnCount, _rowCount);
	}
	
	normalizedMatrix->_normalized = YES;

	return [normalizedMatrix autorelease];
}

- (void)_ensureNormalization {
	if (!_normalized)
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"The matrix must be normalized prior." userInfo:nil];
}

- (const LNKFloat *)normalizationMeanVector {
	[self _ensureNormalization];
	return _columnToMu;
}

- (const LNKFloat *)normalizationStandardDeviationVector {
	[self _ensureNormalization];
	return _columnToSD;
}

- (void)normalizeVector:(LNKFloat *)input {
	if (!_normalized) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The matrix must be normalized prior to calling -normalizeVetor:" userInfo:nil];
	}

	NSParameterAssert(input);
	
	// (input - mean) / standardDeviation
	LNK_vsub(_columnToMu, UNIT_STRIDE, input, UNIT_STRIDE, input, UNIT_STRIDE, _columnCount);
	LNK_vdiv(_columnToSD, UNIT_STRIDE, input, UNIT_STRIDE, input, UNIT_STRIDE, _columnCount);
}

- (void)modifyOutputVector:(void(^)(LNKFloat *outputVector, LNKSize m))transformationBlock {
	transformationBlock(_outputVector, _rowCount);
}

- (void)printMatrix {
	LNKPrintMatrix("Matrix", _matrix, _rowCount, _columnCount);
}

- (void)printOutputVector {
	LNKPrintVector("Output Vector", _outputVector, _rowCount);
}

@end
