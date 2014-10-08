//
//  LNKMatrix.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

#import "LNKAccelerate.h"
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
	
	assert(0); // Invalid type
}

- (instancetype)init {
	NSAssertNotReachable(@"Use one of the initWith... initializers", nil);
	return nil;
}

- (instancetype)initWithCSVFileAtURL:(NSURL *)url addingOnesColumn:(BOOL)addOnesColumn {
	NSParameterAssert(url);
	
	if (!(self = [super init]))
		return nil;
	
	NSError *error = nil;
	NSString *stringContents = [[NSString alloc] initWithContentsOfURL:url encoding:NSUTF8StringEncoding error:&error];
	
	if (!stringContents) {
		NSLog(@"Error while loading matrix: could not load the file at the given URL: %@", error);
		return nil;
	}
	
	if (![self _parseMatrixCSVString:stringContents addingOnesColumn:addOnesColumn]) {
		[stringContents release];
		return nil;
	}
	
	[stringContents release];
	
	return self;
}

- (instancetype)initWithBinaryMatrixAtURL:(NSURL *)matrixURL matrixValueType:(LNKValueType)matrixValueType
						outputVectorAtURL:(NSURL *)outputVectorURL outputVectorValueType:(LNKValueType)outputVectorValueType
							 exampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn {
	NSParameterAssert(matrixURL);
	NSParameterAssert(exampleCount);
	NSParameterAssert(columnCount);
	NSParameterAssert(matrixValueType != LNKValueTypeNone);
	
	if (!(self = [super init]))
		return nil;
	
	const LNKSize matrixValueSize = _sizeOfLNKValueType(matrixValueType);
	const LNKSize expectedMatrixSize = exampleCount * columnCount * matrixValueSize;
	
	_exampleCount = exampleCount;
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
	
	for (LNKSize m = 0; m < _exampleCount; m++) {
		for (LNKSize n = 0; n < columnCountWithoutOnes; n++) {
			const char *matrixValue = &matrixValues[(n * _exampleCount + m) * matrixValueSize];
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
		const LNKSize expectedOutputVectorSize = exampleCount * outputVectorValueSize;
		
		NSData *outputVectorData = [NSData dataWithContentsOfURL:outputVectorURL options:0 error:&error];
		
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
		
		for (LNKSize m = 0; m < _exampleCount; m++) {
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

- (instancetype)initWithExampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn prepareBuffers:(BOOL (^)(LNKFloat *, LNKFloat *))preparationBlock {
	NSParameterAssert(exampleCount);
	NSParameterAssert(columnCount);
	NSParameterAssert(preparationBlock);
	
	if (!(self = [super init]))
		return nil;
	
	_exampleCount = exampleCount;
	_columnCount = columnCount + (addOnesColumn ? 1 : 0);
	_hasBiasColumn = addOnesColumn;
	
	[self _allocateBuffers];
	
	if (!preparationBlock(_matrix, _outputVector)) {
		[self _freeBuffers];
		return nil;
	}
	
	return self;
}

// This initializer does not make a copy of the matrix data. Rather, it establishes a weak reference.
- (instancetype)_initWithExampleCount:(LNKSize)exampleCount columnCount:(LNKSize)columnCount addingOnesColumn:(BOOL)addOnesColumn matrix:(LNKFloat *)matrix prepareOutputBuffer:(BOOL (^)(LNKFloat *))preparationBlock {
	NSParameterAssert(exampleCount);
	NSParameterAssert(columnCount);
	NSParameterAssert(preparationBlock);
	
	if (!(self = [super init]))
		return nil;
	
	_exampleCount = exampleCount;
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
	LNKMatrix *matrix = [[LNKMatrix alloc] _initWithExampleCount:_exampleCount columnCount:_columnCount addingOnesColumn:NO matrix:_matrix prepareOutputBuffer:^BOOL(LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, _outputVector, _exampleCount);
		return YES;
	}];
	
	NSAssert(matrix->_exampleCount == _exampleCount, @"Incorrect example count");
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

- (const LNKFloat *)exampleAtIndex:(LNKSize)index {
	NSParameterAssert(index < _exampleCount);
	return _matrix + (index * _columnCount);
}

- (void)clipExampleCountTo:(LNKSize)exampleCount {
	NSParameterAssert(exampleCount);
	_exampleCount = exampleCount;
}

- (BOOL)_parseMatrixCSVString:(NSString *)stringContents addingOnesColumn:(BOOL)addOnesColumn {
	LNKSize fileColumnCount = LNKSizeMax;
	LNKFastArrayRef lines = LNKFastArrayCreate(sizeof(LNKFastArrayRef));
	const char *rawString = stringContents.UTF8String;
	const NSUInteger stringLength = stringContents.length;
	
	__block LNKFastArrayRef currentLine = NULL;
	NSUInteger startIndex = NSNotFound;
	char buffer[NUMBER_BUFFER_SIZE];
	
	void (^cleanupLines)() = ^{
		if (currentLine) {
			LNKFastArrayFree(currentLine);
			currentLine = NULL;
		}
		
		LNKSize m = LNKFastArrayElementCount(lines);
		for (LNKSize i = 0; i < m; i++) {
			LNKFastArrayRef line = *(LNKFastArrayRef *)LNKFastArrayElementAtIndex(lines, i);
			LNKFastArrayFree(line);
		}
	};
	
	for (NSUInteger n = 0; n < stringLength; n++) {
		if (!currentLine) {
			currentLine = LNKFastArrayCreate(sizeof(LNKFloat));
			startIndex = n;
		}
		
		char c = rawString[n];
		
		if (startIndex != NSNotFound && (c == ',' || c == '\n' || c == '\r')) {
			LNKSize length = n - startIndex;
			memcpy(buffer, rawString + startIndex, length);
			buffer[length] = '\0';
			
			if (length) {
				LNKFloat value = LNK_strtoflt(buffer, length);
				LNKFastArrayAddElement(currentLine, &value);
			}
			
			startIndex = NSNotFound;
			
			if (c == '\n' || c == '\r') {
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
				
				LNKFastArrayAddElement(lines, &currentLine);
				currentLine = NULL;
			}
		}
		else if (startIndex == NSNotFound)
			startIndex = n;
	}
	
	if (!LNKFastArrayElementCount(lines)) {
		NSLog(@"Error while loading matrix: the matrix does not contain any examples");
		cleanupLines();
		return NO;
	}
	
	if (fileColumnCount == LNKSizeMax) {
		NSLog(@"Error while loading matrix: the matrix does not contain any columns");
		cleanupLines();
		return NO;
	}
	
	_exampleCount = LNKFastArrayElementCount(lines);
	
	// The matrix's column count does not include the output vector, but should include the optional ones column.
	_columnCount = fileColumnCount - 1 + (addOnesColumn ? 1 : 0);
	_hasBiasColumn = addOnesColumn;
	
	[self _allocateBuffers];
	
	for (LNKSize m = 0; m < _exampleCount; m++) {
		// The last column contains the output vector.
		LNKFastArrayRef line = *(LNKFastArrayRef *)LNKFastArrayElementAtIndex(lines, m);
		_outputVector[m] = *(LNKFloat *)LNKFastArrayElementAtIndex(line, _columnCount - (addOnesColumn ? 1 : 0));
		
		// Ignore the last column since it's actually our output vector.
		for (LNKSize n = 0; n < _columnCount - (addOnesColumn ? 1 : 0); n++) {
			_matrix[OFFSET_ROW(m) + (addOnesColumn ? 1 : 0) + n] = *(LNKFloat *)LNKFastArrayElementAtIndex(line, n);
		}
	}
	
	cleanupLines();
	return YES;
}

- (void)_allocateBuffersIncludingMatrix:(BOOL)allocateMatrix {
	if (allocateMatrix)
		_matrix = LNKFloatAlloc(_columnCount * _exampleCount);
	
	_outputVector = LNKFloatAlloc(_exampleCount);
	_columnToMu = LNKFloatAlloc(_columnCount);
	_columnToSD = LNKFloatAlloc(_columnCount);
	
	if (_hasBiasColumn) {
		_columnToSD[0] = 1;
		
		if (allocateMatrix) {
			const LNKFloat one = 1;
			LNK_vfill(&one, _matrix, _columnCount, _exampleCount);
		}
	}
}

- (void)_allocateBuffers {
	[self _allocateBuffersIncludingMatrix:YES];
}

- (void)_fillBuffersWithValue:(LNKFloat)fillValue {
	LNK_vfill(&fillValue, _matrix, UNIT_STRIDE, _exampleCount * _columnCount);
	LNK_vfill(&fillValue, _outputVector, UNIT_STRIDE, _exampleCount);
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

- (void)normalize {
	if (_normalized)
		return;
	
	LNKFloat *workgroup = LNKFloatAlloc(_exampleCount);
	
	for (LNKSize n = _hasBiasColumn; n < _columnCount; n++) {
		LNKFloat *columnPointer = _matrix + n;
		
		LNKFloat mean;
		LNK_vmean(columnPointer, _columnCount, &mean, _exampleCount);
		
		LNKFloat minusMean = -mean;
		LNK_vsadd(columnPointer, _columnCount, &minusMean, workgroup, UNIT_STRIDE, _exampleCount);
		
		LNKFloat sd;
		LNK_dotpr(workgroup, UNIT_STRIDE, workgroup, UNIT_STRIDE, &sd, _exampleCount);
		sd = LNK_sqrt(1.0 / (_exampleCount - 1) * sd);
		
		_columnToSD[n] = sd;
		_columnToMu[n] = minusMean;
	}
	
	[self normalizeWithMeanVector:_columnToMu standardDeviationVector:_columnToSD];
	
	free(workgroup);
}

- (void)normalizeWithMeanVector:(const LNKFloat *)meanVector standardDeviationVector:(const LNKFloat *)sdVector {
	NSParameterAssert(meanVector);
	NSParameterAssert(sdVector);
	
	if (meanVector != _columnToMu)
		LNKFloatCopy(_columnToMu, meanVector, _columnCount);
	
	if (sdVector != _columnToSD)
		LNKFloatCopy(_columnToSD, sdVector, _columnCount);
	
	for (LNKSize n = _hasBiasColumn; n < _columnCount; n++) {
		LNKFloat *columnPointer = _matrix + n;
		
		const LNKFloat minusMean = meanVector[n];
		const LNKFloat sd = sdVector[n];
		
		// (column - mean) / standardDeviation
		LNK_vsadd(columnPointer, _columnCount, &minusMean, columnPointer, _columnCount, _exampleCount);
		LNK_vsdiv(columnPointer, _columnCount, &sd, columnPointer, _columnCount, _exampleCount);
	}
	
	_normalized = YES;
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
	NSParameterAssert(input);
	NSAssert(_normalized, @"The matrix needs to be normalized first");
	
	// (input - mean) / standardDeviation
	// Since mu has been negated initially, we add instead of subtracting.
	LNK_vadd(input, UNIT_STRIDE, _columnToMu, UNIT_STRIDE, input, UNIT_STRIDE, _columnCount);
	LNK_vdiv(_columnToSD, UNIT_STRIDE, input, UNIT_STRIDE, input, UNIT_STRIDE, _columnCount);
}

- (void)modifyOutputVector:(void(^)(LNKFloat *outputVector, LNKSize m))transformationBlock {
	transformationBlock(_outputVector, _exampleCount);
}

- (void)printMatrix {
	LNKPrintMatrix("Matrix", _matrix, _exampleCount, _columnCount);
}

@end
