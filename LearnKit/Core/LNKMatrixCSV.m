//
//  LNKMatrixCSV.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKMatrixCSV.h"

#import "LNKAccelerate.h"
#import "LNKCSVColumnRule.h"
#import "LNKFastObjects.h"

@implementation LNKMatrix (CSV)

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

	if (!(self = [super init])) {
		return nil;
	}

	NSError *error = nil;
	NSString *stringContents = [[NSString alloc] initWithContentsOfURL:url encoding:NSUTF8StringEncoding error:&error];

	if (!stringContents) {
		NSLog(@"Error while loading matrix: could not load the file at the given URL: %@", error);
		return nil;
	}

	LNKFastArrayRef lines = [self _parseRawLinesAndElementsForString:stringContents delimiter:delimiter];
	[stringContents release];

	if (LNKFastArrayObjectCount(lines) == 0) {
		NSLog(@"Error while loading matrix: the matrix does not contain any examples");
		return nil;
	}

	LNKFastArrayRef firstLine = (LNKFastArrayRef)LNKFastArrayObjectAtIndex(lines, 0);
	const LNKSize firstLineColumns = LNKFastArrayObjectCount(firstLine);
	__block LNKSize deletedColumns = 0;

	[preprocessingRules enumerateKeysAndObjectsUsingBlock:^(NSNumber *column, LNKCSVColumnRule *rule, BOOL *stop) {
#pragma unused(stop)
		if (rule.type == LNKCSVColumnRuleTypeDelete && column.LNKSizeValue < firstLineColumns) {
			deletedColumns += 1;
		}
	}];

	const LNKSize lineCount = LNKFastArrayObjectCount(lines);
	const LNKSize startRow = (ignoreHeader ? 1 : 0);

	const LNKSize rowCount = ignoreHeader ? lineCount - 1 : lineCount;

	// The matrix's column count does not include the output vector, but should include the optional ones column.
	const LNKSize columnCount = firstLineColumns - 1 - deletedColumns;

	self = [self initWithRowCount:rowCount columnCount:columnCount prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKSize localRow = 0;

		for (LNKSize m = startRow; m < lineCount; m++) {
			// The last column contains the output vector.
			LNKFastArrayRef line = (LNKFastArrayRef)LNKFastArrayObjectAtIndex(lines, m);
			const LNKSize ruleOutputColumnIndex = [self _outputColumnIndexForColumnPreprocessingRules:preprocessingRules];
			const LNKSize outputColumnIndex = ruleOutputColumnIndex == LNKSizeMax ? firstLineColumns - 1 : ruleOutputColumnIndex;
			LNKFastStringRef outputString = (LNKFastStringRef)LNKFastArrayObjectAtIndex(line, outputColumnIndex);

			LNKCSVColumnRule *outputRule = preprocessingRules[@(outputColumnIndex)];
			if (outputRule && outputRule.type == LNKCSVColumnRuleTypeConversion) {
				LNKCSVColumnRuleTypeConversionHandler handler = outputRule.object;
				NSAssert(handler != nil, @"Conversions should always have a handler");

				outputVector[localRow] = handler([NSString stringWithUTF8String:LNKFastStringGetUTF8String(outputString)]);
			} else if (outputRule && outputRule.type == LNKCSVColumnRuleTypeDelete) {
				NSLog(@"Error while loading the matrix: the output column cannot be deleted");
				return NO;
			} else {
				outputVector[localRow] = LNK_strtoflt(LNKFastStringGetUTF8String(outputString));
			}

			LNKSize localColumn = 0;

			// Ignore the last column since it's actually our output vector.
			for (LNKSize n = 0; n < firstLineColumns; n++) {
				if (n == outputColumnIndex) {
					// Skip the output column, but continue because it's not necessarily the last one!
					continue;
				}

				LNKFastStringRef columnString = (LNKFastStringRef)LNKFastArrayObjectAtIndex(line, n);

				LNKCSVColumnRule *outputRule = preprocessingRules[@(n)];
				if (outputRule && outputRule.type == LNKCSVColumnRuleTypeConversion) {
					LNKCSVColumnRuleTypeConversionHandler handler = outputRule.object;
					NSAssert(handler != nil, @"Conversions should always have a handler");

					matrix[localRow * columnCount + localColumn] = handler([NSString stringWithUTF8String:LNKFastStringGetUTF8String(columnString)]);
					localColumn += 1;
				} else if (outputRule && outputRule.type == LNKCSVColumnRuleTypeDelete) {
					// Skips to the next column without storing anything.
				} else {
					matrix[localRow * columnCount + localColumn] = LNK_strtoflt(LNKFastStringGetUTF8String(columnString));
					localColumn += 1;
				}
			}
			
			localRow++;
		}

		return YES;
	}];

	LNKFastArrayRelease(lines);

	return self;
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

// The result must be memory-managed by the caller.
- (LNKFastArrayRef)_parseLine:(NSString *)line delimiter:(unichar)delimiter {
	LNKFastArrayRef currentLine = LNKFastArrayCreate();
	const NSUInteger stringLength = line.length;
	const char *const rawString = line.UTF8String;

	NSUInteger startIndex = NSNotFound;
	BOOL inQuotes = NO;

	for (NSUInteger n = 0; n < stringLength+1 /* one past to indicate EOL */; n++) {
		if (startIndex == NSNotFound) {
			startIndex = n;
		}

		if (n == stringLength || (rawString[n] == delimiter && !inQuotes)) {
			LNKSize length = n - startIndex;

			if (length > 0) {
				LNKFastStringRef string = LNKFastStringCreateWithUTF8String(rawString + startIndex, length);
				LNKFastArrayAddFastObject(currentLine, (LNKFastObjectRef)string);
				LNKFastObjectRelease((LNKFastObjectRef)string);
			}

			startIndex = NSNotFound;
		} else if (rawString[n] == '"') {
			if (n == 0 || rawString[n-1] != '\\') {
				inQuotes = !inQuotes;
			}
		}
	}

	if (LNKFastArrayObjectCount(currentLine) == 0) {
		LNKFastArrayRelease(currentLine);
		return NULL;
	}

	return currentLine;
}

// The result must be memory-managed by the caller.
- (LNKFastArrayRef)_parseRawLinesAndElementsForString:(NSString *)string delimiter:(unichar)delimiter {
	__block LNKSize fileColumnCount = LNKSizeMax;
	LNKFastArrayRef lines = LNKFastArrayCreate();

	__block BOOL failed = NO;

	[string enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
		LNKFastArrayRef currentLine = [self _parseLine:line delimiter:delimiter];
		if (currentLine == NULL) {
			return;
		}

		if (fileColumnCount == LNKSizeMax) {
			fileColumnCount = LNKFastArrayObjectCount(currentLine);

			if (fileColumnCount < 2) {
				NSLog(@"Error while loading matrix: the matrix must have at least two columns");
				failed = YES;
				*stop = YES;
				return;
			}
		}
		else if (fileColumnCount != LNKFastArrayObjectCount(currentLine)) {
			NSLog(@"Error while loading matrix: lines have varying numbers of columns");
			failed = YES;
			*stop = YES;
			return;
		}

		LNKFastArrayAddFastObject(lines, (LNKFastObjectRef)currentLine);
		LNKFastArrayRelease(currentLine);
	}];

	if (failed) {
		LNKFastArrayRelease(lines);
		return NULL;
	}

	if (fileColumnCount == LNKSizeMax) {
		NSLog(@"Error while loading matrix: the matrix does not contain any columns");
		LNKFastArrayRelease(lines);
		return NULL;
	}

	return lines;
}

@end
