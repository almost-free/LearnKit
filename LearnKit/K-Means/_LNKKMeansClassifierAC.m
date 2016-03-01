//
//  _LNKKMeansClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKKMeansClassifierAC.h"

#import "LNKAccelerate.h"
#import "LNKKMeansClassifierPrivate.h"
#import "LNKMatrix.h"
#import "LNKMatrixPrivate.h"

@implementation _LNKKMeansClassifierAC {
	BOOL _isUsingCustomCentroids;
}

#define OFFSET_BY_CLUSTER(value) (value) + cluster * columnCount

- (void)_setRandomClusters:(LNKFloat *)clusterCentroids {
	NSParameterAssert(clusterCentroids);
	
	LNKMatrix *matrix = self.matrix;
	const LNKSize clusterCount = self.classes.count;
	const LNKSize columnCount = matrix.columnCount;
	const LNKSize rowCount = matrix.rowCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	
	NSMutableIndexSet *usedIndices = [[NSMutableIndexSet alloc] init];
	
	for (LNKSize cluster = 0; cluster < clusterCount; cluster++) {
		while (YES) {
			const LNKSize selectedExample = arc4random_uniform((uint32_t)rowCount);
			
			if (![usedIndices containsIndex:selectedExample]) {
				LNKFloatCopy(OFFSET_BY_CLUSTER(clusterCentroids), _ROW_IN_MATRIX_BUFFER(selectedExample), columnCount);
				[usedIndices addIndex:selectedExample];
				break;
			}
		}
	}
	
	[usedIndices release];
}

- (LNKSize)_closestClusterToExample:(const LNKFloat *)example {
	NSParameterAssert(example);
	
	const LNKSize columnCount = self.matrix.columnCount;
	const LNKSize clusterCount = self.classes.count;
	const LNKFloat *clusterCentroids = [self _clusterCentroids];
	
	LNKSize closestCluster = LNKSizeMax;
	LNKFloat closestDistance = LNKFloatMax;
	
	for (LNKSize cluster = 0; cluster < clusterCount; cluster++) {
		LNKFloat distance;
		LNKVectorDistance(OFFSET_BY_CLUSTER(clusterCentroids), example, &distance, columnCount);
		
		if (distance < closestDistance) {
			closestDistance = distance;
			closestCluster = cluster;
		}
	}
	
	NSAssert(closestCluster != LNKSizeMax, @"This should never occur as long as there is at least one cluster");
	
	return closestCluster;
}

- (void)train {
	LNKMatrix *matrix = self.matrix;
	const LNKSize clusterCount = self.classes.count;
	const LNKSize columnCount = matrix.columnCount;
	const LNKSize rowCount = matrix.rowCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKSize iterationCount = self.iterationCount;
	LNKFloat *clusterCentroids = [self _clusterCentroids];
	
	LNKFloat *clusterCentroidsWorkspace = LNKFloatAlloc(clusterCount * columnCount);
	LNKFloat *clusterCounts = LNKFloatAlloc(clusterCount);
	LNKSize *examplesToClusters = malloc(rowCount * sizeof(LNKSize));
	
	if (!_isUsingCustomCentroids)
		[self _setRandomClusters:clusterCentroids];
	
	for (LNKSize iteration = 0; iteration < iterationCount; iteration++) {
		LNK_vclr(clusterCounts, UNIT_STRIDE, clusterCount);
		LNK_vclr(clusterCentroidsWorkspace, UNIT_STRIDE, clusterCount * columnCount);
		
		// Assign examples to clusters.
		for (LNKSize index = 0; index < rowCount; index++) {
			const LNKFloat *example = _ROW_IN_MATRIX_BUFFER(index);
			const LNKSize cluster = [self _closestClusterToExample:example];
			examplesToClusters[index] = cluster;
			
			LNKFloat *workspaceEntry = OFFSET_BY_CLUSTER(clusterCentroidsWorkspace);
			LNK_vadd(example, UNIT_STRIDE, workspaceEntry, UNIT_STRIDE, workspaceEntry, UNIT_STRIDE, columnCount);
			clusterCounts[cluster]++;
		}
		
		// Update cluster centroids.
		for (LNKSize cluster = 0; cluster < clusterCount; cluster++) {
			LNK_vsdiv(OFFSET_BY_CLUSTER(clusterCentroidsWorkspace), UNIT_STRIDE, &clusterCounts[cluster], OFFSET_BY_CLUSTER(clusterCentroids), UNIT_STRIDE, columnCount);
		}
	}
	
	free(clusterCentroidsWorkspace);
	free(clusterCounts);
	free(examplesToClusters);
}

- (void)_setClusterCentroids:(const LNKFloat *)clusterCentroids {
	_isUsingCustomCentroids = YES;
	[super _setClusterCentroids:clusterCentroids];
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	if (!featureVector.data)
		[NSException raise:NSGenericException format:@"The feature vector must contain data"];
		
	if (featureVector.length != self.matrix.columnCount)
		[NSException raise:NSGenericException format:@"The length of the feature vector must match the number of columns in the matrix"];
	
	return [NSNumber numberWithLNKSize:[self _closestClusterToExample:featureVector.data]];
}

@end
