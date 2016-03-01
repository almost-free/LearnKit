//
//  LNKMatrixPCA.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKPCAInformation : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/// `centers` and `scales` hold the means and standard deviations of the features prior to normalization (respectively).
@property (nonatomic, readonly) LNKVector centers;
@property (nonatomic, readonly) LNKVector scales;

/// The rotation matrix obtained from the eigenvectors of the covariance matrix.
@property (nonatomic, retain, readonly) LNKMatrix *rotationMatrix;

/// The standard deviations of the principal components (square roots of the eigenvalues of the covariance matrix).
@property (nonatomic, readonly) LNKVector standardDeviations;

/// The original matrix after being scaled, normalized, and rotated.
@property (nonatomic, retain, readonly) LNKMatrix *rotatedMatrix;

@end


@interface LNKMatrix (PCA)

/// Returns information about the principal components of the dataset using SVD.
/// The matrix will automatically be normalized if it hasn't already been normalized.
/// This method will return `nil` when SVD fails.
- (nullable LNKPCAInformation *)analyzePrincipalComponents;

/// This method reduces (normalizes and rotates) the data to its `dimension` principal components.
/// The dimension must be at least 1 and less than the number of columns in the matrix.
/// The matrix will automatically be normalized if it hasn't already been normalized.
/// This method will return `nil` when SVD fails.
- (nullable LNKMatrix *)matrixReducedToDimension:(LNKSize)dimension;

/// This method reduces (normalizes and rotates) the data to its `dimension` principal components and
/// projects it back to the original coordinate space.
/// The dimension must be at least 1 and less than the number of columns in the matrix.
/// The matrix will automatically be normalized if it hasn't already been normalized.
/// This method will return `nil` when SVD fails.
- (nullable LNKMatrix *)matrixProjectedToDimension:(LNKSize)dimension;

@end

NS_ASSUME_NONNULL_END
