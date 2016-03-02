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

/// `centers` and `scales` hold the means and standard deviations (respectively) of the features prior to normalization.
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

/// Returns information about the principal components of the dataset using the NIPALS
/// (Non-linear Iterative PArtial Least Squares) algorithm.
///
/// For really large datasets, using this method is faster than performing full SVD.
///
/// Only the specified number of principal components is returned. Pass in `LNKSizeMax` to return
/// information about all principal components.
///
/// `LNKFloatMax` can be passed in for the tolerance to run the algorithm for the maximum number of
/// iterations. Similarily, `LNKSizeMax` can be passed in for the maximum number of iterations to run the
/// algorithm until convergence. An exception will be thrown if both parameters are passed in the max value.
///
/// The matrix will automatically be normalized if it hasn't already been normalized.
/// The resulting vector of singular values is empty when computing approximate principal components.
- (LNKPCAInformation *)analyzeApproximatePrincipalComponents:(LNKSize)principalComponents toTolerance:(LNKFloat)tolerance maximumIterations:(LNKSize)maximumIterations;

/// This convenience method uses a tolerance of 1e-9 and a maximum of 500 iterations.
- (LNKPCAInformation *)analyzeApproximatePrincipalComponents:(LNKSize)principalComponents;

/// This method reduces (normalizes and rotates) the data to its `dimension` principal components.
/// The dimension must be at least 1 and less than the number of columns in the matrix.
/// The matrix will automatically be normalized if it hasn't already been normalized.
/// This method will return `nil` when SVD fails.
- (nullable LNKMatrix *)matrixReducedToDimension:(LNKSize)dimension;

/// This method projects a reduced matrix back to its original coordinate space.
/// The dimension must be at least 1 and less than the number of columns in the matrix.
- (LNKMatrix *)matrixProjectedToDimension:(LNKSize)dimension withPCAInformation:(LNKPCAInformation *)pca;

@end

NS_ASSUME_NONNULL_END
