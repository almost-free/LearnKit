//
//  LNKMatrixImages.h
//  LearnKit
//
//  Created by Matt on 3/19/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, LNKImageFormat) {
	LNKImageFormatRGB
};

@interface LNKMatrix (Images)

- (nullable instancetype)initWithImageAtURL:(NSURL *)url format:(LNKImageFormat)format;

@end

NS_ASSUME_NONNULL_END
