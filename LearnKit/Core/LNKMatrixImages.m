//
//  LNKMatrixImages.m
//  LearnKit
//
//  Created by Matt on 3/19/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKMatrixImages.h"

#import "LNKAccelerate.h"
#import <ImageIO/ImageIO.h>

@implementation LNKMatrix (Images)

static const size_t kProcessingBPR = 4;

static CFDataRef __nullable CreateRGBAImageDataWithURL(NSURL *url, size_t *outWidth, size_t *outHeight) CF_RETURNS_RETAINED {
	const CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);

	if (source == NULL) {
		return NULL;
	}

	const CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
	CFRelease(source);

	if (image == NULL) {
		return NULL;
	}

	const size_t w = CGImageGetWidth(image);
	const size_t h = CGImageGetHeight(image);
	const size_t bpp = 8;
	const CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	const CGContextRef context = CGBitmapContextCreate(NULL, w, h, bpp, kProcessingBPR * w, colorSpace, kCGImageAlphaPremultipliedLast);
	CGColorSpaceRelease(colorSpace);
	CGContextDrawImage(context, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h), image);
	CGImageRelease(image);

	const CGImageRef newImage = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	const CGDataProviderRef provider = CGImageGetDataProvider(newImage);
	const CFDataRef data = CGDataProviderCopyData(provider);
	CGImageRelease(newImage);

	if (outWidth) {
		*outWidth = w;
	}
	if (outHeight) {
		*outHeight = h;
	}

	return data;
}

- (nullable instancetype)initWithImageAtURL:(NSURL *)url format:(LNKImageFormat)format {
	size_t w, h;
	const CFDataRef __nullable data = CreateRGBAImageDataWithURL(url, &w, &h);

	if (data == NULL) {
		return nil;
	}

	const size_t pixelCount = h * w;
	const UInt8 *buffer = CFDataGetBytePtr(data);

	LNKSize components;

	switch (format) {
	case LNKImageFormatRGB:
		components = 3;
		break;
	}

	self = [self initWithRowCount:pixelCount columnCount:components prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		for (size_t i = 0; i < pixelCount; i++) {
			switch (format) {
			case LNKImageFormatRGB:
				matrix[i * components + 0] = buffer[i * kProcessingBPR + 0] / 255.0;
				matrix[i * components + 1] = buffer[i * kProcessingBPR + 1] / 255.0;
				matrix[i * components + 2] = buffer[i * kProcessingBPR + 2] / 255.0;
				break;
			}
		}

		return YES;
	}];

	CFRelease(data);

	return self;
}

@end
