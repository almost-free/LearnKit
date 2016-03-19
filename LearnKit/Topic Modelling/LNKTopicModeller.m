//
//  LNKTopicModeller.m
//  LearnKit
//
//  Created by Matt on 3/19/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKTopicModeller.h"

#import "LNKAccelerate.h"
#import "LNKMatrix.h"

@implementation LNKTopicSet {
	NSMutableArray<NSMutableArray<NSString *> *> *_topics;
}

- (instancetype)initWithTopics:(LNKSize)topicCount {
	if (!(self = [super init])) {
		return nil;
	}

	_topics = [[NSMutableArray alloc] init];

	for (LNKSize i = 0; i < topicCount; i++) {
		NSMutableArray<NSString *> *array = [[NSMutableArray alloc] init];
		[_topics addObject:array];
		[array release];
	}

	return self;
}

- (void)addWord:(NSString *)word forTopicAtIndex:(NSUInteger)index {
	NSMutableArray<NSString *> *const topics = _topics[index];
	[topics addObject:word];
}

- (void)dealloc {
	[_topics release];
	[super dealloc];
}

@end

@implementation LNKTopicModeller {
	/* weak */ id <LNKTopicModellerDelegate> _delegate;
	LNKMatrix *_documentMatrix;
	NSArray<NSString *> *_vocabulary;
	LNKSize _topicCount;
}

- (instancetype)initWithDocumentMatrix:(LNKMatrix *)documentMatrix vocabulary:(NSArray<NSString *> *)vocabulary topicCount:(LNKSize)topicCount delegate:(id<LNKTopicModellerDelegate>)delegate {
	NSParameterAssert(documentMatrix);
	NSParameterAssert(vocabulary);
	NSParameterAssert(delegate);

	if (topicCount < 1) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"At least one topic must be specified" userInfo:nil];
	}

	if (!(self = [super init])) {
		return nil;
	}

	_delegate = delegate;
	_topicCount = topicCount;
	_vocabulary = [vocabulary retain];
	_documentMatrix = [documentMatrix retain];

	return self;
}

- (void)dealloc {
	_delegate = nil;
	[_vocabulary release];
	[_documentMatrix release];
	[super dealloc];
}

typedef struct {
	LNKSize index;
	LNKFloat frequency;
} Word;

- (void)main {
	const LNKSize wordCount = _documentMatrix.columnCount;
	const LNKSize documentCount = _documentMatrix.rowCount;
	const LNKFloat *const matrixBuffer = _documentMatrix.matrixBuffer;

	// Stores the topic weights.
	LNKFloat *const pi = LNKFloatAlloc(_topicCount);

	// Initializes topic weights to 1/topicCount.
	const LNKFloat topicWeight = 1.0 / _topicCount;
	LNK_vfill(&topicWeight, pi, UNIT_STRIDE, _topicCount);

	// Probability distributions for each topic.
	LNKFloat *const p = LNKFloatAlloc(_topicCount * wordCount);

	for (LNKSize topic = 0; topic < _topicCount; topic++) {
		// Initialize each topic to a random distribution.
		const LNKSize row = (LNKSize)arc4random_uniform((uint32_t)documentCount);

		LNKFloat *const pRow = &p[topic * wordCount];
		const LNKFloat *const matrixRow = &matrixBuffer[row * wordCount];

		LNKFloat additiveSmoothing = 1.0 / wordCount;
		LNK_vsadd(matrixRow, UNIT_STRIDE, &additiveSmoothing, pRow, UNIT_STRIDE, wordCount);

		LNKFloat normalizer = 0;
		LNK_vsum(pRow, UNIT_STRIDE, &normalizer, wordCount);
		LNK_vsdiv(pRow, UNIT_STRIDE, &normalizer, pRow, UNIT_STRIDE, wordCount);
	}

	LNKFloat previousLikelihood = LNKFloatMax;
	LNKFloat currentLikelihood = LNKFloatMax;
	LNKSize iteration = 0;

	const int topicCountInt = (int)_topicCount;
	const int wordCountInt = (int)wordCount;
	LNKFloat *const logWorkspace = LNKFloatAlloc(wordCount);
	LNKFloat *const sums = LNKFloatCalloc(wordCount);
	LNKFloat *const Alog = LNKFloatAlloc(_topicCount);
	LNKFloat *const wlog = LNKFloatAlloc(_topicCount);

	while (iteration < 2 || fabs(previousLikelihood - currentLikelihood) > 1000) {
		LNKFloat *const W = LNKFloatCalloc(documentCount * _topicCount);

		// Compute W
		for (LNKSize docIndex = 0; docIndex < documentCount; docIndex++) {
			for (LNKSize topic = 0; topic < _topicCount; topic++) {
				Alog[topic] = LNKLog(pi[topic]);

				LNK_vlog(logWorkspace, &p[topic * wordCount], &wordCountInt);

				LNKFloat localSum = 0;
				LNK_dotpr(&matrixBuffer[docIndex * wordCount], UNIT_STRIDE, logWorkspace, UNIT_STRIDE, &localSum, wordCount);
				Alog[topic] += localSum;
			}

			const LNKFloat logSum = LNK_vlogsumexp(Alog, _topicCount);

			LNKFloat negativeLogSum = -logSum;
			LNK_vsadd(Alog, UNIT_STRIDE, &negativeLogSum, wlog, UNIT_STRIDE, _topicCount);

			LNK_vexp(wlog, wlog, &topicCountInt);
			LNKFloatCopy(&W[docIndex * _topicCount], wlog, _topicCount);
		}

		for (LNKSize topic = 0; topic < _topicCount; topic++) {
			LNKFloat weightSum = 0;
			LNKFloat bottom = 0;

			for (LNKSize i = 0; i < documentCount; i++) {
				weightSum += W[i * _topicCount + topic];

				LNKFloat rowSum = 0;
				LNK_vsum(&matrixBuffer[i * wordCount], UNIT_STRIDE, &rowSum, wordCount);
				bottom += rowSum * W[i * _topicCount + topic];
			}

			LNK_vclr(sums, UNIT_STRIDE, wordCount);

			for (LNKSize i = 0; i < documentCount; i++) {
				LNKFloat weight = W[i * _topicCount + topic];
				LNK_vsmul(&matrixBuffer[i * wordCount], UNIT_STRIDE, &weight, logWorkspace, UNIT_STRIDE, wordCount);
				LNK_vadd(logWorkspace, UNIT_STRIDE, sums, UNIT_STRIDE, sums, UNIT_STRIDE, wordCount);
			}

			LNK_vsdiv(sums, UNIT_STRIDE, &bottom, &p[topic * wordCount], UNIT_STRIDE, wordCount);

			pi[topic] = weightSum / documentCount;
		}

		// Re-smooth
		for (LNKSize topic = 0; topic < _topicCount; topic++) {
			LNKFloat *const pRow = &p[topic * wordCount];

			LNKFloat additiveSmoothing = 1.0 / wordCount;
			LNK_vsadd(pRow, UNIT_STRIDE, &additiveSmoothing, pRow, UNIT_STRIDE, wordCount);

			LNKFloat normalizer = 0;
			LNK_vsum(pRow, UNIT_STRIDE, &normalizer, wordCount);
			LNK_vsdiv(pRow, UNIT_STRIDE, &normalizer, pRow, UNIT_STRIDE, wordCount);
		}

		previousLikelihood = currentLikelihood;
		currentLikelihood = 0;

		for (LNKSize docIndex = 0; docIndex < documentCount; docIndex++) {
			for (LNKSize topic = 0; topic < _topicCount; topic++) {
				LNKFloat sumK = 0;
				LNK_vlog(logWorkspace, &p[topic * wordCount], &wordCountInt);
				LNK_dotpr(&matrixBuffer[docIndex * wordCount], UNIT_STRIDE, logWorkspace, UNIT_STRIDE, &sumK, wordCount);
				currentLikelihood += LNKLog(pi[topic]) + sumK;
			}
		}

		iteration += 1;

		free(W);
	}

	free(pi);
	free(logWorkspace);
	free(sums);
	free(Alog);
	free(wlog);

	LNKTopicSet *const topicSet = [[LNKTopicSet alloc] initWithTopics:_topicCount];

	for (LNKSize topic = 0; topic < _topicCount; topic++) {
		Word *const words = malloc(sizeof(Word) * wordCount);

		for (LNKSize word = 0; word < wordCount; word++) {
			words[word] = (Word) { word, p[topic * wordCount + word] };
		}

		qsort_b(words, wordCount, sizeof(Word), ^int(const void *a, const void *b) {
			return ((Word *)a)->frequency > ((Word *)b)->frequency ? -1 : 1;
		});

		for (LNKSize c = 0; c < MIN((LNKSize)10, wordCount); c++) {
			NSString *const word = _vocabulary[words[c].index];
			[topicSet addWord:word forTopicAtIndex:topic];
		}

		free(words);
	}

	free(p);

	dispatch_async(dispatch_get_main_queue(), ^{
		[_delegate topicModeller:self didFindTopics:topicSet];
	});

	[topicSet release];
}

@end
