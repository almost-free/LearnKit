//
//  LNKFastQueue.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKFastFloatQueue.h"

struct _LNKFastFloatQueueBucket {
	LNKFloat value;
	struct _LNKFastFloatQueueBucket *next;
};

struct _LNKFastFloatQueue {
	struct _LNKFastFloatQueueBucket *head;
	struct _LNKFastFloatQueueBucket *tail;
	struct _LNKFastFloatQueueBucket *reuseQueue;
	LNKSize size;
	LNKSize capacity;
};

LNKFastFloatQueueRef LNKFastFloatQueueCreate(LNKSize capacity) {
	assert(capacity);
	
	// Zeros out the head and tail buckets and size for us.
	LNKFastFloatQueueRef queue = calloc(sizeof(LNKFastFloatQueue), 1);
	queue->capacity = capacity;
	
	// Zeros out the buckets' next pointers for us.
	struct _LNKFastFloatQueueBucket *freeBuckets = calloc(sizeof(struct _LNKFastFloatQueueBucket), capacity);
	
	for (LNKSize n = 0; n < capacity - 1; n++) {
		freeBuckets[n].next = &freeBuckets[n+1];
	}
	
	queue->reuseQueue = freeBuckets;
	
	return queue;
}

void LNKFastFloatQueueFree(LNKFastFloatQueueRef queue) {
	free(queue);
}

void LNKFastFloatQueueEnqueue(LNKFastFloatQueueRef queue, LNKFloat value) {
	assert(queue);
	assert(queue->size < queue->capacity);
	
	struct _LNKFastFloatQueueBucket *freeBucket = queue->reuseQueue;
	queue->reuseQueue = freeBucket->next;
	
	freeBucket->value = value;
	freeBucket->next = NULL;
	
	if (!queue->head) {
		queue->head = freeBucket;
		queue->tail = freeBucket;
	}
	else {
		queue->tail->next = freeBucket;
		queue->tail = freeBucket;
	}
	
	queue->size++;
}

LNKFloat LNKFastFloatQueueDequeue(LNKFastFloatQueueRef queue) {
	assert(queue);
	assert(queue->size);
	
	struct _LNKFastFloatQueueBucket *oldestBucket = queue->head;
	LNKFloat value = oldestBucket->value;
	
	if (queue->head == queue->tail)
		queue->tail = oldestBucket->next;
	
	queue->head = oldestBucket->next;
	
	struct _LNKFastFloatQueueBucket *currentReuseQueue = queue->reuseQueue;
	queue->reuseQueue = oldestBucket;
	oldestBucket->next = currentReuseQueue;
	
	queue->size--;
	
	return value;
}

LNKSize LNKFastFloatQueueSize(LNKFastFloatQueueRef queue) {
	assert(queue);
	return queue->size;
}

BOOL LNKFastFloatAreValuesApproximatelyClose(LNKFastFloatQueueRef queue, LNKFloat threshold) {
	struct _LNKFastFloatQueueBucket *currentBucket = queue->head;
	
	while (currentBucket && currentBucket->next) {
		if (fabs(currentBucket->next->value - currentBucket->value) > threshold)
			return NO;
		
		currentBucket = currentBucket->next;
	}
	
	return YES;
}
